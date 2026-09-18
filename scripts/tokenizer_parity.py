#!/usr/bin/env python3
"""分词器批量一致性核对：本项目的 BERT WordPiece 实现 vs HuggingFace 官方实现

做法：
  1. 取样本（默认：仓库压测数据集的全部提问 + 300 条定种子随机串）
  2. 用官方 BertTokenizer（transformers，slow 与 fast 双向自校）算 id
  3. 用 `tests/test_tokenizer --dump` 算本项目的 id
  4. 逐条比对，打印一致率与前若干条不一致样本

注意 truncation：本项目的 `encode(text, max_len)` 默认截断到 512，
因此这里给 dump 端传 `--max-len 100000` 关掉截断，和官方"不截断"逐条对齐；
截断语义本身由 tests/test_tokenizer 的断言覆盖。

用法（需要 transformers 4.x，见 scripts/tokenizer_reference.py 的说明）:
  python3 scripts/tokenizer_parity.py --vocab model/vocab.txt \
      --dump-bin build-fix/dbg2/tests/test_tokenizer
"""

import argparse
import json
import random
import subprocess
import sys

# 随机样本用的字符/词片段：覆盖大小写、连写、标点、汉字、全角、数字
WORDS = ["hello", "world", "Hello", "World", "unhappy", "helloworld", "tripadvisor",
         "bge", "small", "zh", "test", "example", "com", "ai", "AI", "2024",
         "the", "The", "of", "cache", "Gateway", "语义", "缓存", "网关", "向量",
         "检索", "命中", "你好", "世界", "人工智能", "北京"]
PUNCT = ["", " ", "  ", "-", "_", ".", ",", "!", "?", "@", ":", "，", "。", "、", "！"]
CJK = list("的一是不了在人有我他这中大来上国个到说们为子和你地出道也时年得就那要下"
           "以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想"
           "看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身"
           "些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女问力机给等"
           "几很业最间新什打便位因重被走电四第门相次东政海口使教西再平真听世气信北少关"
           "并内数流每")
SAMPLES_PER_LEN = 4


def official_ids(vocab_path, texts, lower):
    from transformers import BertTokenizer, BertTokenizerFast
    kw = dict(vocab_file=vocab_path, do_lower_case=lower, do_basic_tokenize=True,
              tokenize_chinese_chars=True, strip_accents=None, unk_token="[UNK]",
              cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
              mask_token="[MASK]")
    slow = BertTokenizer(**kw)
    fast = BertTokenizerFast(**kw)
    slow_ids = [slow.encode(t, add_special_tokens=True) for t in texts]
    fast_ids = [fast.encode(t, add_special_tokens=True) for t in texts]
    disagree = [i for i, (a, b) in enumerate(zip(slow_ids, fast_ids)) if a != b]
    if disagree:
        print(f"!! 官方 slow/fast 自身不一致的样本数：{len(disagree)}（前 3 条：{disagree[:3]}）")
    return slow_ids


def project_ids(dump_bin, texts, lower, vocab_path):
    payload = "".join("hex:" + t.encode("utf-8").hex() + "\n" for t in texts)
    cmd = [dump_bin, "--dump", "--max-len", "100000"]
    if lower:
        cmd.append("--lowercase")
    r = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                       cwd=".")
    if r.returncode != 0:
        print(f"dump 失败（exit={r.returncode}）：{r.stderr[:400]}", file=sys.stderr)
        sys.exit(2)
    out = []
    for line in r.stdout.splitlines():
        out.append([int(x) for x in line.split()] if line.strip() else [])
    if len(out) != len(texts):
        print(f"dump 行数不匹配：{len(out)} != {len(texts)}", file=sys.stderr)
        sys.exit(2)
    return out


def gen_random(n, seed=20260918):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        pieces = []
        for _ in range(rnd.randint(1, 6)):
            if rnd.random() < 0.25:
                pieces.append("".join(rnd.choice(CJK) for _ in range(rnd.randint(1, 4))))
            else:
                w = rnd.choice(WORDS)
                if rnd.random() < 0.3:
                    w = w.upper() if rnd.random() < 0.5 else w.capitalize()
                pieces.append(w)
            pieces.append(rnd.choice(PUNCT))
        text = "".join(pieces)
        if rnd.random() < 0.2:
            text = text[: max(1, len(text) // 2)] + str(rnd.randint(0, 99999))
        out.append(text)
    return out


def dataset_texts(paths, limit=None):
    out = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        out.append(obj.get("q", "") if isinstance(obj, dict) else str(obj))
                    except Exception:
                        out.append(line)
        except FileNotFoundError:
            print(f"（跳过不存在的数据集 {p}）")
    return out[:limit] if limit else out


def compare(name, texts, gold, got, show=5):
    bad = []
    for t, a, b in zip(texts, gold, got):
        if a != b:
            bad.append((t, a, b))
    rate = (len(texts) - len(bad)) / len(texts) * 100 if texts else 0.0
    print(f"\n[{name}] 样本 {len(texts)} 条，一致 {len(texts) - len(bad)} 条 "
          f"（{rate:.2f}%），不一致 {len(bad)} 条")
    for t, a, b in bad[:show]:
        shown = t if len(t) <= 48 else t[:45] + "..."
        print(f"  - {shown!r}\n      官方: {a[:14]}{'...' if len(a) > 14 else ''}\n"
              f"      本项目: {b[:14]}{'...' if len(b) > 14 else ''}")
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="model/vocab.txt")
    ap.add_argument("--dump-bin", required=True, help="tests/test_tokenizer 可执行文件")
    ap.add_argument("--n", type=int, default=300, help="随机样本条数")
    ap.add_argument("--dataset", action="append", default=None,
                    help="额外数据集（可多次）；默认用仓库两份压测数据集")
    ap.add_argument("--limit", type=int, default=0, help="数据集截断条数（0=全部）")
    args = ap.parse_args()

    datasets = args.dataset if args.dataset else [
        "scripts/datasets/synthetic.jsonl", "scripts/datasets/real.jsonl"]
    corpus = dataset_texts(datasets, args.limit or None)
    texts = corpus + gen_random(args.n)
    # 去重且保序（重复样本没有信息量）
    seen, uniq = set(), []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    texts = uniq
    print(f"样本合计 {len(texts)} 条（数据集 {len(corpus)} + 随机 {args.n}，去重后）")

    total_bad = 0
    for lower in (False, True):
        gold = official_ids(args.vocab, texts, lower)
        got = project_ids(args.dump_bin, texts, lower, args.vocab)
        mode = "do_lower_case=true" if lower else "do_lower_case=false（官方默认）"
        total_bad += compare(mode, texts, gold, got)

    print(f"\n合计不一致：{total_bad} 条")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
