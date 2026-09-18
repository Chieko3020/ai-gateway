#!/usr/bin/env python3
"""语义缓存离线评测：公开数据集上的召回 / 误命中 / F1（+ 阈值扫描 + 实体否决对照）

为什么要有这个脚本（对外指标的可信来源问题）：
  仓库里原有的 scripts/datasets/{real,synthetic}.jsonl 是**自建**数据集——
  真实那份含用户真实提问（不适合公开），合成那份的同义/非同义标注由脚本自动
  生成（客观性弱）。对外说"召回率 X%"时，来源必须是公开、可复核的数据集。
  因此本脚本以 **LCQMC**（中文同义句，C-MTEB 版）与 **PAWS-X 中文**（对抗式
  改写句对）为准；自建数据集只保留作对照（见 scripts/gen_dataset.py 头注释）。

度量口径（必须写清，否则数字没有意义）：
  把句对 (s1, s2) 当作"缓存里已有一条 s2（标注 score/label=1 表示与 s1 同义），
  现在来了一条 s1"：
    命中   = cos(s1, s2) >= threshold（且实体否决未拦下）
    召回   = 同义对里被判命中的比例        （TP / (TP+FN)）
    误命中 = 非同义对里被判命中的比例      （FP / (FP+TN)，即假阳率）
    F1     = 以"命中"为正类的精确率/召回率调和平均
  缓存场景关心的是"宁可少命中，不可答非所问"，因此**误命中率**与 F1 同等重要。

安装与版本约束：
    python3 -m venv /tmp/venv-eval && /tmp/venv-eval/bin/pip install \\
        'onnxruntime>=1.17' 'transformers>=4.35,<5' 'pyarrow>=14' 'numpy>=1.24,<3'
  必须 **transformers<5**：5.x 的 `BertTokenizer` 会静默忽略 `vocab_file` 参数
  （构造出的分词器用的是别的词表，全部变 [UNK]），评测数字会变成噪声而不是报错。
  脚本会在启动时校验版本并直接报错退出。

模型与词表：默认用仓库里的 model/model_int8.onnx + model/vocab.txt
  （工作目录需在仓库根；或用 --model/--vocab 指定）

用法：
    python3 scripts/eval_semantic_cache.py                       # 全部四组设置 + 阈值扫描
    python3 scripts/eval_semantic_cache.py --limit 3000 --datasets lcqmc
    python3 scripts/eval_semantic_cache.py --pairs 0.85:继续下一题:继续12题   # 单对查询
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# transformers 5.x 的 BertTokenizer 会静默忽略 vocab_file（见文件头），
# 因此在 import 之前把版本检查做掉，避免"跑出一个看起来正常的错数字"
try:
    import transformers
except ImportError as e:  # pragma: no cover
    print(f"[fatal] 缺少 transformers：{e}\n安装方式见本文件头注释", file=sys.stderr)
    raise SystemExit(2)

_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
if _ver >= (5, 0):
    print(
        f"[fatal] transformers=={transformers.__version__} 不受支持：5.x 的 "
        "BertTokenizer 会静默忽略 vocab_file（vocab 全部变 [UNK]），评测结果无效。\n"
        "        请安装 'transformers>=4.35,<5'（见脚本头注释）",
        file=sys.stderr,
    )
    raise SystemExit(2)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from entity_rules import entity_mismatch_text  # noqa: E402

DEFAULT_MODEL = "model/model_int8.onnx"
DEFAULT_VOCAB = "model/vocab.txt"
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
BATCH = 64


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_parquet(path: str, s1: str, s2: str, label: str, limit: int):
    import pyarrow.parquet as pq

    rows = [
        (r[s1], r[s2], int(r[label]))
        for r in pq.read_table(path).to_pylist()
    ]
    return rows[:limit] if limit > 0 else rows


def load_datasets(limit: int):
    """返回 [(名称, 正/负样本说明, pairs)]；文件缺失时给出可执行的下载提示"""
    specs = [
        ("lcqmc", os.path.join(DATASET_DIR, "lcqmc-val.parquet"),
         "sentence1", "sentence2", "score"),
        ("pawsx-zh", os.path.join(DATASET_DIR, "pawsx-zh-val.parquet"),
         "sentence1", "sentence2", "label"),
    ]
    out = []
    for name, path, s1, s2, label in specs:
        if not os.path.exists(path):
            print(f"[skip] {name}: 找不到 {path}\n"
                  f"       先运行 bash scripts/fetch_eval_datasets.sh", file=sys.stderr)
            continue
        pairs = load_parquet(path, s1, s2, label, limit)
        pos = sum(1 for p in pairs if p[2] == 1)
        out.append((name, f"{len(pairs)} 对（正 {pos} / 负 {len(pairs) - pos}）", pairs))
    return out


# ---------------------------------------------------------------------------
# 编码器：4 种设置组合（do_lower_case × pooling）
# ---------------------------------------------------------------------------
class Encoder:
    def __init__(self, model_path: str, vocab_path: str):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.vocab_path = vocab_path
        self._cache: dict[tuple[bool, str], object] = {}

    def _build(self, do_lower_case: bool, pooling: str):
        tok = BertTokenizer(vocab_file=self.vocab_path, do_lower_case=do_lower_case)

        def encode(texts: list[str]) -> np.ndarray:
            e = tok(texts, return_tensors="np", padding=True, truncation=True,
                    max_length=512)
            feeds = {
                "input_ids": e["input_ids"].astype(np.int64),
                "attention_mask": e["attention_mask"].astype(np.int64),
                "token_type_ids": e["token_type_ids"].astype(np.int64),
            }
            hidden = self.sess.run(["last_hidden_state"], feeds)[0]
            if pooling == "cls":
                # 官方 1_Pooling/config.json: pooling_mode_cls_token=true
                v = hidden[:, 0, :]
            else:
                m = e["attention_mask"].astype(np.float32)[:, :, None]
                v = (hidden * m).sum(1) / m.sum(1)
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

        return encode

    def encoder(self, do_lower_case: bool, pooling: str):
        key = (do_lower_case, pooling)
        if key not in self._cache:
            self._cache[key] = self._build(do_lower_case, pooling)
        return self._cache[key]

    def tokenizer(self, do_lower_case: bool):
        # 只为 token 序列对照（DMA/DNS 证据）用，走同一个 BertTokenizer 配置
        if not hasattr(self, "_tok_cache"):
            self._tok_cache = {}
        if do_lower_case not in self._tok_cache:
            self._tok_cache[do_lower_case] = BertTokenizer(
                vocab_file=self.vocab_path, do_lower_case=do_lower_case)
        return self._tok_cache[do_lower_case]


def cosines(enc, pairs):
    """逐对余弦（分批推理）"""
    out = []
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i + BATCH]
        texts = [x for r in chunk for x in (r[0], r[1])]
        V = enc(texts)
        for j in range(len(chunk)):
            out.append(float(np.dot(V[2 * j], V[2 * j + 1])))
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def metrics(pairs, cos, threshold: float, veto=None):
    """veto(s1, s2) -> True 表示这条命中被实体否决拦下（可选）"""
    tp = fp = tn = fn = 0
    vetoed_pos = vetoed_neg = 0
    for (s1, s2, lab), c in zip(pairs, cos):
        hit = c >= threshold
        if hit and veto is not None and veto(s1, s2):
            hit = False
            if lab == 1:
                vetoed_pos += 1
            else:
                vetoed_neg += 1
        if lab == 1 and hit:
            tp += 1
        elif lab == 1:
            fn += 1
        elif hit:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": rec,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "prec": prec, "f1": f1,
        "acc": (tp + tn) / max(1, tp + fp + tn + fn),
        "vetoed_pos": vetoed_pos, "vetoed_neg": vetoed_neg,
    }


def fmt_row(*cells, widths=(14, 10, 7, 9, 9, 9, 8, 9, 8)):
    out = []
    for c, w in zip(cells, widths):
        s = str(c)
        out.append(s.ljust(w) if not s.replace(".", "").replace("%", "").isdigit()
                   else s.rjust(w))
    return "".join(out)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--limit", type=int, default=3000,
                    help="每个数据集取多少对（0 = 全部）")
    ap.add_argument("--datasets", default="lcqmc,pawsx-zh")
    ap.add_argument("--pairs", action="append", default=[],
                    help="附加单对查询，格式 threshold:文本A:文本B（可重复）")
    ap.add_argument("--no-scan", action="store_true", help="跳过阈值扫描")
    args = ap.parse_args()

    if not os.path.exists(args.model) or not os.path.exists(args.vocab):
        print(f"[fatal] 模型/词表不存在：{args.model} / {args.vocab}\n"
              f"        请在仓库根目录运行，或用 --model/--vocab 指定",
              file=sys.stderr)
        return 2

    print(f"onnxruntime={ort.__version__}  transformers={transformers.__version__}  "
          f"numpy={np.__version__}")
    print(f"模型={args.model}  词表={args.vocab}")

    enc = Encoder(args.model, args.vocab)

    # ── 证据 0：DMA/DNS 的 token 序列（信息在分词层就丢了的直接证据）──────
    print("\n" + "=" * 100)
    print("证据 A：do_lower_case 对 token 序列的影响（`什么是DMA` vs `什么是DNS`）")
    print("=" * 100)
    for lower in (False, True):
        tok = enc.tokenizer(lower)
        ids_dma = tok("什么是DMA")["input_ids"]
        ids_dns = tok("什么是DNS")["input_ids"]
        same = ids_dma == ids_dns
        print(f"  do_lower_case={str(lower):5}  DMA ids={ids_dma}")
        print(f"  {'':17}DNS ids={ids_dns}  序列相同={same}"
              f"{'   <-- 信息在分词层已经丢失' if same else ''}")
    for lower, pool in ((False, "mean"), (True, "cls")):
        v = enc.encoder(lower, pool)
        a, b = v(["什么是DMA", "什么是DNS"])
        print(f"  {('case' if not lower else 'lower')}+{pool:4}  "
              f"余弦(什么是DMA, 什么是DNS) = {float(np.dot(a, b)):.4f}")

    datasets = load_datasets(args.limit)
    want = set(args.datasets.split(","))
    datasets = [d for d in datasets if d[0] in want]
    if not datasets:
        print("[fatal] 没有可用数据集", file=sys.stderr)
        return 2
    for name, desc, _ in datasets:
        print(f"  数据集 {name}: {desc}")

    # ── 预先算好 4 种设置的余弦（避免重复推理）────────────────────────────
    print("\n" + "=" * 100)
    print("对照实验：4 种设置（do_lower_case × pooling），阈值 0.80")
    print("=" * 100)
    settings = [(False, "mean"), (False, "cls"), (True, "mean"), (True, "cls")]
    cos_cache: dict[tuple, list[float]] = {}
    for name, _desc, pairs in datasets:
        for lower, pool in settings:
            t0 = time.time()
            cos_cache[(name, lower, pool)] = cosines(enc.encoder(lower, pool), pairs)
            if (lower, pool) == settings[0]:
                print(f"  [{name}] 推理完成 {len(pairs)} 对，"
                      f"{time.time() - t0:.1f}s/组")

    hdr = fmt_row("设置", "数据集", "阈值", "召回", "误命中", "精确率", "F1", "准确率", "备注")
    print("\n" + hdr)
    print("-" * len(hdr))
    for lower, pool in settings:
        tag = f"{'lower' if lower else 'case'}+{pool}"
        for name, _desc, pairs in datasets:
            m = metrics(pairs, cos_cache[(name, lower, pool)], 0.80)
            note = "当前默认" if (lower, pool) == (False, "mean") else (
                "官方设置" if (lower, pool) == (True, "cls") else "")
            print(fmt_row(tag, name, "0.80", pct(m["recall"]), pct(m["fpr"]),
                          pct(m["prec"]), pct(m["f1"]), pct(m["acc"]), note))

    # ── 阈值扫描（只看 lower+cls，即对齐官方后的设置）────────────────────
    if not args.no_scan:
        print("\n" + "=" * 100)
        print("阈值扫描（lower+cls）：0.85 是本轮选定的默认值")
        print("=" * 100)
        print(fmt_row("设置", "数据集", "阈值", "召回", "误命中", "精确率", "F1", "准确率", "备注"))
        print("-" * len(hdr))
        for th in (0.80, 0.82, 0.85, 0.88, 0.90, 0.93, 0.95):
            for name, _desc, pairs in datasets:
                m = metrics(pairs, cos_cache[(name, True, "cls")], th)
                note = "本轮默认" if th == 0.85 else ""
                print(fmt_row("lower+cls", name, f"{th:.2f}", pct(m["recall"]),
                              pct(m["fpr"]), pct(m["prec"]), pct(m["f1"]),
                              pct(m["acc"]), note))

    # ── 实体否决对照：before / after ────────────────────────────────────
    print("\n" + "=" * 100)
    print("实体一致性否决对照（lower+cls, 阈值 0.85）")
    print("  before = 纯阈值判定；after = 阈值 + 实体不对称否决")
    print("=" * 100)
    print(fmt_row("设置", "数据集", "阈值", "召回", "误命中", "精确率", "F1", "准确率", "备注"))
    print("-" * len(hdr))
    summary = {}
    for name, _desc, pairs in datasets:
        cos = cos_cache[(name, True, "cls")]
        before = metrics(pairs, cos, 0.85)
        after = metrics(pairs, cos, 0.85, veto=entity_mismatch_text)
        summary[name] = (before, after)
        print(fmt_row("before", name, "0.85", pct(before["recall"]), pct(before["fpr"]),
                      pct(before["prec"]), pct(before["f1"]), pct(before["acc"]),
                      "阈值 only"))
        print(fmt_row("after", name, "0.85", pct(after["recall"]), pct(after["fpr"]),
                      pct(after["prec"]), pct(after["f1"]), pct(after["acc"]),
                      "阈值+实体否决"))
        print(f"      被否决的命中：同义对 {after['vetoed_pos']}（代价=少命中）、"
              f"非同义对 {after['vetoed_neg']}（收益=少答错）")

    # ── 端到端 before/after（"当前 → 本轮修复后"）────────────────────────
    print("\n" + "=" * 100)
    print("端到端 before / after")
    print("  before = case+mean, 阈值 0.80（本轮修复前的默认）")
    print("  after  = lower+cls, 阈值 0.85 + 实体否决（本轮修复后的默认）")
    print("=" * 100)
    print(fmt_row("设置", "数据集", "阈值", "召回", "误命中", "精确率", "F1", "准确率", "备注"))
    print("-" * len(hdr))
    for name, _desc, pairs in datasets:
        b = metrics(pairs, cos_cache[(name, False, "mean")], 0.80)
        a = metrics(pairs, cos_cache[(name, True, "cls")], 0.85,
                    veto=entity_mismatch_text)
        print(fmt_row("before", name, "0.80", pct(b["recall"]), pct(b["fpr"]),
                      pct(b["prec"]), pct(b["f1"]), pct(b["acc"]), "case+mean"))
        print(fmt_row("after", name, "0.85", pct(a["recall"]), pct(a["fpr"]),
                      pct(a["prec"]), pct(a["f1"]), pct(a["acc"]), "lower+cls+veto"))
        print(f"      Δ 召回 {pct(a['recall'] - b['recall'])}  "
              f"Δ 误命中 {pct(a['fpr'] - b['fpr'])}  Δ F1 {pct(a['f1'] - b['f1'])}")

    # ── 机器可读输出（便于贴进简报/CI）────────────────────────────────
    print("\nJSON_SUMMARY " + json.dumps({
        "limit": args.limit,
        "datasets": {k: {
            "before_case_mean_080": v[0],
            "after_lower_cls_085_veto": v[1],
        } for k, v in summary.items()},
    }, ensure_ascii=False))

    # ── 附加单对查询（人工反例）────────────────────────────────────────
    if args.pairs:
        print("\n" + "=" * 100)
        print("附加单对查询")
        print("=" * 100)
        for spec in args.pairs:
            try:
                th_s, a, b = spec.split(":", 2)
                th = float(th_s)
            except ValueError:
                print(f"  [skip] 格式应为 threshold:A:B，收到 {spec!r}")
                continue
            for lower, pool in settings:
                v = enc.encoder(lower, pool)
                va, vb = v([a, b])
                c = float(np.dot(va, vb))
                veto = entity_mismatch_text(a, b)
                print(f"  {('lower' if lower else 'case')}+{pool:4} "
                      f"cos({a!r},{b!r})={c:.4f}  >= {th:.2f}? {c >= th}"
                      f"  实体否决={'是' if veto else '否'}"
                      f"  最终命中={'否' if (veto or c < th) else '是'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
