#!/usr/bin/env python3
"""分词器对照脚本：用官方 BERT WordPiece 实现生成参考 token id

用途：`tests/test_tokenizer.cpp` 里的"金标准"id 表由本脚本生成，避免"自己写规则
自己对齐"的循环论证。官方词表与项目 `model/vocab.txt` 逐字节相同
（md5 3b5b76c4aef48ecf8cb3abaafe960f09，= BAAI/bge-small-zh-v1.5 = google-bert/bert-base-chinese），
官方 `tokenizer_config.json`（取自 huggingface.co，见下）为：

    {"do_lower_case": false, "do_basic_tokenize": true, "tokenize_chinese_chars": true,
     "strip_accents": null, "never_split": null, "unk_token": "[UNK]",
     "cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]", "model_max_length": 512}

注意 `do_lower_case: false`：官方对本模型的默认行为**不做**小写化，因此 "Hello" 在
官方分词器下同样是 [UNK]（词表里只有 "the"，没有 "The"）。项目侧对应开关
`do_lower_case` 的默认值也与官方一致（false），需要时可在配置/调用处打开。

用法:
  pip install "transformers<5" tokenizers
  python3 scripts/tokenizer_reference.py --vocab model/vocab.txt            # 人读表
  python3 scripts/tokenizer_reference.py --vocab model/vocab.txt --emit-cpp # 生成测试用表

注意必须用 transformers 4.x：5.x 的 `BertTokenizer` 不再有纯 Python 慢速实现
（没有 `basic_tokenizer`/`wordpiece_tokenizer`），且会**静默忽略** `vocab_file`
参数（词表只剩 5 个 special token，任何输入都变成 [UNK]），那样的"对照"毫无意义。
本脚本同时用 BertTokenizerFast（Rust 实现）交叉验证，两者必须逐条一致。
"""

import argparse

# 与 tests/test_tokenizer.cpp 保持同一批样本：每条样本都对应一条 WordPiece 规则
SAMPLES = [
    "hello world",              # 词表内单词 + 空格切分
    "Hello World",              # 大小写：官方 do_lower_case=false 保留原样
    "helloworld",               # 连写：靠 ## 续接子词切分
    "unhappy",                  # 前缀子词
    "xxhello",                  # 整词 OOV：官方整词一个 [UNK]，不逐字符回退
    "你好，世界",                # 中文逐字 + 中文标点
    "中国的首都是北京",           # 中文：tokenize_chinese_chars 让每个汉字独立成词
    "人工智能 AI 2024",          # 中英数混排
    "3.14159",                  # 数字与 ASCII 标点切分
    "test@example.com",         # 标点切分（@ 与 .）
    "bge-small-zh-v1.5",        # 连字符
    "tripadvisor",              # 11 字符的长词条：整词在词表里，最长匹配必须一次命中
    "ab한",                     # 前半段能匹配、后半段不行 -> 整词 [UNK]（不逐字符回退）
    "한국어",                    # 整词 OOV -> 单个 [UNK]
    "a" * 120,                  # 超过 max_input_chars_per_word(100) -> [UNK]
    "café",                     # 重音字符：官方 strip_accents=null（= 不做去音标）
    "   ",                      # 纯空白：只剩 [CLS][SEP]
]


def build_tokenizers(vocab_path):
    from transformers import BertTokenizer, BertTokenizerFast
    slow = BertTokenizer(vocab_file=vocab_path, do_lower_case=False,
                         do_basic_tokenize=True, tokenize_chinese_chars=True,
                         strip_accents=None, unk_token="[UNK]", cls_token="[CLS]",
                         sep_token="[SEP]", pad_token="[PAD]", mask_token="[MASK]")
    slow_lower = BertTokenizer(vocab_file=vocab_path, do_lower_case=True,
                               do_basic_tokenize=True, tokenize_chinese_chars=True,
                               strip_accents=None, unk_token="[UNK]",
                               cls_token="[CLS]", sep_token="[SEP]",
                               pad_token="[PAD]", mask_token="[MASK]")
    try:
        fast = BertTokenizerFast(vocab_file=vocab_path, do_lower_case=False,
                                 tokenize_chinese_chars=True, strip_accents=None,
                                 unk_token="[UNK]", cls_token="[CLS]",
                                 sep_token="[SEP]", pad_token="[PAD]",
                                 mask_token="[MASK]")
    except Exception as e:  # fast 需要 tokenizers 后端；缺了就只用 slow
        print(f"# BertTokenizerFast 不可用（{e}），仅用 slow 对照")
        fast = None
    return slow, slow_lower, fast


def emit_punctuation_table():
    """打印官方 _is_punctuation 用到的 Unicode 标点码点区间（C++ 表）

    官方实现是 `unicodedata.category(char).startswith("P")`，因此这里用同一个
    Python/unicodedata 把"类别为 P*"的码点合并成连续区间，供 C++ 侧二分查找。
    """
    import unicodedata
    ranges = []
    for cp in range(0x110000):
        if unicodedata.category(chr(cp)).startswith("P"):
            if ranges and cp == ranges[-1][1] + 1:
                ranges[-1][1] = cp
            else:
                ranges.append([cp, cp])
    total = sum(b - a + 1 for a, b in ranges)
    print(f"// unicodedata {unicodedata.unidata_version}：P* 类别共 {total} 个码点，"
          f"合并为 {len(ranges)} 个区间")
    print("constexpr CodeRange kPunctRanges[] = {")
    line = "   "
    for a, b in ranges:
        item = " {0x%X, 0x%X}," % (a, b)
        if len(line) + len(item) > 96:
            print(line)
            line = "   "
        line += item
    if line.strip():
        print(line)
    print("};")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="model/vocab.txt")
    ap.add_argument("--emit-cpp", action="store_true",
                    help="输出可直接粘进 tests/test_tokenizer.cpp 的表")
    ap.add_argument("--emit-punctuation", action="store_true",
                    help="输出 Unicode 标点区间表（粘进 src/cache/onnx_embedding.cpp）")
    args = ap.parse_args()

    if args.emit_punctuation:
        emit_punctuation_table()
        return 0

    slow, slow_lower, fast = build_tokenizers(args.vocab)

    rows = []
    for text in SAMPLES:
        ids = slow.encode(text, add_special_tokens=True)
        ids_lower = slow_lower.encode(text, add_special_tokens=True)
        ids_fast = fast.encode(text, add_special_tokens=True) if fast else None
        agree = (ids_fast is None) or (ids == ids_fast)
        rows.append((text, ids, ids_lower, agree, slow.tokenize(text),
                     slow_lower.tokenize(text)))

    if args.emit_cpp:
        print("// 由 scripts/tokenizer_reference.py --emit-cpp 生成")
        for text, ids, ids_lower, agree, _, _ in rows:
            lit = text if len(text) <= 40 else text[:8] + "…(len=%d)" % len(text)
            if len(text) > 40:
                # 长样本（>100 字符）在测试里用 std::string(n, 'a') 构造
                lit = f'"a" * {len(text)}'
            ids_s = ", ".join(str(i) for i in ids)
            low_s = ", ".join(str(i) for i in ids_lower)
            print(f'  {{"{lit}", {{{ids_s}}}, {{{low_s}}}}},')
        return 0

    print(f"{'样本':<24} {'piece 切分（官方，cased）':<44} ids")
    for text, ids, ids_lower, agree, pieces, pieces_lower in rows:
        shown = text if len(text) <= 20 else text[:17] + "..."
        print(f"{shown:<24} {'|'.join(pieces):<44} {ids}")
        print(f"{'':<24} {'(lowercase=true) ' + '|'.join(pieces_lower):<44} "
              f"{ids_lower}")
        if not agree:
            print(f"{'':<24} !! slow 与 fast 不一致：{ids_lower}")
    mismatched = [t for t, _, _, agree, _, _ in rows if not agree]
    print(f"\nslow/fast 一致：{len(rows) - len(mismatched)}/{len(rows)}"
          + (f"，不一致样本：{mismatched}" if mismatched else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
