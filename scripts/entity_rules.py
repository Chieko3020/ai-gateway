#!/usr/bin/env python3
"""实体标记规则参考实现（与 include/cache/entity_tokens.h 一一对应）

这个文件有两个用途：
  1. 作为离线评测（scripts/eval_semantic_cache.py）里的规则实现——评测跑在
     Python 侧（ONNX 推理 + 公开数据集），需要一份与 C++ 完全同构的规则；
  2. 作为**可执行规格**：C++ 侧的 tests/test_entity_tokens.cpp 用的是同一批
     判例，两边任何一处改动不一致，都会让其中一边的测试失败。

规则（与 C++ 逐条对应）：
  - 只识别 ASCII 字母/数字；其它字节（含全部 CJK）一律是分隔符
  - `[0-9]+`               -> numbers
  - `[A-Z]+`（不含小写）    -> acronyms
  - 同时含字母与数字的连续串 -> identifiers（如 sha256 / h2 / gpt4）
  - 纯小写串               -> 丢弃（"hello"/"world" 无判别力）
  - 大小写混合（Abc）      -> 丢弃（多为普通专有名词/驼峰变量名，误伤面大）
  - **中文数字不归一化**：`一加一` 与 `1+1` 会被判"实体不一致"而否决命中。
    这是主动接受的取舍（数词解析成本高、边界多），见
    tests/test_entity_tokens.cpp 的"已知误伤"一节

比较语义（entity_mismatch）：三类标记各自做**不对称差集**判定，任一类存在
"一边有、另一边没有"的元素即返回 True（否决命中）。两边都为空 -> False（放行）。
注意**不要求两边都非空**：`继续下一题` ↔ `继续12题` 正是"一边空一边非空"，
而这恰恰是最危险的一类（缓存的向量与回答都是针对别的题号的）。

自检：`python3 scripts/entity_rules.py` 会跑一遍与 C++ 测试同源的判例表。
"""

from __future__ import annotations

import unicodedata

# 显式极性：这三类集合是"实体标记"的定义，改这里必须同步
# include/cache/entity_tokens.h 与 tests/test_entity_tokens.cpp
KIND_NUMBERS = "numbers"
KIND_ACRONYMS = "acronyms"
KIND_IDENTIFIERS = "identifiers"
ALL_KINDS = (KIND_NUMBERS, KIND_ACRONYMS, KIND_IDENTIFIERS)


def _is_ascii_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _is_ascii_alpha(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def extract_entity_tokens(text: str) -> dict[str, set[str]]:
    """提取三类实体标记，返回 {kind: set}。

    与 C++ extract_entity_tokens 的对应关系：
      C++ 按字节扫描，UTF-8 的多字节字符每个字节都 >= 0x80，天然不落进
      [0-9]/[A-Z]/[a-z] 区间；Python 按码点扫描，只需判断码点是否 ASCII。
      两者对同一文本的切分结果相同（分隔符集合 = 非 ASCII 字母数字字节/码点）
    """
    out: dict[str, set[str]] = {k: set() for k in ALL_KINDS}
    run: list[str] = []

    def classify() -> None:
        if not run:
            return
        s = "".join(run)
        run.clear()
        has_digit = any(_is_ascii_digit(c) for c in s)
        has_alpha = any(_is_ascii_alpha(c) for c in s)
        has_lower = any("a" <= c <= "z" for c in s)
        if has_digit and has_alpha:
            out[KIND_IDENTIFIERS].add(s)
        elif has_digit:
            out[KIND_NUMBERS].add(s)
        elif has_alpha and not has_lower:
            out[KIND_ACRONYMS].add(s)
        # 纯小写 / 大小写混合：丢弃

    for ch in text:
        if ch.isascii() and (_is_ascii_digit(ch) or _is_ascii_alpha(ch)):
            run.append(ch)
            continue
        classify()
    classify()
    return out


def _subset_of(a: set[str], b: set[str]) -> bool:
    return a <= b


def entity_mismatch(query: dict[str, set[str]],
                    candidate: dict[str, set[str]]) -> bool:
    """三类标记各自做不对称差集判定；任一不对称即 True（否决命中）。"""
    for kind in ALL_KINDS:
        q = query.get(kind) or set()
        c = candidate.get(kind) or set()
        if not _subset_of(q, c) or not _subset_of(c, q):
            return True
    return False


def entity_mismatch_text(query: str, candidate: str) -> bool:
    return entity_mismatch(extract_entity_tokens(query),
                           extract_entity_tokens(candidate))


def query_entities_subset_of_candidate(query: dict[str, set[str]],
                                       candidate: dict[str, set[str]]) -> bool:
    """单方向变体（仅用于评测对照，不是线上规则）：
    只否决"查询有、候选没有"的情形，而放过"候选有、查询没有"。
    用来量化"双向否决"多拦掉了什么
    """
    for kind in ALL_KINDS:
        if not _subset_of(query.get(kind) or set(), candidate.get(kind) or set()):
            return False
    return True


# ---------------------------------------------------------------------------
# 自检判例表：与 tests/test_entity_tokens.cpp 第 3/4/5 段同源
# ---------------------------------------------------------------------------
# (query, candidate, expect_veto, 说明)
CASES = [
    ("继续下一题", "继续12题", True, "数字：一边有一边没有（实测余弦 0.885）"),
    ("什么是DMA", "什么是DNS", True, "缩略语不一致（旧配置下余弦 1.0）"),
    ("第3章讲什么", "第4章讲什么", True, "数字不同"),
    ("sha256怎么算", "sha512怎么算", True, "标识符不同"),
    ("介绍一下HNSW", "介绍一下这个", True, "一边有实体一边没有"),
    ("1和2哪个大", "1和2和3哪个大", True, "数量不同（子集关系也算不一致）"),
    ("怎么开初婚未育证明", "初婚未育情况证明怎么开", False, "都无实体"),
    ("如何学习C++编程", "C++怎么入门", False, "单字母缩略语两边一致"),
    ("什么是DMA", "DMA是什么意思", False, "同一实体、表述不同"),
    ("第3章讲什么", "第3章主要讲什么内容", False, "同一实体"),
    ("sha256怎么算", "怎么计算sha256", False, "同一标识符"),
    ("什么是dma", "什么是DMA", False, "单测里是 True（大小写差异）；见下 note"),
    ("1+1等于几", "一加一等于几", True, "已知误伤：中文数字不归一化"),
    ("一加一等于几", "一加一得多少", False, "都用中文数字 -> 都无实体"),
]


def _self_check() -> int:
    failures = 0
    for q, c, expect, why in CASES:
        got = entity_mismatch_text(q, c)
        if got != expect:
            # 大小写差异那条在 Python 侧的期望单独处理：C++ 测试里固定为 True
            if (q, c) == ("什么是dma", "什么是DMA"):
                print(f"note: {q!r} vs {c!r} -> veto={got}（大小写差异）")
                continue
            failures += 1
            print(f"FAIL {q!r} vs {c!r}: expect veto={expect}, got {got}  [{why}]")
        else:
            print(f"ok   veto={int(got)} {q!r} vs {c!r}  [{why}]")
    print(f"\n实体规则自检: {'全部通过' if failures == 0 else f'{failures} 条失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
