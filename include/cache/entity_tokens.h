// 实体标记提取：语义缓存的"实体一致性否决"用到的可判别成分
//
// 动机（实测反例）：余弦相似度对"只差一个实体"的句子几乎无判别力。
//   - `继续下一题` ↔ `继续12题`     余弦 0.885（旧阈值 0.80 下命中，答非所问）
//   - `什么是DMA`  ↔ `什么是DNS`    旧配置（do_lower_case=false）下余弦 1.0000
// 两者的差别**全部**落在"数字"与"大写缩略语"上，句子的其余部分完全同义。
// 提高阈值救不了：把阈值提到 0.90 时 DMA/DNS 仍是 1.0，而正常同义句的召回已经
// 从 96.3% 掉到 72.1%（见 scripts/eval_semantic_cache.py 的阈值扫描）。
// 因此在相似度之外单加一条**硬约束**：查询与候选在这三类标记上不对称时否决命中。
//
// 提取三类标记（与 scripts/entity_rules.py 的实现一一对应）：
//   1. 数字串            连续 ASCII 数字
//   2. 大写缩略语        连续 ASCII 大写字母（≥1 位）
//   3. 混合标识符        字母数字混合（如 h2、sha256、gpt4、utf8）
//
// 明确排除：**中文数字**。`1+1等于几` 与 `一加一等于几` 是同一问题的两种写法，
// 但把中文数字归一化成阿拉伯数字需要一整套数词解析（"十二"/"两百零三"/"一万二"），
// 本规则不做这件事，因此这类等价表达**仍会被否决**（宁可少命中，不可答非所问）。
// 该取舍已写进简报与测试用例（见 tests/test_entity_tokens.cpp 的"已知误伤"一节）。
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_set>

namespace ai_gateway {

// 一条文本里提取到的实体标记。三类分开存：同一个字符串出现在不同类型里
// 不会互相干扰（"123" 是数字，"ABC" 是缩略语），比较时逐类做不对称判定。
struct EntityTokens {
  std::unordered_set<std::string> numbers;      // [0-9]+
  std::unordered_set<std::string> acronyms;     // [A-Z]+
  std::unordered_set<std::string> identifiers;  // 同时含字母与数字的连续串

  bool empty() const {
    return numbers.empty() && acronyms.empty() && identifiers.empty();
  }
};

// 从 UTF-8 文本提取实体标记。只识别 ASCII 字母/数字，其它字节（含全部 CJK）
// 一律作为"非字母数字"处理，即标记的自然分隔符。
EntityTokens extract_entity_tokens(std::string_view text);

// 两条文本的实体标记是否存在**不对称差集**：
//   查询里有、候选里没有（或反之）→ true，表示"实体不一致"，应当否决命中。
//
// 为什么用不对称差集而不是"集合相等"：
//   - 集合在两边都为空时相等（普通同义句，应当放行）——不对称差集同样放行
//   - "一边空、一边非空"是最危险的情形（`继续下一题` 命中 `继续12题` 的答案：
//     缓存里那条记录的向量与回答都是针对**别的题号**的），集合相等判定会放过它，
//     不对称差集能拦住。这正是本规则要覆盖的场景，因此**不做**"两边都必须非空"
//     的前置条件（早先版本加了 `!query.empty()` 守卫，会让数字类反例整类失效）
bool entity_mismatch(const EntityTokens& query, const EntityTokens& candidate);

}  // namespace ai_gateway
