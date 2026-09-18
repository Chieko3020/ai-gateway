// 实体一致性否决（entity veto）回归
//
// 动机是实测反例，不是"理论上可能有"：
//   - `继续下一题` ↔ `继续12题`    余弦 0.885（旧阈值 0.80 下命中，答非所问）
//   - `什么是DMA`  ↔ `什么是DNS`   旧配置（do_lower_case=false）下余弦 1.0000
// 两者的差别**全部**落在数字与大写缩略语上，句子其余部分完全同义；阈值扫描救不了
// （0.90 时 DMA/DNS 仍是 1.0，而正常同义句召回已从 96.3% 掉到 72.1%）。
// 因此在相似度之外加一条硬约束：实体标记不对称即否决。
//
// 判别力说明（回退修复即失败）：
//   - 把 extract_entity_tokens 里的 classify() 改成"什么都不做"：第 1/2 段的
//     提取断言全挂
//   - 把 entity_mismatch 改成 `return false`：第 3/4 段的反例断言全挂
//   - 把子词集比较改成"只比较数字"：第 3 段的缩略语反例与第 4 段的标识符断言挂
//
// "已知误伤"一节固定的是**主动接受的取舍**（中文数字不归一化），不是 bug：
// 它在 LCQMC 上的代价见 scripts/eval_semantic_cache.py 的实体否决前后对照
#include <string>
#include <vector>

#include "cache/entity_tokens.h"
#include "test_check.h"

using namespace ai_gateway;

namespace {

// 判定一对文本是否会被否决（= entity_mismatch 为真）
bool vetoed(const std::string& a, const std::string& b) {
  return entity_mismatch(extract_entity_tokens(a), extract_entity_tokens(b));
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1. 提取：三类标记 ────────────────────────────────────────────────
  {
    auto t = extract_entity_tokens("继续12题");
    CHECK(t.numbers.size() == 1 && t.numbers.count("12") == 1); ok++;
    CHECK(t.acronyms.empty()); ok++;
    CHECK(t.identifiers.empty()); ok++;

    auto t2 = extract_entity_tokens("什么是DMA");
    CHECK(t2.acronyms.size() == 1 && t2.acronyms.count("DMA") == 1); ok++;
    CHECK(t2.numbers.empty()); ok++;
    // 汉字是 CJK，不是 ASCII 字母，因此不进任何一类
    CHECK(!t2.empty()); ok++;

    // 混合标识符：字母+数字
    auto t3 = extract_entity_tokens("用sha256校验");
    CHECK(t3.identifiers.size() == 1 && t3.identifiers.count("sha256") == 1); ok++;
    CHECK(t3.numbers.empty()); ok++;
    CHECK(t3.acronyms.empty()); ok++;

    // 纯小写普通词没有判别力，不入任何一类（否则"hello"与"world"会互相否决）
    auto t4 = extract_entity_tokens("hello world foo bar");
    CHECK(t4.empty()); ok++;

    // 小写字母 + 数字 = 标识符（h2 / gpt4），不是数字
    auto t5 = extract_entity_tokens("h2数据库");
    CHECK(t5.identifiers.count("h2") == 1); ok++;
    CHECK(t5.numbers.empty()); ok++;

    // 大小写混合（Abc）既不是缩略语也不是纯小写 -> 不入任何一类。
    // 这是刻意的：这类词多是普通专有名词/驼峰变量名，判定"实体不一致"误伤面大
    auto t6 = extract_entity_tokens("Abc");
    CHECK(t6.empty()); ok++;

    // 多个标记 + 重复标记去重
    auto t7 = extract_entity_tokens("A和B和A，12与12");
    CHECK(t7.acronyms.size() == 2); ok++;
    CHECK(t7.numbers.size() == 1); ok++;

    // 数字串必须连续：分隔符（中英文标点/空白）会切断
    auto t8 = extract_entity_tokens("12 34");
    CHECK(t8.numbers.size() == 2); ok++;
    CHECK(t8.numbers.count("12") == 1 && t8.numbers.count("34") == 1); ok++;
  }

  // ── 2. 提取：空文本与非 ASCII ────────────────────────────────────────
  {
    CHECK(extract_entity_tokens("").empty()); ok++;
    CHECK(extract_entity_tokens("完全中文没有问题").empty()); ok++;
    // 中文数字是汉字，提取不到 —— 这就是"已知误伤"的根因（见第 5 段）
    CHECK(extract_entity_tokens("十二").empty()); ok++;
    // 全角数字（U+FF11）也是非 ASCII，同样提取不到
    CHECK(extract_entity_tokens("１２").empty()); ok++;
  }

  // ── 3. 否决判定：实测反例必须被拦住 ─────────────────────────────────
  {
    // 反例 1：数字不一致（余弦 0.885）
    CHECK(vetoed("继续下一题", "继续12题")); ok++;
    CHECK(vetoed("继续12题", "继续下一题")); ok++;  // 对称
    // 反例 2：大写缩略语不一致
    CHECK(vetoed("什么是DMA", "什么是DNS")); ok++;
    CHECK(vetoed("什么是DNS", "什么是DMA")); ok++;
    // 反例 3：数字不同
    CHECK(vetoed("第3章讲什么", "第4章讲什么")); ok++;
    // 反例 4：标识符不同
    CHECK(vetoed("sha256怎么算", "sha512怎么算")); ok++;
    // 反例 5：一边有实体、另一边完全没有（"下一题"这类无实体文本）
    CHECK(vetoed("介绍一下HNSW", "介绍一下这个")); ok++;
    // 反例 6：数量不同（子集关系也算不一致 —— 缺一个数字就是缺信息）
    CHECK(vetoed("1和2哪个大", "1和2和3哪个大")); ok++;
  }

  // ── 4. 不得误伤：实体一致/都无实体时必须放行 ────────────────────────
  {
    // 都无实体：普通同义句
    CHECK(!vetoed("怎么开初婚未育证明", "初婚未育情况证明怎么开")); ok++;
    CHECK(!vetoed("如何学习C++编程", "C++怎么入门")); ok++;
    // C++ 里的 "C" 是单个大写字母，两边都提取到 -> 一致，不否决
    CHECK(!vetoed("C和Python哪个好", "Python和C哪个好")); ok++;
    // 同一实体、表述不同
    CHECK(!vetoed("什么是DMA", "DMA是什么意思")); ok++;
    CHECK(!vetoed("什么是DMA", "请解释DMA")); ok++;
    CHECK(!vetoed("第3章讲什么", "第3章主要讲什么内容")); ok++;
    CHECK(!vetoed("sha256怎么算", "怎么计算sha256")); ok++;
    // 大小写不同：小写不在任何一类里，因此 `dma` vs `DMA` 会被视为
    // "一边有、一边没有" -> 否决。这是刻意的保守方向：
    // 分词层已按 do_lower_case=true 处理，文本层的大小写差异不影响向量，
    // 但"实体名的大小写写错"本身就值得人工确认，宁可少命中
    CHECK(vetoed("什么是dma", "什么是DMA")); ok++;
  }

  // ── 5. 已知误伤（主动接受的取舍，不是 bug）──────────────────────────
  // 中文数字 ↔ 阿拉伯数字的等价表达会被否决：把中文数字归一化需要完整的数词解析
  // （"十二"=12、"两百零三"=203、"一万二"=12000、"两个"≈"2个"），本规则不做。
  // 代价：同义句里只要一方用中文数字，就会被判不一致、放弃这次缓存命中。
  // 方向上是安全的（少命中一次，多调一次上游），而反向错误（命中错答案）不可接受。
  {
    CHECK(vetoed("1+1等于几", "一加一等于几")); ok++;
    CHECK(vetoed("第2章讲什么", "第二章讲什么")); ok++;
    // 两边都用中文数字则一致（都无实体）→ 放行
    CHECK(!vetoed("一加一等于几", "一加一得多少")); ok++;
  }

  // ── 6. entity_mismatch 的调用契约（CacheEngine 依赖它）──────────────
  // CacheEngine 只在**查询侧有实体**时才查候选的 source 做比对；
  // 查询侧为空时直接跳过否决。这里把"查询为空"的行为也固定下来，
  // 保证"没有实体的普通问句"无论如何都不会被这条规则拦住
  {
    EntityTokens none;
    CHECK(!entity_mismatch(none, none)); ok++;
    CHECK(entity_mismatch(none, extract_entity_tokens("继续12题"))); ok++;
    CHECK(entity_mismatch(extract_entity_tokens("继续12题"), none)); ok++;
  }

  return test_check::finish("test_entity_tokens", ok);
}
