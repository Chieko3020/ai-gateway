// 实体标记提取实现（规则见 include/cache/entity_tokens.h）
#include "cache/entity_tokens.h"

namespace ai_gateway {

namespace {

// ASCII 判定只看字节：UTF-8 的多字节序列每个字节都 >= 0x80，天然不会落进
// [0-9]/[A-Z]/[a-z] 三个区间，因此不需要先解码码点——对 CJK 文本既省一次
// 解码，也避免了"非法字节怎么算"的边界问题（非法字节一律当分隔符）
inline bool ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool ascii_upper(unsigned char c) { return c >= 'A' && c <= 'Z'; }
inline bool ascii_lower(unsigned char c) { return c >= 'a' && c <= 'z'; }
inline bool ascii_alpha(unsigned char c) {
  return ascii_upper(c) || ascii_lower(c);
}
inline bool ascii_alnum(unsigned char c) {
  return ascii_alpha(c) || ascii_digit(c);
}

// 把当前扫描到的连续字母数字串分类存入 out。
//   纯数字            -> numbers
//   纯大写            -> acronyms（小写字母出现即不满足，故 "Abc" 不入此类）
//   含数字且含字母     -> identifiers
//   纯小写            -> 丢弃（没有判别力："hello"/"world" 遍地都是）
void classify(std::string& run, EntityTokens& out) {
  if (run.empty()) return;

  bool has_digit = false, has_alpha = false, has_lower = false;
  for (unsigned char c : run) {
    if (ascii_digit(c)) has_digit = true;
    if (ascii_alpha(c)) has_alpha = true;
    if (ascii_lower(c)) has_lower = true;
  }

  if (has_digit && has_alpha) {
    out.identifiers.insert(run);
  } else if (has_digit) {
    out.numbers.insert(run);
  } else if (has_alpha && !has_lower) {
    out.acronyms.insert(run);
  }
  run.clear();
}

// 两个集合是否互为子集（即"没有不对称差集"）。
// 不用 std::set 的 includes/equal：这里只需要"是否存在只在一边出现的元素"，
// 对 unordered_set 直接逐元素查表，O(n) 且不需要排序/构造临时容器。
template <typename Set>
bool subset_of(const Set& a, const Set& b) {
  for (const auto& v : a) {
    if (b.find(v) == b.end()) return false;
  }
  return true;
}

}  // namespace

EntityTokens extract_entity_tokens(std::string_view text) {
  EntityTokens out;
  std::string run;
  run.reserve(16);

  for (unsigned char c : text) {
    if (ascii_alnum(c)) {
      run.push_back(static_cast<char>(c));
      continue;
    }
    classify(run, out);
  }
  classify(run, out);  // 收尾：文本以一个标记结束时循环里不会调用 classify
  return out;
}

bool entity_mismatch(const EntityTokens& query, const EntityTokens& candidate) {
  if (query.empty() && candidate.empty()) return false;  // 两边都没标记：不否决
  return !subset_of(query.numbers, candidate.numbers) ||
         !subset_of(candidate.numbers, query.numbers) ||
         !subset_of(query.acronyms, candidate.acronyms) ||
         !subset_of(candidate.acronyms, query.acronyms) ||
         !subset_of(query.identifiers, candidate.identifiers) ||
         !subset_of(candidate.identifiers, query.identifiers);
}

}  // namespace ai_gateway
