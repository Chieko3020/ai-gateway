// 安全过滤器实现
#include "server/filter.h"

#include <algorithm>
#include <regex>

namespace ai_gateway {

MessageFilter::MessageFilter(const FilterConfig& config)
    : config_(config) {}

FilterResult MessageFilter::check_input(std::string_view message) const {
  FilterResult result;

  // 1. 反注入检测（最高优先级）
  if (contains_injection(message)) {
    result.action = FilterAction::kReject;
    result.reject_msg = "Request rejected: potential injection detected";
    return result;
  }

  // 2. URL 检测
  if (config_.block_urls && contains_url(message)) {
    result.action = FilterAction::kReject;
    result.reject_msg = "Request rejected: URLs are not allowed";
    return result;
  }

  // 3. 关键词检测
  if (contains_blocked_keyword(message)) {
    result.action = FilterAction::kReject;
    result.reject_msg = "Request rejected: blocked content";
    return result;
  }

  // 4. 长度限制（<=0 表示不截断，与 check_output 口径一致）
  if (config_.max_input_chars > 0 &&
      static_cast<int>(message.size()) > config_.max_input_chars) {
    result.action = FilterAction::kTruncate;
    result.sanitized = std::string(message.substr(0, config_.max_input_chars));
    return result;
  }

  result.action = FilterAction::kPass;
  result.sanitized = std::string(message);
  return result;
}

FilterResult MessageFilter::check_output(std::string_view message) const {
  FilterResult result;

  // 输出 URL 检测
  if (config_.block_urls && contains_url(message)) {
    result.action = FilterAction::kReject;
    result.reject_msg = "Response rejected: contains URL";
    return result;
  }

  // 长度截断：max_output_chars <= 0 表示不截断。
  // 旧默认值 600 会把绝大多数正常回答静默截断到 600 字节——对"透明代理"定位而言
  // 这是错误的默认值，因此语义改为"0 = 不限制"（报告 L9）
  if (config_.max_output_chars > 0 &&
      static_cast<int>(message.size()) > config_.max_output_chars) {
    result.action = FilterAction::kTruncate;
    result.sanitized = std::string(message.substr(0, config_.max_output_chars));
    return result;
  }

  result.action = FilterAction::kPass;
  result.sanitized = std::string(message);
  return result;
}

bool MessageFilter::contains_url(std::string_view text) {
  // 匹配常见 URL 模式
  static const std::regex url_re(
      R"((https?://|ftp://|www\.)[^\s<>"']+)",
      std::regex::icase | std::regex::optimize);
  return std::regex_search(text.begin(), text.end(), url_re);
}

bool MessageFilter::contains_blocked_keyword(std::string_view text) const {
  for (const auto& kw : config_.blocked_keywords) {
    if (text.find(kw) != std::string_view::npos) return true;
  }
  return false;
}

bool MessageFilter::contains_injection(std::string_view text) {
  // 检测常见 prompt injection 模式。
  // 只保留"本身就是攻击指令"的特征：system prompt / you are now / new instructions
  // 这类良性短语（用户正常讨论提示词工程就会命中）已移除——默认配置下它们会把
  // 大量正常请求判成注入并拒绝（报告 L9）
  static const std::vector<std::string_view> patterns = {
      "ignore previous",
      "ignore all instructions",
      "ignore the above",
      "disregard previous",
      "disregard all prior",
      "forget your prompt",
      "forget all previous",
      "reveal your system prompt",
      "print your system prompt",
      "repeat your system prompt",
      "override your instructions",
      "[/INST]",
      "<|im_start|>",
      "<|system|>",
      "DAN mode",
      "jailbreak",
  };
  // 逐字符大小写不敏感匹配（避免堆分配）
  for (auto& p : patterns) {
    auto it = std::search(
        text.begin(), text.end(), p.begin(), p.end(),
        [](unsigned char a, unsigned char b) {
          return std::tolower(a) == std::tolower(b);
        });
    if (it != text.end()) return true;
  }
  return false;
}

}  // namespace ai_gateway
