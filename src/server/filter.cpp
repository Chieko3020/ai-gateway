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

std::string MessageFilter::sse_feed(SseFilterState& st, std::string_view chunk,
                                    bool final_chunk) const {
  std::string out;
  if (st.rejected) return out;  // 已判定拒绝：后续字节一律不放行

  st.pending.append(chunk);

  // 事件边界 = 空行。SSE 规范里空行（\n\n / \r\n\r\n / \r\r）结束一个事件
  size_t search_from = 0;
  while (true) {
    size_t event_end = std::string::npos;  // 事件内容结束位置（不含空行）
    size_t next_start = std::string::npos; // 下一个事件的起始位置
    for (size_t i = search_from; i < st.pending.size(); ++i) {
      const char c = st.pending[i];
      if (c != '\n' && c != '\r') continue;
      // 找一个由相同/混合换行字符组成的"空行"
      size_t j = i;
      int newlines = 0;
      while (j < st.pending.size() &&
             (st.pending[j] == '\n' || st.pending[j] == '\r')) {
        // "\r\n" 算一个换行，别数成两个
        if (st.pending[j] == '\r' && j + 1 < st.pending.size() &&
            st.pending[j + 1] == '\n') {
          j += 2;
        } else {
          ++j;
        }
        ++newlines;
      }
      if (newlines >= 2) {
        event_end = i;
        next_start = j;
        break;
      }
      if (j >= st.pending.size()) break;  // 这一段的换行还没到齐
      i = j - 1;                          // 单个换行：继续往后找
    }

    if (event_end == std::string::npos) break;  // 没有完整事件了

    std::string event = st.pending.substr(0, event_end);
    // 每条事件按整条判定（含 data: 前缀与 JSON 内容）：
    // URL 可能横跨 `data:` 与后续字节，只看 JSON 正文反而会漏
    auto fr = check_output(event);
    if (fr.action == FilterAction::kReject) {
      st.rejected = true;
      st.rejected_event = std::move(event);
      return out;  // 不返回任何未交付的字节？——已放行的部分照常返回
    }
    st.accepted.append(st.pending, 0, next_start);
    st.accepted_bytes += next_start;
    out.append(st.pending, 0, next_start);

    st.pending.erase(0, next_start);
    search_from = 0;

    // 输出长度上限：按事件整体丢弃后续事件（截断），保持事件边界完整
    if (config_.max_output_chars > 0 &&
        st.accepted_bytes >= static_cast<size_t>(config_.max_output_chars)) {
      st.truncated = true;
      st.pending.clear();
      return out;
    }
  }

  if (final_chunk && !st.pending.empty() && !st.rejected) {
    // 上游结束但最后一段没有以空行结尾（真实上游的最后一个事件通常是
    // "data: [DONE]\n\n"，但断流时会缺尾）：按最后一个事件处理
    auto fr = check_output(st.pending);
    if (fr.action == FilterAction::kReject) {
      st.rejected = true;
      st.rejected_event = st.pending;
      st.pending.clear();
      return out;
    }
    out += st.pending;
    st.accepted += st.pending;
    st.accepted_bytes += st.pending.size();
    st.pending.clear();
  }
  return out;
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
