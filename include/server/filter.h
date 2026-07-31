// 安全过滤器：在请求到达业务逻辑前拦截不合规消息
#pragma once

#include <string>
#include <string_view>

#include "common/config.h"

namespace ai_gateway {

// 过滤结果
enum class FilterAction {
  kPass,     // 放行
  kReject,   // 拒绝（返回错误）
  kTruncate, // 截断后放行
};

struct FilterResult {
  FilterAction action = FilterAction::kPass;
  std::string sanitized;   // 处理后的文本（截断/清理后）
  std::string reject_msg;  // 拒绝原因
};

class MessageFilter {
 public:
  explicit MessageFilter(const FilterConfig& config);

  // 过滤输入消息
  FilterResult check_input(std::string_view message) const;

  // 过滤输出消息（LLM 回复）
  FilterResult check_output(std::string_view message) const;

 private:
  // 检测是否包含 URL
  static bool contains_url(std::string_view text);

  // 检测是否包含屏蔽关键词
  bool contains_blocked_keyword(std::string_view text) const;

  // 检测注入攻击特征
  static bool contains_injection(std::string_view text);

  FilterConfig config_;
};

}  // namespace ai_gateway
