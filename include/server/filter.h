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

  // ---- SSE 流式输出的按事件过滤 -----------------------------------------
  // 为什么不能对整段 SSE 做正则替换：一段流里有多个 `data: {...}` 事件，
  // 对整段做替换会把事件边界与 JSON 一起毁掉（旧实现对整段响应体做
  // "替换成 {\"error\":...}" 就属这类）。正确做法是按 **SSE 事件边界**
  // （空行分隔）逐条判定：命中屏蔽规则的那一条事件被丢弃并判定为 reject。
  //
  // 用法（流式 handler 持有状态跨 chunk 调用）：
  //   SseFilterState st;
  //   st.feed(chunk, false);                    // 每个上游 chunk
  //   if (st.rejected()) -> 中断
  //   st.feed({}, true);                        // 上游结束时冲刷尾部
  struct SseFilterState {
    std::string pending;        // 尚未构成完整事件（无空行结尾）的字节
    std::string accepted;       // 已放行的事件（供日志/统计与缓冲式路径统一口径）
    std::string rejected_event; // 被拒的那一条事件原文（供日志）
    bool rejected = false;
    bool truncated = false;
    size_t accepted_bytes = 0;
  };

  // 消费一块上游字节：
  //   final_chunk=false → 只处理已经完整的事件（以空行结束），余下留在 pending
  //   final_chunk=true  → 把 pending 里的残余也当作最后一个事件处理
  // 返回本次新放行的事件（可直接写到客户端）。一旦 rejected_ 被置位，
  // 后续调用不再放行任何字节
  std::string sse_feed(SseFilterState& st, std::string_view chunk,
                       bool final_chunk) const;

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
