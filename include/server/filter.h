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

// 分级滑窗保留的字节上限：约 2–3 个 delta（URL 通常只跨 1 个）。
// 窗口越大判定越准，但"被拦时已经发出去的内容"也越多——流式不可撤回，
// 这是协议决定的，不是实现取舍
inline constexpr size_t kSseWindowBytes = 512;

// 截断一条 SSE 流时补发的终止事件。
// 与上游自己发的终止事件同名同形：客户端（含 OpenAI 兼容 SDK）只认这一个流结束
// 标志，截断若只把后续字节丢掉而不补它，客户端会一直等下去（keep-alive 下连接
// 不会关，只能等读超时）。2026-09 的真上游复测正是靠它发现"截断吞掉 [DONE]"的
inline constexpr std::string_view kSseDoneEvent = "data: [DONE]\n\n";

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
    // 最近若干事件的拼接：违规串可能被 SSE 事件边界劈开（`https://evil.` 与
    // `example.com/x` 分属两个 delta），逐事件判定必然漏检。这里保留一小段近期
    // 内容，只在"看起来可能出现 URL"时才拿它一起判定（分级滑窗）——
    // 否则把跨事件判定摊到每个事件上，正常文本也要多付一次正则
    std::string window;
    // 已放行的事件原文。**只在启用输出截断（max_output_chars > 0）时维护**：
    // 旧实现无条件累积，而默认配置 max_output_chars = 0（不截断），于是整条流的
    // 每个事件都会被完整留到最后——纯占内存，且没有任何读端（SsePassthroughSink
    // 只读 rejected / rejected_event，见报告 L6）。截断判定本身只用 accepted_bytes
    std::string accepted;
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
  // 把"已放行的字节"记进 SseFilterState::accepted —— 只在启用输出截断时维护
  // （见 accepted 字段的说明；默认不截断时这里什么都不做，避免整条流被留一份副本）
  void accumulate_accepted(SseFilterState& st, std::string_view data,
                           size_t pos, size_t len) const {
    if (config_.max_output_chars <= 0 || len == 0) return;
    st.accepted.append(data.data() + pos, len);
  }

  // 检测是否包含 URL
  static bool contains_url(std::string_view text);

  // 检测是否包含屏蔽关键词
  bool contains_blocked_keyword(std::string_view text) const;

  // 检测注入攻击特征
  static bool contains_injection(std::string_view text);

  FilterConfig config_;
};

}  // namespace ai_gateway
