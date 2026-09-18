// LLM API 客户端：转发请求到 OpenAI 兼容后端
//
// 两条路径共用同一套「每线程一个 CURL*」句柄模型（报告 H6）：
//   call_llm()        —— 整段缓冲，返回 {status_code, body}
//   call_llm_stream() —— 逐块回调（SSE 真透传），上游字节到达即交给调用方
#pragma once

#include <string>
#include <string_view>

namespace ai_gateway {

// 向 backend_url 发送 POST 请求，body 为 JSON 字符串
// 返回 {status_code, response_body}
// 失败时 status_code 为 0
struct LlmResponse {
  int status_code = 0;
  std::string body;
};

LlmResponse call_llm(const std::string& url,
                     const std::string& api_key,
                     const std::string& request_body,
                     int timeout_seconds);

// ---------------------------------------------------------------------------
// 流式调用
// ---------------------------------------------------------------------------

// 流被中断的原因。区分它们是因为后续动作不同：
//   kClientGone —— 下游客户端断开，上游应当立刻中停（省 token）
//   kDeadline   —— 下游写死线到点，同样是中停上游
//   kNullSink   —— 前端根本没人接（编程错误，落日志）
enum class StreamAbortReason {
  kNone = 0,
  kClientGone,    // 下游写不进去了（客户端断开 / 写死线）
  kDeadline,      // 整条响应的总死线或上游超时
  kUpstreamIdle,  // 上游两次数据之间的间隔超过 stream_idle_ms
  kNullSink,
};

// 调用方实现的分块接收器。
// 生命周期约定：on_begin 一定会被调用一次（无论状态码），
//               随后是若干 on_chunk，最后是 on_done。
class LlmStreamSink {
 public:
  virtual ~LlmStreamSink() = default;

  // 上游响应头到齐。返回 false = 不接受这种 Content-Type，请把响应体
  // 缓冲下来按普通响应返回（call_llm_stream 会把 body 填进 result.body）。
  // 此时 on_chunk / on_done 都不会再被调用。
  virtual bool on_begin(int /*status_code*/, std::string_view /*content_type*/) {
    return true;
  }
  // 一块上游字节（不保证与 SSE 事件边界对齐）。返回 false = 立刻中断上游传输。
  virtual bool on_chunk(const char* /*data*/, size_t /*len*/) { return true; }
  // 传输结束（含中断）。reason != kNone 时 body 只保证是已收到的部分
  virtual void on_done(StreamAbortReason /*reason*/) {}
};

struct StreamCallResult {
  // 上游状态码；curl 自身失败（连不上/超时）时为 0，由调用方映射 502/504
  int status_code = 0;
  // 上游 Content-Type 原文（原样带回，不做重写）
  std::string content_type;
  // 仅在「未流式」（sink 拒绝 / 上游非 SSE / curl 失败）时有意义：
  // 完整（或已收到的部分）响应体
  std::string body;
  // 是否走了逐块透传路径
  bool streamed = false;
  // 中断原因（streamed=true 时才有意义）
  StreamAbortReason abort_reason = StreamAbortReason::kNone;
  // curl 的错误串（status_code==0 时供日志与错误响应使用）
  std::string curl_error;
};

// 流式转发：上游字节到达即交给 sink->on_chunk，不再整段缓冲。
// 复用 call_llm 的句柄模型与超时配置：
//   - 每线程一个 CURL*（thread_local），无锁，多 worker 真并发
//   - CURLOPT_TIMEOUT = timeout_seconds 作为**整条流的总时限兜底**（默认 60s，
//     由 backend.timeout_seconds 配置；预期长回答会超过它就该把这个值调大）
//   - stream_idle_ms（server.stream_idle_timeout_seconds）作为**上游空闲死线**：
//     由 CURLOPT_XFERINFOFUNCTION 的进度回调检查"两次数据之间的间隔"，超时即中止
//     并把 abort_reason 置为 kUpstreamIdle。这是"上游不发数据"这一侧的唯一防线
//     ——下游写死线只在**写客户端**时被检查，管得住"客户端不读"，管不住"上游不发"
//     （集成测试 C 段实测过这个缺口）
//   - CURLOPT_LOW_SPEED_TIME 只作极慢速兜底：它判的是**平均速率**，对"突发一次
//     再长期静默"无效（实测静默 10s 仍未触发），因此不再承担空闲判定
StreamCallResult call_llm_stream(const std::string& url,
                                 const std::string& api_key,
                                 const std::string& request_body,
                                 int timeout_seconds,
                                 int stream_idle_ms,
                                 LlmStreamSink* sink);

}  // namespace ai_gateway
