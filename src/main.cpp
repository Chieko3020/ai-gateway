// 网关入口：加载配置 初始化各模块 启动 HTTP 服务

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <format>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "backend/llm_client.h"
#include "cache/cache_engine.h"
#include "cache/embedding_fingerprint.h"
#include "cache/lru_store.h"
#include "cache/entity_tokens.h"
#include "cache/hnsw_index.h"
#include "common/log_file.h"
#include "cache/onnx_embedding.h"
#include "common/config.h"
#include "common/logger.h"
#include "common/singleflight.h"
#include "common/types.h"
#include "server/filter.h"
#include "server/http_server.h"
#include "server/metrics.h"
#include "server/response.h"
#include "server/sse_usage.h"
#include "stats/stats.h"

using namespace ai_gateway;
using json = nlohmann::json;
using namespace std::chrono_literals;

// FNV-1a 64-bit 确定性哈希：保证跨进程一致（std::hash 有随机种子，重启后同一
// system prompt 会算出不同 namespace，缓存隔离失效）。
// 32 位版本碰撞概率虽低但非零，碰撞即"不同 system prompt 共用一个缓存命名空间"，
// 后果是跨对话串答案，因此升到 64 位（报告 L4）
static uint64_t fnv1a_64(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ull;
  }
  return h;
}

// 分词器实现标识：**改了分词规则就必须改这个字符串**。
// 它进 embedding 指纹，用来覆盖"模型与词表文件都没变、但分词实现改了"这一情形
// （本项目 2026-09 刚补过 Bert WordPiece 的 lowercase/## 续接规则，正属此类）。
// 历史值：v1 = 贪心 10 字符窗口（与官方不一致率 58.7%）；v2 = 对齐官方 WordPiece
static constexpr const char* kTokenizerId = "bert-wordpiece@v2";

// 向量产生方式的完整标识（进 embedding 指纹）：
//   分词器实现 + 是否小写化 + 池化方式
// 这三者任何一个改变，**同一段文本产生的向量就不同**，旧落盘向量与新查询向量
// 不在同一个空间里，余弦相似度失去意义且不会报错。把它们拼进指纹，加载时会
// 判定不一致 → 丢弃旧向量、保留文本条目、按新配置重建索引。
// 本轮（v2-case-mean → v2-lower-cls）正是一次这样的变更：do_lower_case 默认
// 由 false 改为 true、池化由 mean 改为 cls，因此**所有旧向量必然失效**。
static std::string embedding_variant_id(const EmbeddingConfig& emb) {
  return std::format("{}|lower={}|pooling={}", kTokenizerId,
                     emb.do_lower_case ? "1" : "0",
                     emb.pooling == PoolingMode::kCls ? "cls" : "mean");
}

// 把一条 message 的 content 拍平成纯文本：
//   content 为 string       → 原样
//   content 为数组（多模态） → 拼接各 text 片段（image_url 等非文本片段跳过）
//   content 缺失/其它类型    → 空串
// 旧实现直接 value("content","")：数组形式会抛 type_error 被吞掉，
// 于是多模态请求的 check_input 根本不执行（报告 M11 的绕过面）
static std::string message_text(const json& msg) {
  if (!msg.is_object()) return "";
  auto it = msg.find("content");
  if (it == msg.end()) return "";
  if (it->is_string()) return it->get<std::string>();
  if (it->is_array()) {
    std::string out;
    for (const auto& part : *it) {
      if (part.is_string()) {
        out += part.get<std::string>();
        continue;
      }
      if (!part.is_object()) continue;
      auto t = part.find("text");
      if (t != part.end() && t->is_string()) out += t->get<std::string>();
    }
    return out;
  }
  return "";
}

// 提取 system 消息内容用于缓存隔离：扫描全部 messages（system 不在首位时
// 旧实现会静默丢失隔离），拼接多段 system 文本后取 FNV-1a 64 位。
// 返回值形如 "ns<16 位 hex>"：
//   - 带 ns 前缀，避免与"客户端消息本身恰好是 16 位十六进制数"撞进同一 key 空间
//   - 位宽变化会让已有落盘缓存的 namespace 前缀全部改变（一次性失配，见简报）
static std::string extract_namespace(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    auto& msgs = req.at("messages");
    std::string system_text;
    for (const auto& m : msgs) {
      if (!m.is_object()) continue;
      if (m.value("role", "") != "system") continue;
      auto text = message_text(m);
      if (text.empty()) continue;
      if (!system_text.empty()) system_text += "\n";
      system_text += text;
    }
    if (!system_text.empty())
      return std::format("ns{:016x}", fnv1a_64(system_text));
  } catch (...) {}
  return "";  // 客户端没有 system message 或解析失败，返回空字符串表示不做隔离
}

// 从 OpenAI 格式请求体中提取最后一条 user message（用于缓存键与向量化）
static std::string extract_user_message(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    auto& msgs = req.at("messages");
    for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
      if (!it->is_object()) continue;
      if (it->value("role", "") == "user") return message_text(*it);
    }
  } catch (...) {}
  return "";
}

// 拼接所有 message 的文本：输入过滤的检查面（旧实现只查最后一条 user 消息）
static std::string collect_all_text(const std::string& request_body) {
  std::string out;
  try {
    auto req = json::parse(request_body);
    auto it = req.find("messages");
    if (it == req.end() || !it->is_array()) return "";
    for (const auto& m : *it) {
      auto text = message_text(m);
      if (text.empty()) continue;
      if (!out.empty()) out += "\n";
      out += text;
    }
  } catch (...) {}
  return out;
}

// 请求分类：旁路（tool 类）与拒绝（stream）是两件事，必须分开判定。
// 旧实现把二者合并成一个 is_uncacheable_request()，于是 stream:true 也走
// "转发给上游再把整段缓冲的 SSE 文本塞进 JSON 信封"的伪透传路径。
static bool wants_stream(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    return req.contains("stream") && req.value("stream", false);
  } catch (...) {}
  return false;
}

// 流式请求是否要求上游在最后一个事件里带上 usage。
// OpenAI 兼容实现（含 DeepSeek）只在 `stream_options.include_usage: true` 时才发
// usage 事件；客户端没要，网关就拿不到 token 数。
// 网关**不会**擅自改写客户端请求去补这个字段（透明代理不能悄悄改变上游看到的
// 请求），这种情况按"上游未提供 usage"处理并在报表里单列计数
static bool stream_includes_usage(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    auto it = req.find("stream_options");
    if (it == req.end() || !it->is_object()) return false;
    return it->value("include_usage", false);
  } catch (...) {}
  return false;
}

// 工具调用类请求（tools/functions 定义、tool/function 角色的消息、tool_calls）：
// 与上下文强相关，缓存会丢失 tool_calls 或返回不适用答案，因此不缓存、不参与合并，
// 但仍然走输入/输出过滤并按上游状态码透传。
static bool is_tool_request(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    if (req.contains("tools") || req.contains("functions")) return true;
    auto& msgs = req.at("messages");
    for (const auto& m : msgs) {
      auto role = m.value("role", "");
      if (role == "tool" || role == "function") return true;
      if (m.contains("tool_calls") || m.contains("function_call")) return true;
    }
  } catch (...) {}
  return false;
}

// stream:true 真透传（本轮实现）。
//
// 历史：此前 stream:true 被直接 400 拒绝（连"伪透传"都不是），原因是 DSH 的 LLM 层
// @earendil-works/pi-ai 在 openai-completions API 里硬编码 stream: true
// （~/.dsh/profiles/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js:587），
// 于是它的每个请求都会命中这个分支，网关根本接不进去。
//
// 实现要点（与 TODO 里的方案对应）：
//   1. llm_client 提供 call_llm_stream：write 回调里逐块把上游字节交给 sink，
//      handler 收到即写客户端 socket（ResponseWriter 直接 send，无缓冲等待）
//   2. 响应头由 handler 自己写：Content-Type 按上游原样透传（text/event-stream），
//      长度语义用 Transfer-Encoding: chunked
//   3. 输出过滤按 SSE 事件边界逐条判定（MessageFilter::sse_feed），不再对整段
//      做正则替换
//   4. 客户端断开 / 写死线到点 → sink 返回 false → curl write 回调返回 0 →
//      上游传输立刻中断（省 token），worker 立即归还 fd
// 在 OpenAI 响应体中注入缓存状态字段，供压测脚本精确判定是否命中；
// 额外字段不影响下游对标准字段的解析。
static std::string annotate_cache_status(const std::string& body,
                                         const char* status) {
  try {
    auto resp = json::parse(body);
    if (!resp.is_object()) return body;
    resp["_cache"] = status;
    return resp.dump();
  } catch (...) {
    return body;  // 非 JSON（如错误页）原样返回
  }
}

// 响应体/消息的短摘要：日志里不落原文，只落"长度 + 64 位哈希"，
// 既能给排查用的关联标识，又不泄露用户内容（报告 M5）
static std::string body_digest(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ull;
  }
  return std::format("len={},h={:016x}", s.size(), h);
}

// 透传上游状态码：curl 自身失败时 llm_client 已映射为 502/504，
// 这里再兜一层，确保不会出现 status_code=0 被当成正常码发回客户端
static int upstream_status(int status_code) {
  return status_code > 0 ? status_code : 502;
}

// 线程安全的关闭标志与后台线程唤醒机制
static std::atomic<bool> g_shutdown{false};
static std::atomic<bool> g_dump_stats{false};  // SIGUSR1 置位，由后台线程输出统计
static std::mutex g_bg_mutex;
static std::condition_variable g_bg_cv;

void handle_signal(int sig) {
  if (sig == SIGUSR1) {
    // 按需导出统计：信号处理器里只置标志（report() 会加锁+写日志，不是 async-signal-safe）
    g_dump_stats.store(true, std::memory_order_release);
    return;
  }
  g_shutdown.store(true, std::memory_order_release);
}

// 处理结果的统一形状：pair<HttpReply, 能否复用连接>
//
// writer 只在流式（SSE 真透传）路径上被使用：那条路径要边收边发，不能等整段
// 响应体凑齐。缓冲式路径只用返回值。
//
// 为什么 keep-alive 要跟 HttpReply 一起返回：流式响应由 handle_stream_request
// 自己写响应头，keep-alive 的决定必须与 Connection 头在同一处产生，否则会出现
// "头里写了 keep-alive、连接却被关掉"（或反之）的不一致。
using HandleOutcome = std::pair<HttpReply, bool>;

// 网关自身生成的 JSON 响应（默认 200；拒绝类响应给出语义正确的状态码：
// 输入被拒 400、上游内容被拦 502）
static HandleOutcome json_reply(std::string body, int status_code = 200) {
  return {HttpReply{status_code, "application/json", std::move(body)}, false};
}

// 网关自己生成的错误响应：与 json_reply 同义，但名字在错误路径上更直白
static HandleOutcome error_outcome(std::string body, int status_code) {
  return json_reply(std::move(body), status_code);
}

// ---------------------------------------------------------------------------
// SSE 真透传
// ---------------------------------------------------------------------------

// 流式请求的 keep-alive：上游是 SSE 长连接，下游同样可以复用。
// 注意这是"允许复用"，客户端显式 Connection: close 时仍然会关闭
static constexpr bool kStreamKeepAlive = true;

namespace {

// SSE 透传的接收器：上游每来一块字节，就（按事件边界过滤后）立刻写到客户端。
//
// 三件事同时发生在这里：
//   1. 输出过滤按 SSE 事件边界逐条判定（不能对整段做正则替换）
//   2. 写失败（客户端断开 / 写死线到）立刻返回 false，让 curl 中断上游传输
//   3. 统计首字节时刻与 [DONE] 是否完整送达
class SsePassthroughSink : public LlmStreamSink {
 public:
  SsePassthroughSink(MessageFilter* filter, ResponseWriter* writer,
                     std::chrono::steady_clock::time_point t0, bool keep_alive)
      : filter_(filter), writer_(writer), t0_(t0), keep_alive_(keep_alive) {}

  // 上游响应头到齐：**在这里**把响应头发给下游。
  // 为什么不提前发：只有这一刻才知道上游的状态码与 Content-Type，按上游原样
  // 透传（text/event-stream）是硬要求；也不能等第一个 chunk，那样"上游迟迟不吐
  // 第一个 token"会被客户端当成网关不响应。
  bool on_begin(int st, std::string_view ct) override {
    head_sent_ = writer_->write_stream_head(st, ct, keep_alive_);
    if (!head_sent_) write_failed_ = true;
    return head_sent_;
  }

  bool on_chunk(const char* data, size_t len) override {
    // 首字节：从"收到客户端请求"到"上游第一个字节到达"。这才是流式体验的真实延迟
    if (!first_byte_seen_) {
      first_byte_seen_ = true;
      first_byte_ms_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t0_)
              .count();
    }
    bytes_in_ += len;
    scan_for_done(data, len);
    // 旁路观察一份 usage（不改动透传的字节）：token 用量只在最后一个 SSE 事件里，
    // 客户端要求了 stream_options.include_usage 才存在。解析是有界窗口的
    // 增量扫描，见 server/sse_usage.h
    usage_.feed(data, len);

    std::string out =
        filter_->sse_feed(filter_state_, std::string_view(data, len),
                          /*final_chunk=*/false);
    if (!out.empty() && !writer_->write_body(out)) {
      // 客户端断开 / 写死线到：标记原因并让 curl 中断上游
      write_failed_ = true;
      abort_ = writer_->abort_reason();
      return false;
    }
    if (filter_state_.rejected) {
      // 命中屏蔽规则：停止继续放行（上游同样被中断，不再为被拦内容付费）
      filter_rejected_ = true;
      return false;
    }
    return true;
  }

  void on_done(StreamAbortReason reason) override {
    if (write_failed_) return;  // 已因写失败退出，上游传输也已被中断
    // 冲刷尾部：上游结束时最后一段可能没有以空行结尾
    std::string tail = filter_->sse_feed(filter_state_, {}, /*final_chunk=*/true);
    if (!tail.empty() && !writer_->write_body(tail)) {
      write_failed_ = true;
      abort_ = writer_->abort_reason();
    }
    if (filter_state_.rejected) filter_rejected_ = true;
    (void)reason;
  }

  // ---- 观测数据（handler 收尾时读取） ----
  bool head_sent() const { return head_sent_; }
  bool done_seen() const { return done_seen_; }
  bool filter_rejected() const { return filter_rejected_; }
  bool write_failed() const { return write_failed_; }
  AbortReason abort_reason() const { return abort_; }
  int64_t first_byte_ms() const { return first_byte_ms_; }
  size_t bytes_in() const { return bytes_in_; }
  const StreamUsage& usage() const { return usage_.usage(); }
  const std::string& rejected_event() const {
    return filter_state_.rejected_event;
  }

 private:
  // 在后端字节流里找 "[DONE]"：事件可能跨 chunk，因此把上一块的尾巴（32 字节）
  // 拼到本次搜索窗口前面。只做"是否出现过"的判定，不改变透传的字节
  void scan_for_done(const char* data, size_t len) {
    std::string window;
    window.reserve(tail_.size() + len);
    window += tail_;
    window.append(data, len);
    if (window.find("[DONE]") != std::string::npos) done_seen_ = true;
    tail_ = window.size() > 32 ? window.substr(window.size() - 32) : window;
  }

  MessageFilter* filter_;
  ResponseWriter* writer_;
  std::chrono::steady_clock::time_point t0_;
  bool keep_alive_ = true;
  bool head_sent_ = false;
  MessageFilter::SseFilterState filter_state_;
  SseUsageParser usage_;  // 旁路观察 token 用量，不参与透传
  std::string tail_;
  bool done_seen_ = false;
  bool filter_rejected_ = false;
  bool write_failed_ = false;
  bool first_byte_seen_ = false;
  int64_t first_byte_ms_ = 0;
  size_t bytes_in_ = 0;
  AbortReason abort_ = AbortReason::kNone;
};

}  // namespace

// 处理 stream:true：把上游 SSE 逐块透传给客户端
static HandleOutcome handle_stream_request(
    const std::string& request_body, const GatewayConfig& cfg,
    MessageFilter* filter, Stats* stats, ResponseWriter& writer,
    std::chrono::steady_clock::time_point t0, bool client_wants_keep_alive) {
  // 响应头的 Connection 必须如实反映客户端意愿（客户端显式要求 close 时
  // 写 keep-alive 会把它挂死在"等下一个响应"上）；真正是否复用由连接处理器
  // 用同一份意愿决定
  const bool head_keep_alive = kStreamKeepAlive && client_wants_keep_alive;
  const bool request_includes_usage = stream_includes_usage(request_body);
  SsePassthroughSink sink(filter, &writer, t0, head_keep_alive);
  // 上游空闲死线：server.stream_idle_timeout_seconds（秒）-> 毫秒。
  // 这是"上游不发数据"那一侧的防线——写死线只在**写客户端**时被检查，
  // 上游静默时根本没有写调用发生（集成测试 C 段实测过这个缺口）
  const int stream_idle_ms = cfg.server.stream_idle_timeout_seconds * 1000;
  auto result = call_llm_stream(cfg.backend.url, cfg.backend.api_key,
                                request_body, cfg.backend.timeout_seconds,
                                stream_idle_ms, &sink);

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);

  if (result.streamed) {
    // ---- 成功进入流式路径 ----
    // 响应头在 sink.on_begin 里已发（write_stream_head）。若它没发出去（客户端
    // 在拿到上游头之前就断了），这里不能再补一个响应：连接已经没有意义，
    // 只能按"提前结束"处理并关闭连接
    if (!sink.head_sent()) {
      stats->record_stream_aborted();
      LOG_WARN("stream: response head not sent (client gone after {} bytes)",
               sink.bytes_in());
      return {HttpReply{200, "text/event-stream", {}}, false};
    }
    // 结束时写 chunked 终止块（0\r\n\r\n）。写不进去说明客户端已经走了，
    // 此时连接不能复用
    bool finished = writer.finish_stream();
    bool keep_alive = kStreamKeepAlive && finished && !sink.write_failed() &&
                      sink.abort_reason() != AbortReason::kDeadline &&
                      sink.abort_reason() != AbortReason::kClientGone;

    if (sink.write_failed() || sink.abort_reason() != AbortReason::kNone) {
      // 客户端提前断开 / 写死线到点：不进延迟样本池（口径见 stats.h）
      stats->record_stream_aborted();
      LOG_WARN("stream: aborted after {} bytes ({}), keep_alive={}",
               sink.bytes_in(),
               sink.abort_reason() == AbortReason::kIdle
                   ? "write idle timeout"
                   : (sink.abort_reason() == AbortReason::kDeadline
                          ? "write deadline"
                          : "client gone"),
               keep_alive ? "yes" : "no");
    } else {
      // token 统计：usage 在最后一个 SSE 事件里（且只在客户端带了
      // stream_options.include_usage 时上游才会发）。SseUsageParser 旁路解析这份
      // 用量，透传的字节一个都没动。上游没给 usage 时**优雅退化为 0**，
      // 并用 streams_no_usage 单列计数——"网关没解析"与"上游没给"在报表里可区分
      const StreamUsage usage = sink.usage();
      if (!usage.seen) stats->record_stream_no_usage();
      // 延迟口径：进样本池的是首字节延迟 TTFT，total 只进日志
      stats->record_stream(sink.first_byte_ms(), elapsed.count(),
                           static_cast<int>(usage.prompt_tokens),
                           static_cast<int>(usage.completion_tokens));
      LOG_INFO_SAMPLED(
          "stream: done ttft={}ms total={}ms bytes={} done_event={} "
          "filter_rejected={} keep_alive={} tokens_in={} tokens_out={} "
          "usage_seen={} include_usage={}",
          sink.first_byte_ms(), elapsed.count(), sink.bytes_in(),
          sink.done_seen() ? "yes" : "no", sink.filter_rejected() ? "yes" : "no",
          keep_alive ? "yes" : "no", usage.prompt_tokens,
          usage.completion_tokens, usage.seen ? "yes" : "no",
          request_includes_usage ? "yes" : "no");
    }
    if (sink.filter_rejected())
      LOG_WARN("filter: rejected SSE event ({} bytes, contains URL)",
               sink.rejected_event().size());
    // 流式响应的状态码与响应头已经写出去了，这里返回的 HttpReply 不会再被发送
    // （连接处理器见到 writer.committed() 就跳过）
    return {HttpReply{200, "text/event-stream", {}}, keep_alive};
  }

  // ---- 没走成流式 ----
  // 上游返回的不是 SSE（常见：上游 4xx/5xx 的 JSON 错误体）、curl 自己失败，
  // 或 sink 拒绝了流式。此时 result.body 是完整（或已收到的部分）响应体，
  // 按普通 JSON 响应回给客户端，状态码沿用上游，保证错误体仍可解析、SDK 能判错
  if (result.status_code == 0) {
    LOG_ERROR("stream: upstream call failed: {}", result.curl_error);
    stats->record_stream_aborted();
    return error_outcome(
        std::format(R"({{"error":"upstream error: {}"}})",
                    result.curl_error.empty() ? std::string("unknown")
                                              : result.curl_error),
        502);
  }
  if (sink.filter_rejected()) {
    LOG_WARN("filter: rejected SSE event before headers were sent");
    return json_reply(R"({"error":"Response filtered"})", 502);
  }

  auto out_result = filter->check_output(result.body);
  if (out_result.action == FilterAction::kReject) {
    LOG_WARN("filter: rejected non-stream upstream output containing URL");
    return json_reply(R"({"error":"Response filtered"})", 502);
  }
  LOG_INFO_SAMPLED("stream: upstream not streamable (status={} ct={}), "
                   "returned as buffered {} bytes",
                   result.status_code, result.content_type, result.body.size());
  // 非流式兜底：状态码透传，但这条连接**不复用**——客户端是按流式发起的，
  // 异常路径上少一层连接状态歧义更稳妥
  return {HttpReply{upstream_status(result.status_code), "application/json",
                    annotate_cache_status(out_result.sanitized,
                                          "stream_fallback")},
          false};
}

static HandleOutcome handle_request(const std::string& request_body,
                                    const GatewayConfig& cfg,
                                    MessageFilter* filter,
                                    CacheEngine* engine,
                                    Singleflight* sf,
                                    Stats* stats,
                                    ResponseWriter& writer,
                                    bool client_wants_keep_alive) {
  auto t0 = std::chrono::steady_clock::now();

  // 1. 输入过滤（所有路径都必须执行）
  //    旁路（工具调用/流式）只应跳过缓存与请求合并，不能跳过安全过滤：
  //    否则客户端加一个 "stream": true 就能绕过注入检测、URL 拦截、屏蔽词与长度截断
  std::string user_msg = extract_user_message(request_body);
  std::string filter_text = collect_all_text(request_body);
  if (!filter_text.empty()) {
    auto f_result = filter->check_input(filter_text);
    if (f_result.action == FilterAction::kReject) {
      LOG_WARN("filter: rejected input: {}", f_result.reject_msg);
      // 输入被拒是客户端错误（旧实现返回 200 + error body，调用方无法据此重试/降级）
      return json_reply(R"({"error":"Request rejected"})", 400);
    }
    if (f_result.action == FilterAction::kTruncate)
      user_msg = f_result.sanitized;
  }

  // 2. 工具调用类流量：不查缓存、不写缓存、不参与请求合并，但仍走双向过滤与状态码透传
  if (is_tool_request(request_body)) {
    auto result = call_llm(cfg.backend.url, cfg.backend.api_key,
                           request_body, cfg.backend.timeout_seconds);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);
    int pt = 0, ct = 0;
    if (result.status_code >= 200 && result.status_code < 300) {
      try {
        auto resp = json::parse(result.body);
        auto& usage = resp.at("usage");
        pt = usage.value("prompt_tokens", 0);
        ct = usage.value("completion_tokens", 0);
      } catch (...) {}
    }
    stats->record_bypass(elapsed.count(), pt, ct);
    LOG_INFO_SAMPLED("cache: bypass (tool) status={} {}ms",
                     result.status_code, elapsed.count());

    auto out_result = filter->check_output(result.body);
    if (out_result.action == FilterAction::kReject) {
      LOG_WARN("filter: rejected output containing URL");
      // 上游内容无法交付给客户端：502（既非客户端错误，也不该沿用上游状态码）
      return json_reply(R"({"error":"Response filtered"})", 502);
    }
    return {HttpReply{upstream_status(result.status_code), "application/json",
                      annotate_cache_status(out_result.sanitized, "bypass")},
            true};
  }

  // 3. stream:true：真透传（SSE）。放在输入过滤之后，保证被拒请求同样经过过滤器
  if (wants_stream(request_body)) {
    return handle_stream_request(request_body, cfg, filter, stats, writer, t0,
                                 client_wants_keep_alive);
  }

  // 缓存命中检查时带回的 embedding（避免 cache_reply 重复计算）
  std::vector<float> cached_embedding;

  // 4. 语义缓存
  std::string ns;
  std::string ns_key;
  if (cfg.cache.enabled && !user_msg.empty()) {
    ns = extract_namespace(request_body);
    ns_key = ns.empty() ? user_msg : ns + ":" + user_msg;
    auto hit = engine->try_hit(user_msg, ns);
    if (hit.hit) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0);
      stats->record_cache_hit(elapsed.count());
      // 命中路径与未命中路径统一口径：缓存里存的是上游原文（未过滤），取出来同样
      // 要过 check_output，否则"先让含 URL 的答案入缓存、再命中"即可绕过输出过滤
      auto hit_out = filter->check_output(hit.reply);
      if (hit_out.action == FilterAction::kReject) {
        LOG_WARN("filter: rejected cached output containing URL");
        return json_reply(R"({"error":"Response filtered"})", 502);
      }
      return {HttpReply{200, "application/json",
                        annotate_cache_status(hit_out.sanitized, "hit")},
              true};
    }
    cached_embedding = std::move(hit.embedding);
  }

  // 5. 请求合并（singleflight）
  // 本请求作为 leader 占用的槽位（insert 的返回值）；只有它有权 complete/cancel，
  // 这样"等待超时后被新 leader 顶替"的旧 leader 不会误写别人的 promise
  std::shared_ptr<std::promise<std::string>> sf_slot;
  if (!user_msg.empty() && !cached_embedding.empty()) {
    // 合并候选的实体一致性校验用查询自己的实体标记（与缓存命中的口径一致：
    // 从 user_msg 提，不带 namespace 前缀——前缀自身的十六进制会被当成混合标识符）
    const EntityTokens query_entities =
        cfg.cache.entity_veto ? extract_entity_tokens(user_msg) : EntityTokens{};
    auto fut = sf->try_merge(ns_key, cached_embedding, query_entities,
                             cfg.cache.entity_veto);
    if (fut.has_value()) {
      auto status = fut->wait_for(
          std::chrono::seconds(cfg.backend.timeout_seconds));
      if (status == std::future_status::ready) {
        // 主请求失败/被取消时等待者会拿到 SingleflightCancelled（而不是
        // broken_promise 触发的 std::future_error）：此时不共享结果、自己回源
        std::string merged;
        bool merged_ok = false;
        try {
          merged = fut->get();
          merged_ok = true;
        } catch (const std::exception& e) {
          LOG_WARN("singleflight: leader failed ({}), fallback to upstream",
                   e.what());
        }
        if (merged_ok) {
          auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t0);
          // 合并命中不是缓存命中：单独计数、单独状态（报告 M10）。
          // 日志只落长度与哈希，不落 key 原文（= namespace:完整用户消息，报告 M5）
          stats->record_merge(elapsed.count());
          LOG_INFO_SAMPLED("singleflight: merged {}", body_digest(ns_key));
          // 合并回来的同样是上游原文，必须与未命中路径一样过输出过滤
          auto merged_out = filter->check_output(merged);
          if (merged_out.action == FilterAction::kReject) {
            LOG_WARN("filter: rejected merged output containing URL");
            return json_reply(R"({"error":"Response filtered"})", 502);
          }
          return {
              HttpReply{200, "application/json",
                        annotate_cache_status(merged_out.sanitized, "merged")},
              true};
        }
      } else {
        LOG_DEBUG("singleflight: wait timeout (src_len={})", ns_key.size());
      }
    }
    sf_slot = sf->insert(ns_key, cached_embedding, query_entities);
  }

  // 6. 缓存未命中 转发 LLM
  auto result = call_llm(cfg.backend.url, cfg.backend.api_key,
                         request_body, cfg.backend.timeout_seconds);

  // 7. singleflight 完成/取消（仅当本请求仍是该 key 的 leader）
  bool ok = (result.status_code >= 200 && result.status_code < 300);
  if (sf_slot) {
    if (ok) sf->complete(ns_key, result.body, sf_slot);
    else sf->cancel(ns_key, sf_slot);
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);

  // 8. 解析 token 用量
  int prompt_tokens = 0, completion_tokens = 0;
  if (result.status_code >= 200 && result.status_code < 300) {
    try {
      auto resp = json::parse(result.body);
      auto& usage = resp.at("usage");
      prompt_tokens = usage.value("prompt_tokens", 0);
      completion_tokens = usage.value("completion_tokens", 0);
    } catch (...) {}
  }

  // 9. 写入缓存
  if (ok && cfg.cache.enabled && !user_msg.empty())
    engine->cache_reply(user_msg, result.body, cached_embedding,
                        extract_namespace(request_body));

  // 10. 统计
  stats->record_api_call(elapsed.count(), prompt_tokens, completion_tokens);

  // 每请求一条 INFO 属热路径：默认全量输出，可用 log_sample_every 采样降级。
  // 只记长度与短摘要，不记上游响应体原文（可能含用户数据，报告 M5）
  if (ok)
    LOG_INFO_SAMPLED("{} {} bytes {}ms", result.status_code, result.body.size(),
                     elapsed.count());
  else
    LOG_WARN("upstream {} {} bytes {}ms body_digest={}", result.status_code,
             result.body.size(), elapsed.count(), body_digest(result.body));

  // 11. 输出过滤
  auto out_result = filter->check_output(result.body);
  if (out_result.action == FilterAction::kReject) {
    LOG_WARN("filter: rejected output containing URL");
    return json_reply(R"({"error":"Response filtered"})", 502);
  }
  // 上游错误码（4xx/5xx/502/504）原样透传，不再一律 200。
  // 状态码 >= 500 时不复用连接（上游故障期间客户端多半会重试，让它们重新握手
  // 反而能错开到别的实例；且这类响应体常常很短，复用收益有限）
  const int status = upstream_status(result.status_code);
  return {HttpReply{status, "application/json",
                    annotate_cache_status(out_result.sanitized, "miss")},
          status < 500};
}

int main(int argc, char* argv[]) {
  // ---- 0. 初始化日志文件（与 systemd journal 双写） ----
  ai_gateway::detail::set_log_file("gateway.log");

  // ---- 1. 加载配置 ----
  const char* config_path = (argc > 1) ? argv[1] : "config/gateway.json";
  GatewayConfig cfg;
  if (GatewayConfig::load(config_path, cfg) != 0)
    return static_cast<int>(ErrorCode::kConfigError);
  if (cfg.backend.url.empty()) {
    LOG_ERROR("backend.url is required");
    return static_cast<int>(ErrorCode::kConfigError);
  }
  // 日志轮转策略：必须在配置加载后立刻生效（否则配置里的 max_bytes 要等到
  // 第一条热路径日志才被读取）。若当前文件已超限，set_log_rotation 会先归档
  ai_gateway::detail::set_log_rotation(
      ai_gateway::detail::LogRotation{cfg.log.max_bytes, cfg.log.keep_files});
  // 热路径日志采样：必须在服务开始处理请求前生效（默认 1 = 全量）
  ai_gateway::detail::set_log_sample_every(cfg.log.sample_every);
  if (cfg.log.sample_every > 1)
    LOG_WARN("log sampling enabled: 1 of every {} hot-path INFO lines is kept",
             cfg.log.sample_every);
  LOG_INFO("log rotation: max_bytes={} keep_files={} (0 = disabled)",
           cfg.log.max_bytes, cfg.log.keep_files);

  // ---- 2. 初始化模块 ----
  // ttl_days * 86400 是 int 乘法：ttl_days > 24855 会溢出成负数，
  // 于是"永不过期"被静默打开（报告 L5）
  constexpr int kMaxTtlDays = 3650;  // 10 年，超出视为配置错误
  if (cfg.cache.ttl_days < 0 || cfg.cache.ttl_days > kMaxTtlDays) {
    LOG_ERROR("cache.ttl_days={} out of range [0, {}]", cfg.cache.ttl_days,
              kMaxTtlDays);
    return static_cast<int>(ErrorCode::kConfigError);
  }
  const int64_t ttl_seconds = static_cast<int64_t>(cfg.cache.ttl_days) * 86400;
  auto lru = std::make_shared<LruStore>(cfg.cache.max_entries, ttl_seconds);

  // 本地 ONNX 嵌入推理：模型路径与维度来自 embedding 配置段（不再硬编码，
  // 否则示例里的模型名永远不会生效，报告 M15）
  auto onnx_embed = std::make_shared<OnnxEmbedding>(
      cfg.embedding.model_path, cfg.embedding.vocab_path, cfg.embedding.dim);
  // 分词大小写与池化方式必须在**任何 encode() 之前**设置：
  // 构造期做的那次"维度探测"推理也会用到它们，更关键的是随后计算指纹时
  // 要如实反映"这批向量是怎么产生的"（见 embedding_variant_id）
  onnx_embed->set_do_lower_case(cfg.embedding.do_lower_case);
  onnx_embed->set_pooling(cfg.embedding.pooling);

  // 维度不符属配置错误：启动即失败（旧实现按 min(dims, out_dim) 静默截断，
  // 索引维度与配置声明不一致且没有任何告警，报告 M15）。
  // 模型文件缺失等不可用情形仍降级为精确匹配，避免"没有模型就不能跑"
  if (onnx_embed->load_error() ==
      OnnxEmbedding::LoadError::kDimensionMismatch) {
    LOG_ERROR("embedding.dim={} does not match {} (model output dim={}), "
              "fix config or model", cfg.embedding.dim,
              cfg.embedding.model_path, onnx_embed->output_dim());
    return static_cast<int>(ErrorCode::kConfigError);
  }
  if (!onnx_embed->ready()) {
    LOG_WARN("onnx embedding unavailable ({}), semantic cache degrades to "
             "exact match", cfg.embedding.model_path);
  }

  auto embed_fn = [onnx_embed](const std::string& text,
                                int) -> std::vector<float> {
    return onnx_embed->ready() ? onnx_embed->encode(text)
                                : std::vector<float>{};
  };

  // 索引由缓存引擎独占持有：rebuild_index() 会替换 index_，main 不再保留副本，
  // 否则会长期持有一个已失效的旧索引对象（日志里的向量数也会取自旧对象）
  auto engine = std::make_shared<CacheEngine>(
      cfg.embedding, cfg.cache, lru,
      std::make_shared<HnswIndex>(
          HnswConfig{cfg.embedding.dim, 16, 100, 50}),
      embed_fn);
  auto stats = std::make_shared<Stats>();
  // 输入/输出分档单价：配置缺省时是 0.001/0.001（与旧口径完全一致）
  stats->set_pricing(TokenPricing{cfg.cost.input_per_1k, cfg.cost.output_per_1k});
  auto filter = std::make_shared<MessageFilter>(cfg.filter);

  // ---- 2.5 embedding 指纹：向量只对"产生它的模型"有意义 ----
  // 换模型或改分词（本项目刚改过 WordPiece 规则）之后，旧向量与新查询向量不在
  // 同一个空间里，余弦相似度失去意义，且**不会报错**——只会静默返回错误答案。
  // 因此落盘时记录指纹、加载时比对；不一致就丢弃向量（保留文本）并按新模型重建。
  EmbeddingFingerprint fp;
  if (cfg.cache.enabled && onnx_embed->ready()) {
    const std::string variant = embedding_variant_id(cfg.embedding);
    fp = make_embedding_fingerprint(cfg.embedding.model_path,
                                    cfg.embedding.vocab_path, cfg.embedding.dim,
                                    variant);
    LOG_INFO("cache: embedding variant {}", variant);
    if (!fp.valid()) {
      // 模型能加载却算不出文件哈希（权限/IO 异常）：不能把空指纹当成"匹配"
      LOG_WARN("cache: embedding fingerprint unavailable "
               "(model={} vocab={}), vector origin will NOT be verified",
               cfg.embedding.model_path, cfg.embedding.vocab_path);
    } else {
      lru->set_fingerprint(fp.to_string());
      LOG_INFO("cache: embedding fingerprint {}", fp.to_string());
    }
  }

  // 从磁盘恢复缓存
  if (cfg.cache.enabled) {
    // 返回值必须检查：路径不可读/文件损坏时 load 会失败，静默忽略会让"重启不丢缓存"
    // 的说法失真（报告 M9）
    if (!lru->load("cache/lru_store.json"))
      LOG_WARN("cache: load from cache/lru_store.json failed "
               "(missing or malformed), starting empty");
    // 指纹判定结果必须显式落日志：这是"静默串答案"唯一的可观测面
    if (lru->fingerprint_mismatch()) {
      if (!lru->loaded_file_had_fingerprint()) {
        LOG_WARN("cache: lru_store.json has no embedding fingerprint (legacy "
                 "format): dropped {} vector(s), kept text entries; index will "
                 "be rebuilt with the current model",
                 lru->dropped_vectors());
      } else {
        LOG_WARN("cache: embedding fingerprint mismatch: file=[{}] current=[{}] "
                 "-> dropped {} vector(s), kept text entries, rebuilding index",
                 lru->loaded_fingerprint(),
                 fp.valid() ? fp.to_string() : std::string("<unavailable>"),
                 lru->dropped_vectors());
      }
    } else if (fp.valid() && lru->loaded_file_had_fingerprint()) {
      LOG_INFO("cache: embedding fingerprint verified ({} vector entries)",
               lru->size());
    }
    // 加载后立即清理过期条目：落盘时仍有效、此后超过 TTL 的条目不应继续占用内存与索引
    size_t purged_on_load = lru->purge_expired();
    // 索引由 LruStore 重建，无需独立加载;
    engine->rebuild_index();
    if (lru->size() == 0) {
      LOG_INFO("cache restored: 0 entries (缓存为空或条目均已超过 ttl_days={})",
               cfg.cache.ttl_days);
    } else {
      LOG_INFO("cache restored: {} entries, {} vectors{}", lru->size(),
               engine->index_size(),
               purged_on_load > 0
                   ? std::format(", {} expired purged", purged_on_load)
                   : "");
    }
  }

  struct sigaction sa{};
  sa.sa_handler = handle_signal;
  sa.sa_flags = SA_RESTART;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  // SIGUSR1：按需导出统计到日志（`kill -USR1 <pid>`，最迟 60s 由后台线程输出）
  sigaction(SIGUSR1, &sa, nullptr);

  // ---- 3. 启动定期统计 + 定时持久化线程 ----
  std::thread bg_thread([stats, lru, engine, &cfg] {
    while (!g_shutdown.load(std::memory_order_acquire)) {
      {
        std::unique_lock lk(g_bg_mutex);
        g_bg_cv.wait_for(lk, 60s,
                         [] { return g_shutdown.load(std::memory_order_acquire); });
      }
      if (g_shutdown.load(std::memory_order_acquire)) break;
      if (g_dump_stats.exchange(false, std::memory_order_acq_rel))
        LOG_INFO("stats dump requested by SIGUSR1");
      stats->report();
      if (cfg.cache.enabled) engine->try_rebuild_if_ghosty();
      // 主动清理过期条目：避免失效条目长期占用内存，并让落盘内容只含有效条目
      if (cfg.cache.enabled) {
        size_t purged = lru->purge_expired();
        if (purged > 0) {
          // HNSW 无删除接口：清理后重建索引，保持索引与 LruStore 一致
          engine->rebuild_index();
          LOG_INFO("cache: purged {} expired entries, {} remaining", purged,
                   lru->size());
        }
      }
      // 定期持久化缓存，避免宕机丢了cache
      if (cfg.cache.enabled && lru->size() > 0) {
        if (!lru->save("cache/lru_store.json"))
          LOG_ERROR("cache: periodic save to cache/lru_store.json failed");
        // 持久化空桩（HNSW 索引通过 LruStore 重建）;
      }
    }
  });

  // ---- 4. 构建 HTTP 服务 + 注册处理器 ----
  HttpServer server(cfg.server);
  g_shutdown.store(false, std::memory_order_release);

  Singleflight flight_merge;
  // handle_request 的返回值是 pair<HttpReply, keep_alive>，而路由层只接受
  // HttpReply（keep-alive 由连接处理器按请求头自行判定）。流式响应在
  // 连接处理器里走的是"writer.committed() 就直接返回"的旁路，不依赖这里返回的
  // HttpReply——因此 keep_alive 这一项在本路由上是多余的，丢弃即可
  server.set_handler([&](const std::string& body, ResponseWriter& writer,
                         const HttpRequestInfo& info) -> HttpReply {
    // 兜底：线程池 worker 里逃出的异常会直接 std::terminate 整个进程，
    // 因此任何异常都必须在这里被拦住并转成一个普通错误响应。
    // 注意：异常若发生在流式响应已经开始写之后，这里无法回退已发出的响应头
    // （客户端会看到一个被截断的流），只能保证进程存活——见简报"已知限制"
    try {
      // 客户端是否希望复用连接：由连接处理器从版本 + Connection 头解析后传入。
      // 响应头的 Connection 与"是否真的复用"必须用同一份意愿
      return handle_request(body, cfg, filter.get(), engine.get(),
                            &flight_merge, stats.get(), writer, info.keep_alive)
          .first;
    } catch (const std::exception& e) {
      LOG_ERROR("request handler threw: {}", e.what());
      return HttpReply{500, "application/json", R"({"error":"Internal error"})"};
    } catch (...) {
      LOG_ERROR("request handler threw non-std exception");
      return HttpReply{500, "application/json", R"({"error":"Internal error"})"};
    }
  });

  // GET /metrics：Prometheus 文本格式抓取端点。
  //   - 与 POST /v1/chat/completions 是两条独立路由（key = "METHOD /path"），
  //     新增它不改变既有行为（连接处理器只对"没有对应 method 路由的非 POST"
  //     返回 405，见 connection_handler.cpp）
  //   - 只暴露聚合计数与分位数，不含任何请求内容／哈希，脱敏口径与日志一致
  //   - 不读 active_connections()：conns_ 由 reactor 线程独占，worker 里读是数据竞争
  const auto process_start = std::chrono::steady_clock::now();
  server.add_route("GET", "/metrics",
                   [stats, &server, process_start](const std::string&,
                                                   ResponseWriter&,
                                                   const HttpRequestInfo&) {
                     auto uptime =
                         std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - process_start)
                             .count();
                     auto body = render_metrics(
                         *stats, uptime,
                         static_cast<int64_t>(server.pending_tasks()),
                         static_cast<int64_t>(server.active_tasks()),
                         static_cast<int64_t>(server.worker_threads()),
                         static_cast<int64_t>(server.accepted_connections()));
                     return HttpReply{200, kMetricsContentType, std::move(body)};
                   });

  // ---- 5. 启动 ----
  LOG_INFO("ai-gateway starting on :{}, backend={}", cfg.server.port, cfg.backend.url);
  // 传入关闭标志：否则 SIGTERM/SIGINT 只置位而无人检查，优雅退出（保存缓存）永不执行
  server.run(&g_shutdown);

  // ---- 6. 清理：保存缓存 + 统计 ----
  g_shutdown.store(true, std::memory_order_release);
  g_bg_cv.notify_one();  // 唤醒 bg_thread 避免等待 60s 超时
  if (bg_thread.joinable()) bg_thread.join();
  stats->report();
  if (cfg.cache.enabled) {
    size_t purged = lru->purge_expired();
    if (!lru->save("cache/lru_store.json"))
      LOG_ERROR("cache: final save to cache/lru_store.json failed "
                "({} entries were not persisted)",
                lru->size());
    // 持久化空桩（HNSW 索引通过 LruStore 重建）;
    LOG_INFO("cache persisted: {} entries, {} vectors{}", lru->size(),
             engine->index_size(),
             purged > 0 ? std::format(", {} expired purged", purged) : "");
  }
  LOG_INFO("cache hits={} misses={} hit_rate={:.1f}%",
           stats->cache_hits(), stats->cache_misses(),
           stats->hit_rate() * 100);
  LOG_INFO("ai-gateway stopped");
  return 0;
}
