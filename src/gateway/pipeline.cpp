// 网关请求管道实现。这里就是生产路径本身（main.cpp 与单测链接同一份对象代码），
// 拆分理由与边界见 include/gateway/pipeline.h 顶部说明。
//
// 本文件从 src/main.cpp 搬移而来（纯移动，仅把对外接口的 static 去掉），
// 目的是让 tests/test_pipeline.cpp 能直接链接生产管道；main.cpp 只保留进程
// 生命周期（配置加载、模块装配、信号、优雅关闭顺序）。

#include "gateway/pipeline.h"

#include <chrono>
#include <format>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "backend/llm_client.h"
#include "cache/entity_tokens.h"
#include "common/logger.h"
#include "common/singleflight.h"
#include "server/sse_capture.h"
#include "server/sse_usage.h"

namespace ai_gateway {

using json = nlohmann::json;

static uint64_t fnv1a_64(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ull;
  }
  return h;
}
// 向量产生方式的完整标识（进 embedding 指纹）：
//   分词器实现 + 是否小写化 + 池化方式
// 这三者任何一个改变，**同一段文本产生的向量就不同**，旧落盘向量与新查询向量
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
std::string extract_namespace(const std::string& request_body) {
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

// 处理结果的统一形状：pair<HttpReply, 能否复用连接>
//
// writer 只在流式（SSE 真透传）路径上被使用：那条路径要边收边发，不能等整段
// 响应体凑齐。缓冲式路径只用返回值。
//
// 为什么 keep-alive 要跟 HttpReply 一起返回：流式响应由 handle_stream_request
// 自己写响应头，keep-alive 的决定必须与 Connection 头在同一处产生，否则会出现
// "头里写了 keep-alive、连接却被关掉"（或反之）的不一致。这里的 keep_alive
// 由连接处理器与客户端意愿取合取（见 include/gateway/pipeline.h 的说明）。

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

  // 开启"上游原始 SSE 字节"捕获（流式写回缓存用）。
  // 为什么默认关：绝大多数流不需要在网关里留副本（纯白占内存）；只有
  // "可缓存 + 未命中、准备回填"的请求才开。
  // max_bytes 是保护上限：超长流放弃回填而不是把内存交给一条长回答，
  // 溢出后 overflowed() 为真，调用方据此跳过写缓存
  void enable_capture(size_t max_bytes) {
    capture_ = true;
    capture_max_ = max_bytes;
  }
  const std::string& captured() const { return captured_; }
  bool capture_overflowed() const { return capture_overflowed_; }

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
    if (capture_) {
      if (captured_.size() + len <= capture_max_)
        captured_.append(data, len);
      else
        capture_overflowed_ = true;  // 超上限：这条流不回填缓存
    }
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
      // 命中屏蔽规则：停止继续放行（上游同样被中断，不再为被拦内容付费）。
      // 但**不要**直接断流——那样客户端 SSE 解析器看到的是"连接异常中断"。
      // 发一个协议内的错误事件 + 终止事件，让它正常收尾。提示里不回显违规内容
      // （那是二次泄漏）
      filter_rejected_ = true;
      const std::string tail =
          std::format("data: {{\"error\":\"response filtered\"}}\n\n{}",
                      kSseDoneEvent);
      (void)writer_->write_body(tail);
      return false;
    }
    return true;
  }

  void on_done(StreamAbortReason reason) override {
    // curl 侧的中止原因必须原样留下来（报告 M1）。
    // 旧实现是 `(void)reason;`，于是"上游静默被空闲死线中停"在 handler 眼里与
    // "上游正常发完"完全一样：不打 WARN、不计 streams_abort、还把这个超时值算进
    // TTFT 样本池，而客户端收到的是一个"看起来正常结束但没有 [DONE]"的流
    // （OpenAI 兼容 SDK 报 "Stream ended without finish_reason"，网关侧却查不到线索）。
    // kClientGone 由 on_chunk 的写失败路径设置，这里不覆盖已有信息
    if (reason != StreamAbortReason::kNone && relay_abort_ == StreamAbortReason::kNone)
      relay_abort_ = reason;
    if (write_failed_) return;  // 已因写失败退出，上游传输也已被中断
    // 冲刷尾部：上游结束时最后一段可能没有以空行结尾
    std::string tail = filter_->sse_feed(filter_state_, {}, /*final_chunk=*/true);
    if (!tail.empty() && !writer_->write_body(tail)) {
      write_failed_ = true;
      abort_ = writer_->abort_reason();
    }
    if (filter_state_.rejected) filter_rejected_ = true;
  }

  // ---- 观测数据（handler 收尾时读取） ----
  bool head_sent() const { return head_sent_; }
  bool done_seen() const { return done_seen_; }
  bool filter_rejected() const { return filter_rejected_; }
  bool write_failed() const { return write_failed_; }
  // 中止原因 = "写客户端失败的原因"（abort_，由 ResponseWriter 给出）或
  // "curl 侧中止传输的原因"（relay_abort_，上游空闲/总时限/客户端断开），
  // 两者取先有值的那个。为什么要合并：写失败时 on_done 会提前返回、
  // relay_abort_ 可能只有 on_chunk 设的 kClientGone，而 detail 仍在 abort_ 里
  StreamAbortReason relay_abort_reason() const { return relay_abort_; }
  AbortReason abort_reason() const { return abort_; }
  int64_t first_byte_ms() const { return first_byte_ms_; }
  size_t bytes_in() const { return bytes_in_; }
  const StreamUsage& usage() const { return usage_.usage(); }
  const std::string& rejected_event() const {
    return filter_state_.rejected_event;
  }

  // 本次流是否被**上游侧**中止（而不是客户端侧）。
  // 上游静默（kUpstreamIdle）与上游超时（kDeadline）属于这一类：客户端并没有
  // 走开，是网关按死线主动中停了上游 —— 这是"防线生效"的证据，必须与
  // "客户端提前断开"分开统计与告警
  bool upstream_aborted() const {
    return relay_abort_ == StreamAbortReason::kUpstreamIdle ||
           relay_abort_ == StreamAbortReason::kDeadline;
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
  // curl 侧的中止原因（on_done 的入参），与 abort_ 分开保存：
  // 两者语义不同（一个来自写客户端，一个来自上游/传输层），合并会丢掉诊断信息
  StreamAbortReason relay_abort_ = StreamAbortReason::kNone;
  // 原始 SSE 字节捕获（默认关，见 enable_capture）
  bool capture_ = false;
  bool capture_overflowed_ = false;
  size_t capture_max_ = 0;
  std::string captured_;
};

// 流式中止原因的可读文本：日志与告警按它分类（报告 M1）。
// 顺序有讲究——**上游侧中止优先于"客户端断开"**：过滤拒绝与上游静默这两条路径
// 上 curl 侧看到的原因都是"写回调返回 0 ⇒ kClientGone"，但真实原因分别是
// "命中屏蔽规则"与"上游空闲死线到点"。先报具体原因，运维才不会把它误读成
// "客户端自己断了"（旧实现既不区分、也不告警，这条防线在可观测面上完全隐形）
std::string stream_abort_detail(const SsePassthroughSink& sink) {
  if (sink.filter_rejected()) return "output filter rejected the SSE event";
  switch (sink.relay_abort_reason()) {
    case StreamAbortReason::kUpstreamIdle:
      return "upstream idle timeout (no data between chunks)";
    case StreamAbortReason::kDeadline:
      return "upstream total deadline";
    case StreamAbortReason::kClientGone:
      return "client gone / write failed";
    case StreamAbortReason::kNullSink:
      return "internal: null sink";
    case StreamAbortReason::kNone:
      break;
  }
  // curl 侧没给原因：那就是写客户端时失败的（写死线 / 客户端断开）
  switch (sink.abort_reason()) {
    case AbortReason::kIdle:
      return "write idle timeout";
    case AbortReason::kDeadline:
      return "write deadline";
    case AbortReason::kClientGone:
      return "client gone";
    default:
      return "unknown";
  }
}


}  // namespace

// 流式命中缓存：把当初记下的**上游原始 SSE 字节**回放给客户端。
//
// 为什么回放字节而不是"把文本重新合成为事件"：流式响应没有可以改写的 JSON 容器，
// 合成一条流要自己造 finish_reason、usage 与事件切分，产出的只是"看起来像流"；
// 回放原始字节则与上游输出同源（data: 前缀、多事件结构、usage、[DONE] 全都在）。
HandleOutcome serve_stream_cache_hit(const GatewayConfig& /*cfg*/,
                                     MessageFilter* filter, Stats* stats,
                                     ResponseWriter& writer,
                                     std::chrono::steady_clock::time_point t0,
                                     bool head_keep_alive,
                                     const std::string& key, float similarity,
                                     const std::string& sse) {
  // 命中回放同样要过输出过滤，而且判定口径与写入侧**完全一致**：把缓存的字节
  // 当作"一条待过滤的流"整体跑一遍 sse_feed（按事件边界逐条判定，与写入时同一套）。
  //
  // 为什么不能省这一步：缓存里存的是**上游原文**（流式路径的 captured_ 是在
  // sse_feed 过滤之前拷走的），非流式命中也是每次重新过滤——否则"先让含 URL 的
  // 答案入缓存、再命中"就能绕过输出过滤。此前流式命中漏了这一步（两条路径口径
  // 不一致），这里补齐。
  MessageFilter::SseFilterState check_state;
  std::string payload = filter->sse_feed(check_state, sse, /*final_chunk=*/true);
  if (check_state.rejected) {
    // 与非流式命中被拦时的处理保持一致：整条拒绝，且**在写响应头之前**判定，
    // 所以客户端看到的是一个干净的 502，而不是"已经被污染的半截流"
    LOG_WARN("stream: cached payload rejected by output filter ({} bytes)",
             sse.size());
    return {HttpReply{502, "application/json", R"({"error":"Response filtered"})"},
            false};
  }
  // 流式响应头由网关自己构造（此刻并不存在上游），因此可以带上缓存探针头。
  // 头必须立刻发：客户端要看到 200 + text/event-stream 才开始处理事件
  if (!writer.write_stream_head(
          200, "text/event-stream; charset=utf-8", head_keep_alive,
          std::format("X-Cache: {}\r\n", kCacheStatusHit))) {
    LOG_WARN("stream: cache HIT key={} but response head not sent (client gone)",
             key);
    return {HttpReply{200, "text/event-stream", {}}, false};
  }
  // SSE 注释行：规范允许、客户端默认忽略 —— 让 `curl -N` 这种"看得见的流"里
  // 也能一眼看出命中，不必去翻响应头或 /metrics
  std::string body = std::format(": cache {}\n\n", kCacheStatusHit);
  body += payload;  // 过滤后的字节（可能丢弃了命中规则的事件）
  // 缓存字节里理应含 [DONE]（只有正常流完的流才会回填）；万一没有（人为构造的
  // 缓存文件、旧格式），补一个，否则客户端等不到流结束标志
  if (!sse_has_done(body)) body += std::string(kSseDoneEvent);
  const bool written = writer.write_body(body);
  const bool finished = written && writer.finish_stream();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);
  stats->record_stream_hit(elapsed.count());
  LOG_INFO_SAMPLED(
      "stream: cache HIT key={} sim={:.3f} bytes={} elapsed={}ms keep_alive={}",
      key, similarity, sse.size(), elapsed.count(), finished ? "yes" : "no");
  // 响应已经写出去了，返回的 HttpReply 不会再被发送（连接处理器看 committed()）
  return {HttpReply{200, "text/event-stream", {}}, kStreamKeepAlive && finished};
}

// 处理 stream:true：把上游 SSE 逐块透传给客户端
HandleOutcome handle_stream_request(
    const std::string& request_body, const GatewayConfig& cfg,
    MessageFilter* filter, CacheEngine* engine, Stats* stats,
    ResponseWriter& writer, std::chrono::steady_clock::time_point t0,
    bool client_wants_keep_alive) {
  // 响应头的 Connection 必须如实反映客户端意愿（客户端显式要求 close 时
  // 写 keep-alive 会把它挂死在"等下一个响应"上）；真正是否复用由连接处理器
  // 用同一份意愿决定
  const bool head_keep_alive = kStreamKeepAlive && client_wants_keep_alive;
  const bool request_includes_usage = stream_includes_usage(request_body);

  // ---- 流式语义缓存：命中则回放，未命中边回源边捕获 ------------------------
  // 可缓存判定与非流式路径同源：缓存开启 + 非工具请求 + 有 user_message。
  // 工具请求（带 tools）一律旁路：同一句话的答案取决于工具执行结果
  const bool may_cache = engine != nullptr && cfg.cache.enabled &&
                         !is_tool_request(request_body);
  std::string user_message = may_cache ? extract_user_message(request_body)
                                       : std::string{};
  const std::string ns = may_cache ? extract_namespace(request_body)
                                   : std::string{};
  const bool cacheable_stream = may_cache && !user_message.empty();
  std::vector<float> embedding;  // 未命中时由 try_hit 带回，回填时复用
  if (cacheable_stream) {
    auto hit = engine->try_hit(user_message, ns);
    if (hit.hit) {
      auto payload = engine->sse_of(hit.key);
      if (payload.has_value() && !payload->empty())
        return serve_stream_cache_hit(cfg, filter, stats, writer, t0,
                                      head_keep_alive, hit.key, hit.similarity,
                                      *payload);
      // 命中但没有 SSE 字节（早于本功能写入的条目 / 非流式路径写入的条目）：
      // 不能把非流式 JSON 当流发出去，按未命中回源，回源后再补一条带字节的
      LOG_INFO_SAMPLED(
          "cache: HIT key={} sim={:.3f} but no SSE payload, relaying upstream",
          hit.key, hit.similarity);
    }
    embedding = std::move(hit.embedding);
  }

  SsePassthroughSink sink(filter, &writer, t0, head_keep_alive);
  // 只有准备回填的请求才捕获原始字节（默认关，避免给每条流都留一份副本）
  if (cacheable_stream) sink.enable_capture(kMaxSseCaptureBytes);
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
    // 中止 = 写客户端失败，或 curl 侧中止了上游传输（报告 M1）。
    // 后者以前被 sink 丢掉，于是"上游静默被中停"被当成正常完成
    const bool aborted = sink.write_failed() || sink.upstream_aborted() ||
                         sink.abort_reason() != AbortReason::kNone;
    bool keep_alive = kStreamKeepAlive && finished && !aborted;

    if (aborted) {
      // 客户端提前断开 / 写死线到点 / 上游静默被中停：不进延迟样本池
      // （口径见 stats.h——TTFT 只统计正常完成的流，否则被中止的流会把它的
      //  等待时长算成"首字节延迟"，污染分位数）
      stats->record_stream_aborted();
      LOG_WARN("stream: aborted after {} bytes ({}), keep_alive={} done_event={}",
               sink.bytes_in(), stream_abort_detail(sink),
               keep_alive ? "yes" : "no", sink.done_seen() ? "yes" : "no");
    } else {
      // token 统计：usage 在最后一个 SSE 事件里（且只在客户端带了
      // stream_options.include_usage 时上游才会发）。SseUsageParser 旁路解析这份
      // 用量，透传的字节一个都没动。上游没给 usage 时**优雅退化为 0**，
      // 并用 streams_no_usage 单列计数——"网关没解析"与"上游没给"在报表里可区分
      const StreamUsage usage = sink.usage();
      if (!usage.seen) stats->record_stream_no_usage();
      // 延迟口径：进样本池的是首字节延迟 TTFT，total 只进日志。
      // 经过缓存查询的流式流量记 miss（进命中率分母），工具请求等仍记 bypass——
      // 两者混算会让命中率的分子分母口径不一致
      if (cacheable_stream)
        stats->record_stream_miss(sink.first_byte_ms(), elapsed.count(),
                                  static_cast<int>(usage.prompt_tokens),
                                  static_cast<int>(usage.completion_tokens));
      else
        stats->record_stream(sink.first_byte_ms(), elapsed.count(),
                             static_cast<int>(usage.prompt_tokens),
                             static_cast<int>(usage.completion_tokens));

      // ---- 回填缓存：只有**正常流完**（拿到 [DONE]）的流才写 ----
      // 客户端中途断开留下的半截答案不能固化 30 天；捕获超出上限的巨流也不写
      if (cacheable_stream && !sink.capture_overflowed()) {
        const std::string& raw = sink.captured();
        std::string answer = extract_sse_content(raw);
        if (!answer.empty() && sse_has_done(raw)) {
          engine->cache_reply(user_message, answer, embedding, ns, raw);
          stats->record_cache_write();
          LOG_INFO_SAMPLED(
              "stream: cache filled text_len={} sse_bytes={} ns={}",
              answer.size(), raw.size(), ns.empty() ? "default" : ns);
        } else if (!answer.empty()) {
          // 有内容但没有终止事件：多半是上游断流，不写缓存但留痕（便于区分
          // "没抓到内容"与"抓到了但流不完整"）
          LOG_INFO_SAMPLED("stream: skip cache fill (no [DONE], bytes={})",
                           raw.size());
        }
      }
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

HandleOutcome handle_request(const std::string& request_body,
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
    return handle_stream_request(request_body, cfg, filter, engine, stats,
                                 writer, t0,
                                 client_wants_keep_alive);
  }

  // 缓存命中检查时带回的 embedding（避免 cache_reply 重复计算）
  std::vector<float> cached_embedding;

  // 4. 语义缓存
  std::string ns;
  std::string ns_key;
  // namespace 前缀（"ns<hash>:" 或空串）：缓存命中路径用它二次过滤候选，
  // 请求合并路径也用它做同一维度的隔离（报告 H2，见 singleflight.h 文件头）
  std::string ns_prefix;
  if (cfg.cache.enabled && !user_msg.empty()) {
    ns = extract_namespace(request_body);
    ns_prefix = ns.empty() ? std::string{} : ns + ":";
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
    // 合并候选的两道闸（口径与缓存命中路径一致）：
    //   实体：从 user_msg 提标记，不带 namespace 前缀（前缀自身的十六进制会被
    //         当成混合标识符）
    //   namespace：ns_prefix 由本请求自己的 system prompt 决定；无 system 时为空串，
    //         此时只允许与同样无 ns 的在途条目合并（报告 H2）
    const EntityTokens query_entities =
        cfg.cache.entity_veto ? extract_entity_tokens(user_msg) : EntityTokens{};
    auto fut = sf->try_merge(ns_key, cached_embedding, ns_prefix, query_entities,
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
  //
  // 只有 leader 写缓存，第 5 步被合并的请求（拿到 merged 结果后直接返回）**不写**。
  // 这是有意的取舍而不是遗漏，结论与依据（第六轮评估，代码未改）：
  //
  //   1. 合并判定比缓存命中判定更严格：合并要求 cosine ≥ 0.95 且实体一致，
  //      命中只要求 cosine ≥ similarity_threshold（默认 0.85）且实体一致，两者
  //      对实体的判定是同一套规则。因此"能合并到 leader"的请求，其文本也必然能以
  //      leader 的条目为候选通过命中判定 ⇒ leader 那条条目天然覆盖跟随者，
  //      跟随者不需要自己的条目。
  //      实证（慢上游 + 8 条同义改写并发）：一条只被合并过、自己没有条目的文本
  //      单独重发时返回 _cache=hit，取到的正是 leader 的答案。
  //   2. 让跟随者各写一条的代价：一批同义请求会产出 N 条近重复向量（实测边际
  //      约 13.8KB/条），挤占 max_entries、加速 LRU 淘汰掉真正不同的问答，并把
  //      HNSW 的近邻区域塞满重复向量、放大幽灵向量（索引里有节点、store 里没有）。
  //      收益只是在"leader 条目被淘汰/过期"时多一层冗余，而那层冗余在容量压力下
  //      同样会被淘汰，且淘汰的是别人。
  //   3. 反方向的证伪：修好第 5 步的实体校验后，跟随者不再是与"别人的问题"共享
  //      答案，而 1 万条灌入的第二轮命中率从 40.1% 升到 99.7%——说明"合并削弱缓存"
  //      这个现象的主因是**误合并**（每个被误合并的请求都没落盘 + leader 的条目对
  //      它也不可用），不是"跟随者没写自己的条目"。
  //   已知边界：跟随者继承的是 leader 条目的写入时间与 TTL，leader 条目被淘汰时
  //   跟随者一起失去覆盖；这是可接受的代价。
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

}  // namespace ai_gateway
