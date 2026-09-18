// LLM API 客户端：转发请求到 OpenAI 兼容后端
//
// 连接复用与并发：每个线程持有一个自己的 CURL*（thread_local），彼此不共享句柄，
// 因此不需要覆盖 curl_easy_perform 的全局锁——旧实现用「单句柄 + 全局 mutex」，
// 把全进程的未命中请求串行化（吞吐上限 = 1 / 上游延迟，报告 H6）。
// thread_local 的清理用显式生命周期钩子完成，不能让 thread_local 对象的析构在
// curl_global_cleanup 之后运行。
//
// 流式路径（call_llm_stream）复用同一个句柄池：write 回调里逐块把上游字节交给
// LlmStreamSink，而不是像 call_llm 那样追加到 std::string。二者共用 URL/超时/
// 鉴权配置，区别只在 write 回调与 Content-Type 决策。
#include "backend/llm_client.h"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "common/curl_client.h"
#include "common/logger.h"

namespace ai_gateway {

namespace {

// libcurl 全局初始化：必须在任何线程创建 CURL* 之前完成且只做一次
void ensure_curl_global_init() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (rc != CURLE_OK)
      LOG_ERROR("curl_global_init failed: {}", curl_easy_strerror(rc));
  });
}

// 每个线程一个句柄：保留底层 TCP 连接复用（reset() 只清句柄状态，不断连接），
// 线程退出时由 ThreadLocalCurl 的析构做 curl_easy_cleanup
CurlClient& thread_local_pool() {
  ensure_curl_global_init();
  thread_local std::unique_ptr<CurlClient> client;
  if (!client) client = std::make_unique<CurlClient>();
  return *client;
}

// ===========================================================================
// 流式路径
// ===========================================================================

// 上游"连上了但不再发字节"最多容忍多久（秒）。CURLOPT_TIMEOUT 是整条流的总
// 时限，对僵死的长连接没有约束力，必须另设低速中断，否则一个早夭流会把 worker
// 一直占住（见简报的生命周期一节）。
constexpr long kStreamStallSeconds = 30;

std::string to_lower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s)
    out.push_back(static_cast<char>(std::tolower(c)));
  return out;
}

void trim_inplace(std::string& s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  auto b = std::find_if(s.begin(), s.end(), not_space);
  auto e = std::find_if(s.rbegin(), s.rend(), not_space).base();
  if (b >= e) {
    s.clear();
    return;
  }
  s = std::string(b, e);
}

// 一次流式调用的共享状态。只在同一个 worker 线程里访问
// （curl 回调跑在 curl_easy_perform 的调用线程上），不需要同步。
struct StreamCtx {
  LlmStreamSink* sink = nullptr;
  bool streaming = false;   // 已确定的路径：true = 逐块透传
  bool decided = false;     // 是否已就 Content-Type 做出决策
  int status_code = 0;      // 从状态行解析
  std::string content_type; // 上游 Content-Type 原文
  std::string buffer;       // 非流式路径的响应体
  StreamAbortReason abort = StreamAbortReason::kNone;
  bool header_done = false;
};

size_t stream_header_cb(char* buffer, size_t size, size_t nitems,
                        void* userdata) {
  auto* ctx = static_cast<StreamCtx*>(userdata);
  const size_t len = size * nitems;
  std::string_view line(buffer, len);

  if (line.rfind("HTTP/", 0) == 0) {
    // 状态行 "HTTP/1.1 200 OK"：只取状态码数字段
    auto sp = line.find(' ');
    if (sp != std::string_view::npos) {
      int code = 0;
      for (char c : line.substr(sp + 1)) {
        if (c < '0' || c > '9') break;
        code = code * 10 + (c - '0');
      }
      if (code >= 100) {
        ctx->status_code = code;
        // 1xx（100-continue）不是最终响应，别把它的头当成正文头
        ctx->header_done = false;
      }
    }
    return len;
  }

  if (len == 0 || line == "\r\n" || line == "\n") {
    ctx->header_done = true;
    return len;
  }
  if (ctx->header_done) return len;

  constexpr std::string_view kKey = "content-type:";
  if (len >= kKey.size() && to_lower(line.substr(0, kKey.size())) == kKey) {
    std::string value(line.substr(kKey.size()));
    trim_inplace(value);
    if (ctx->content_type.empty()) ctx->content_type = value;
  }
  return len;
}

size_t stream_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* ctx = static_cast<StreamCtx*>(userdata);
  const size_t len = size * nmemb;
  if (!ctx) return len;

  // 首次拿到响应体时决定走哪条路：
  //   - 2xx 且 Content-Type 是 text/event-stream -> 逐块透传给下游
  //   - 其它（含上游 4xx/5xx 的 JSON 错误体、Content-Type 缺失的怪响应）
  //     -> 缓冲成普通响应，保证错误体仍是完整可解析的 JSON
  if (!ctx->decided) {
    ctx->decided = true;
    const bool ok_status = ctx->status_code >= 200 && ctx->status_code < 300;
    const bool sse =
        to_lower(ctx->content_type).find("text/event-stream") !=
        std::string::npos;
    if (ok_status && sse && ctx->sink &&
        ctx->sink->on_begin(ctx->status_code, ctx->content_type))
      ctx->streaming = true;
  }

  if (!ctx->streaming) {
    ctx->buffer.append(ptr, len);
    return len;
  }
  if (!ctx->sink->on_chunk(ptr, len)) {
    // 下游写不进去了（客户端断开 / 写死线）：返回 0 让 curl 立刻中止传输，
    // 不再为没人要的响应消耗上游 token
    ctx->abort = StreamAbortReason::kClientGone;
    return 0;
  }
  return len;
}

}  // namespace

LlmResponse call_llm(const std::string& url,
                     const std::string& api_key,
                     const std::string& request_body,
                     int timeout_seconds) {
  LlmResponse result;

  // 本线程独占该句柄：无锁，4 个 worker 可以真正并发地等各自的上游响应
  auto& curl = thread_local_pool();
  if (!curl) {
    LOG_ERROR("curl_easy_init failed");
    return result;
  }

  // 每次请求重置句柄状态，但保留底层 TCP 连接
  curl.reset();

  curl.set_url(url);
  curl.set_post_body(request_body);
  curl.set_timeout(timeout_seconds);
  curl.set_auth(api_key);
  curl.set_response_target(&result.body);

  long http_code = 0;
  CURLcode res = curl.perform(&http_code);
  if (res == CURLE_OK) {
    result.status_code = static_cast<int>(http_code);
  } else {
    // curl 自身失败（连不上 / 超时）：没有上游状态码，必须映射为网关错误码，
    // 否则 status_code 保持 0 会被上层当成正常码，最终以 HTTP 200 返回给客户端
    result.status_code =
        (res == CURLE_OPERATION_TIMEDOUT) ? 504 : 502;
    LOG_ERROR("curl request failed: {} (mapped to HTTP {})",
              curl_easy_strerror(res), result.status_code);
    result.body = std::format(R"(\{{\"error\":\"{}\"}})", curl_easy_strerror(res));
  }

  return result;
}

StreamCallResult call_llm_stream(const std::string& url,
                                 const std::string& api_key,
                                 const std::string& request_body,
                                 int timeout_seconds,
                                 LlmStreamSink* sink) {
  StreamCallResult out;
  auto& curl = thread_local_pool();
  if (!curl) {
    LOG_ERROR("curl_easy_init failed");
    out.curl_error = "curl_easy_init failed";
    return out;
  }
  if (!sink) {
    // 编程错误：没有接收方就别浪费上游额度
    LOG_ERROR("call_llm_stream called with null sink");
    out.curl_error = "null sink";
    out.abort_reason = StreamAbortReason::kNullSink;
    return out;
  }

  // 重置句柄状态，但保留底层 TCP 连接（与 call_llm 同一套复用模型）
  curl.reset();
  curl.set_url(url);
  curl.set_post_body(request_body);
  curl.set_timeout(timeout_seconds);
  curl.set_auth(api_key);

  CURL* h = curl.raw();
  if (!h) {
    out.curl_error = "curl handle unavailable";
    return out;
  }

  StreamCtx ctx;
  ctx.sink = sink;

  curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, stream_header_cb);
  curl_easy_setopt(h, CURLOPT_HEADERDATA, &ctx);
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, stream_write_cb);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, &ctx);
  // 低速中断：连续 kStreamStallSeconds 秒速率低于 1 字节/秒即中止。
  // 与 CURLOPT_TIMEOUT 的总时限互补，专治"连上了却不发数据"的僵死流
  curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1L);
  curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, kStreamStallSeconds);

  long http_code = 0;
  CURLcode res = curl.perform_raw(&http_code);
  if (ctx.status_code == 0 && http_code > 0)
    ctx.status_code = static_cast<int>(http_code);

  out.status_code = ctx.status_code;
  out.content_type = ctx.content_type;
  out.streamed = ctx.streaming;
  out.body = std::move(ctx.buffer);
  out.abort_reason = ctx.abort;
  if (res != CURLE_OK) out.curl_error = curl_easy_strerror(res);

  if (ctx.streaming) {
    if (ctx.abort == StreamAbortReason::kNone && res == CURLE_WRITE_ERROR)
      ctx.abort = StreamAbortReason::kClientGone;
    if (ctx.abort == StreamAbortReason::kNone &&
        res == CURLE_OPERATION_TIMEDOUT)
      ctx.abort = StreamAbortReason::kDeadline;
    out.abort_reason = ctx.abort;
    sink->on_done(ctx.abort);
  } else if (res != CURLE_OK) {
    // 非流式路径下 curl 失败：没有可用响应体，映射成网关错误码
    // （与 call_llm 同一口径：超时 504、其余 502）
    out.status_code = (res == CURLE_OPERATION_TIMEDOUT) ? 504 : 502;
    out.body = std::format(R"({{"error":"{}"}})", curl_easy_strerror(res));
    LOG_ERROR("curl stream request failed: {} (mapped to HTTP {})",
              curl_easy_strerror(res), out.status_code);
  }

  return out;
}

}  // namespace ai_gateway
