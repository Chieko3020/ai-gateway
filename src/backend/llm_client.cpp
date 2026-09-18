// LLM API 客户端：转发请求到 OpenAI 兼容后端
//
// 连接复用与并发：每个线程持有一个自己的 CURL*（thread_local），彼此不共享句柄，
// 因此不需要覆盖 curl_easy_perform 的全局锁——旧实现用「单句柄 + 全局 mutex」，
// 把全进程的未命中请求串行化（吞吐上限 = 1 / 上游延迟，报告 H6）。
// thread_local 的清理用显式生命周期钩子完成，不能让 thread_local 对象的析构在
// curl_global_cleanup 之后运行。
#include "backend/llm_client.h"

#include <curl/curl.h>

#include <format>
#include <memory>
#include <mutex>
#include <string>
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

}  // namespace ai_gateway
