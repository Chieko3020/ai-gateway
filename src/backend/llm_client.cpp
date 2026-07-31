// LLM API 客户端：转发请求到 OpenAI 兼容后端
// 使用持久 CurlClient 复用 TCP 连接（Keep-Alive）
// 全局连接池由 mutex 保护，多线程安全
#include "backend/llm_client.h"

#include <format>
#include <memory>
#include <mutex>

#include "common/curl_client.h"
#include "common/logger.h"

namespace ai_gateway {

namespace {
// 持久连接池：单句柄复用，由 mutex 保护并发访问
std::unique_ptr<CurlClient> g_llm_client;
std::mutex g_llm_mutex;

static CurlClient& pool() {
  if (!g_llm_client) g_llm_client = std::make_unique<CurlClient>();
  return *g_llm_client;
}
}  // namespace

LlmResponse call_llm(const std::string& url,
                     const std::string& api_key,
                     const std::string& request_body,
                     int timeout_seconds) {
  LlmResponse result;

  // 全局连接池需要序列化访问（libcurl 句柄非线程安全）
  std::lock_guard lock(g_llm_mutex);

  auto& curl = pool();
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
    LOG_ERROR("curl request failed: {}", curl_easy_strerror(res));
    result.body = std::format(R"(\{{\"error\":\"{}\"}})", curl_easy_strerror(res));
  }

  return result;
}

}  // namespace ai_gateway
