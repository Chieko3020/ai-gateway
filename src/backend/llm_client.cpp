// LLM API 客户端：转发请求到 OpenAI 兼容后端
#include "ai-gateway/backend/llm_client.h"

#include <format>

#include "ai-gateway/common/curl_client.h"
#include "ai-gateway/common/logger.h"

namespace ai_gateway {

LlmResponse call_llm(const std::string& url,
                     const std::string& api_key,
                     const std::string& request_body,
                     int timeout_seconds) {
  LlmResponse result;
  CurlClient curl;
  if (!curl) {
    LOG_ERROR("curl_easy_init failed");
    return result;
  }

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
    result.body = std::format(R"({{"error":"{}"}})", curl_easy_strerror(res));
  }

  return result;
}

}  // namespace ai_gateway
