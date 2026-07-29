// LLM API 客户端：转发请求到 OpenAI 兼容后端，返回原始 JSON 响应体
#pragma once

#include <string>

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
                     const std::string& model,
                     const std::string& request_body,
                     int timeout_seconds);

}  // namespace ai_gateway
