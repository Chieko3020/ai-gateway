// HTTP 响应构造：生成符合 HTTP/1.1 规范的响应报文
#pragma once

#include <string>

namespace ai_gateway {

// 构造完整的 HTTP 响应字符串（可直接 send()）
std::string make_response(int status_code,
                          std::string content_type,
                          std::string body);

// 快捷响应
inline std::string make_ok_json(std::string body) {
  return make_response(200, "application/json", std::move(body));
}

inline std::string make_bad_request(std::string body) {
  return make_response(400, "application/json", std::move(body));
}

inline std::string make_service_unavailable(std::string body) {
  return make_response(503, "application/json", std::move(body));
}

}  // namespace ai_gateway
