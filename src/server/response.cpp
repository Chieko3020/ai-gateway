// HTTP 响应构造实现
#include "response.h"

#include <format>

#include "types.h"

namespace ai_gateway {

std::string make_response(int status_code,
                          std::string content_type,
                          std::string body) {
  return std::format(
      "HTTP/1.1 {} {}\r\n"
      "Content-Type: {}\r\n"
      "Content-Length: {}\r\n"
      "Connection: close\r\n"
      "\r\n"
      "{}",
      status_code, status_text(status_code),
      content_type,
      body.size(),
      body);
}

}  // namespace ai_gateway
