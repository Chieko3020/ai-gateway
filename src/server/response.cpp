// HTTP 响应构造实现
#include "server/response.h"

#include <format>

#include "common/types.h"

namespace ai_gateway {

std::string build_response_head(const ResponseHeader& h) {
  const bool chunked = h.chunked || h.content_length == kChunkedLength;
  std::string out = std::format("HTTP/1.1 {} {}\r\n", h.status_code,
                                status_text(h.status_code));
  // 上游 Content-Type 原样透传（流式响应必须是 text/event-stream，
  // 统一成 application/json 会让下游 SSE 解析器直接失效）
  out += std::format("Content-Type: {}\r\n",
                     h.content_type.empty() ? "application/json"
                                            : h.content_type);
  if (chunked)
    out += "Transfer-Encoding: chunked\r\n";
  else
    out += std::format("Content-Length: {}\r\n", h.content_length);
  out += h.keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
  out += "\r\n";
  return out;
}

std::string encode_chunk(std::string_view data) {
  if (data.empty()) return {};
  std::string out;
  out.reserve(data.size() + 20);
  out += std::format("{:x}\r\n", data.size());
  out.append(data);
  out += "\r\n";
  return out;
}

std::string make_response(int status_code,
                          std::string content_type,
                          std::string body) {
  ResponseHeader h;
  h.status_code = status_code;
  h.content_type = std::move(content_type);
  h.content_length = body.size();
  h.keep_alive = false;  // 连接即将关闭：诚实声明，别让客户端以为还能复用
  return build_response_head(h) + body;
}

}  // namespace ai_gateway
