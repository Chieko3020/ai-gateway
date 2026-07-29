// HTTP 请求解析实现：手工解析 HTTP/1.1 请求行 + 头部 + 正文
#include "request.h"

#include <charconv>
#include <cstring>

namespace ai_gateway {

std::string_view ParsedRequest::header(std::string_view key) const {
  auto it = headers.find(key);
  return (it != headers.end()) ? it->second : std::string_view{};
}

ParsedRequest parse_request(const char* raw_data, size_t len) {
  ParsedRequest req;
  if (!raw_data || len == 0) return req;

  std::string_view data(raw_data, len);

  // ---- 1. 请求行: "METHOD /path HTTP/1.1\r\n" ----
  auto line_end = data.find("\r\n");
  if (line_end == std::string_view::npos) return req;

  std::string_view request_line = data.substr(0, line_end);
  size_t pos = line_end + 2;  // 跳过 \r\n

  // 提取 method
  auto sp1 = request_line.find(' ');
  if (sp1 == std::string_view::npos) return req;
  req.method = request_line.substr(0, sp1);

  // 提取 path
  auto sp2 = request_line.find(' ', sp1 + 1);
  if (sp2 == std::string_view::npos) return req;
  req.path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

  // ---- 2. 头部: "Key: Value\r\n" 直到空行 ----
  while (pos < len) {
    auto eol = data.find("\r\n", pos);
    if (eol == std::string_view::npos) return req;

    std::string_view line = data.substr(pos, eol - pos);
    pos = eol + 2;

    // 空行 → 头部结束
    if (line.empty()) break;

    auto colon = line.find(':');
    if (colon == std::string_view::npos) continue;

    auto key = line.substr(0, colon);
    // 跳过冒号后的空格
    auto val_start = colon + 1;
    while (val_start < line.size() && line[val_start] == ' ') ++val_start;
    auto value = line.substr(val_start);

    req.headers[key] = value;
  }

  // ---- 3. 正文 (body) ----
  // 读取 Content-Length 决定正文大小
  auto cl_str = req.header("Content-Length");
  if (!cl_str.empty()) {
    auto [ptr, ec] = std::from_chars(cl_str.data(), cl_str.data() + cl_str.size(),
                                   req.content_length);
    if (ec != std::errc()) req.content_length = 0;
  }

  if (req.content_length > 0) {
    if (pos + req.content_length > len) {
      req.valid = false;  // 声明长度与实际不符 → 拒绝
      return req;
    }
    req.body = data.substr(pos, req.content_length);
  }

  req.valid = true;
  return req;
}

}  // namespace ai_gateway
