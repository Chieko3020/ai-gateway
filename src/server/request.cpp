// HTTP 请求解析 HTTP/1.1 请求行 + 头部 + 正文
#include "server/request.h"

#include <charconv>
#include <cstring>
#include <string>

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

  // 提取 path（剥离 query string：路由表是精确匹配，
  // `POST /v1/chat/completions?x=1` 不剥 query 会直接 404，报告 L6）
  auto sp2 = request_line.find(' ', sp1 + 1);
  if (sp2 == std::string_view::npos) return req;
  auto target = request_line.substr(sp1 + 1, sp2 - sp1 - 1);
  auto qmark = target.find('?');
  req.path = (qmark == std::string_view::npos) ? target : target.substr(0, qmark);

  // ---- 2. 头部: "Key: Value\r\n" 直到空行 ----
  while (pos < len) {
    auto eol = data.find("\r\n", pos);
    if (eol == std::string_view::npos) return req;

    std::string_view line = data.substr(pos, eol - pos);
    pos = eol + 2;

    // 空行表示头部结束
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
  // Transfer-Encoding: chunked 显式拒绝：本实现没有 chunked 解码，
  // 旧注释写"取剩余数据兼容 chunked"，实际会把分块长度行原样转发给上游（报告 M7）
  auto te = req.header("Transfer-Encoding");
  if (!te.empty()) {
    // 只支持 identity；chunked 等其它编码一律判为无效请求（上层回 400）
    std::string lower;
    lower.reserve(te.size());
    for (unsigned char c : te) {
      if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c - 'A' + 'a');
      lower.push_back(static_cast<char>(c));
    }
    // 允许 "identity" 以及 identity 列表形式（如 "identity, identity"）
    bool only_identity = true;
    size_t pos = 0;
    while (pos <= lower.size()) {
      auto comma = lower.find(',', pos);
      auto item = lower.substr(pos, comma == std::string::npos
                                        ? std::string::npos
                                        : comma - pos);
      // 去首尾空格
      auto b = item.find_first_not_of(" \t");
      auto e = item.find_last_not_of(" \t");
      item = (b == std::string::npos) ? "" : item.substr(b, e - b + 1);
      if (item != "identity") { only_identity = false; break; }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    if (!only_identity) return req;  // valid = false
  }

  // 读取 Content-Length 决定正文大小
  auto cl_str = req.header("Content-Length");
  if (!cl_str.empty()) {
    auto [ptr, ec] = std::from_chars(cl_str.data(), cl_str.data() + cl_str.size(),
                                   req.content_length);
    if (ec != std::errc()) req.content_length = 0;
  }

  if (req.content_length > 0) {
    if (pos + req.content_length > len) {
      req.valid = false;
      return req;
    }
    req.body = data.substr(pos, req.content_length);
  } else if (pos < len) {
    // 无 Content-Length（且已排除 chunked）：取剩余数据为 body
    req.body = data.substr(pos);
    req.content_length = req.body.size();
  }

  req.valid = true;
  return req;
}

}  // namespace ai_gateway
