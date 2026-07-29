// HTTP 请求解析：从原始字节流中提取 method、path、headers、body
#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

namespace ai_gateway {

// 解析结果 不拥有数据（数据在原始缓冲区中）
struct ParsedRequest {
  std::string_view method;
  std::string_view path;
  std::string_view body;
  std::unordered_map<std::string_view, std::string_view> headers;
  bool valid = false;
  size_t content_length = 0;

  // 便捷查询
  std::string_view header(std::string_view key) const;
};

// 从原始 HTTP 数据解析请求
// raw_data 指针必须指向完整的 HTTP 请求报文
// 解析失败时 valid=false
ParsedRequest parse_request(const char* raw_data, size_t len);

}  // namespace ai_gateway
