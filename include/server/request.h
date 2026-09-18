// HTTP 请求解析：从原始字节流中提取 method、path、headers、body
#pragma once

#include <string>
#include <string_view>
#include <unordered_map>

namespace ai_gateway {

// 头部查表的大小写不敏感键：HTTP 头名大小写不敏感，
// 精确匹配的 unordered_map 会让 "content-length" / "CONTENT-TYPE" 查不到（报告 M7）
struct CaseInsensitiveHash {
  size_t operator()(std::string_view s) const noexcept {
    size_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
      if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c - 'A' + 'a');
      h ^= c;
      h *= 1099511628211ull;
    }
    return h;
  }
};

struct CaseInsensitiveEqual {
  bool operator()(std::string_view a, std::string_view b) const noexcept {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      unsigned char ca = static_cast<unsigned char>(a[i]);
      unsigned char cb = static_cast<unsigned char>(b[i]);
      if (ca >= 'A' && ca <= 'Z') ca = static_cast<unsigned char>(ca - 'A' + 'a');
      if (cb >= 'A' && cb <= 'Z') cb = static_cast<unsigned char>(cb - 'A' + 'a');
      if (ca != cb) return false;
    }
    return true;
  }
};

// 解析结果 不拥有数据（数据在原始缓冲区中）
struct ParsedRequest {
  std::string_view method;
  std::string_view path;
  std::string_view body;
  std::unordered_map<std::string_view, std::string_view,
                     CaseInsensitiveHash, CaseInsensitiveEqual>
      headers;
  bool valid = false;
  size_t content_length = 0;

  // 便捷查询（大小写不敏感）
  std::string_view header(std::string_view key) const;
};

// 从原始 HTTP 数据解析请求
// raw_data 指针必须指向完整的 HTTP 请求报文
// 解析失败时 valid=false
ParsedRequest parse_request(const char* raw_data, size_t len);

}  // namespace ai_gateway
