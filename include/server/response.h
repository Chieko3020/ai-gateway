// HTTP 响应构造：生成符合 HTTP/1.1 规范的响应报文
//
// 两种长度语义：
//   1. Content-Length（普通 JSON 响应）—— 连接可 keep-alive 复用
//   2. Transfer-Encoding: chunked（流式/SSE 响应）—— 逐块发送，靠 0 长度块结束，
//      连接同样可 keep-alive
// 第三种（无长度、靠 close 界定响应体结束）本实现不采用，理由见 http_server.h
// 顶部的"流式 + keep-alive"注释。
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ai_gateway {

// 响应头的构造参数
struct ResponseHeader {
  int status_code = 200;
  std::string content_type = "text/event-stream";
  // 普通响应：正文字节数；流式响应：kChunkedLength（或把 chunked 置 true）
  size_t content_length = 0;
  // 是否在响应头里声明 Connection: keep-alive。
  // 注意这只是"声明"，真正决定连接是否复用是 worker 写完后的处置动作
  bool keep_alive = true;
  bool chunked = false;
};

// 流式响应用这个哨兵长度：不写 Content-Length，改写 Transfer-Encoding: chunked
inline constexpr size_t kChunkedLength = static_cast<size_t>(-1);

// 构造响应头（含结尾空行）
std::string build_response_head(const ResponseHeader& h);

// 把一个响应体分块编码（"<hex>\r\n<data>\r\n"）。空块返回空串
// （HTTP 不允许 0 长度块，0 块专用于终止）
std::string encode_chunk(std::string_view data);

// chunked 终止块
inline constexpr const char* kChunkedTerminator = "0\r\n\r\n";

// 构造完整的 HTTP 响应字符串（可直接 send()）。
// 保持"恒 Content-Length + Connection: close"语义，用于连接即将关闭的场景
// （503 连接数超限、413 body 过大等由 reactor 直接发出的拒绝响应）
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

inline std::string make_payload_too_large(std::string body) {
  return make_response(413, "application/json", std::move(body));
}

}  // namespace ai_gateway
