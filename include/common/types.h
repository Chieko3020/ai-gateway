// 公共类型定义：错误码、请求上下文等
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ai_gateway {

// 网关自定义错误码，零表示成功
enum class ErrorCode : int {
  kOk = 0,
  kConfigError = 1,
  kSocketError = 2,
  kBindError = 3,
  kBackendTimeout = 4,
  kBackendUnreachable = 5,
  kInvalidRequest = 6,
  kInternalError = 99,
};

// 返回 HTTP 状态码对应的描述
inline std::string_view status_text(int code) noexcept {
  switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    default:  return "Unknown";
  }
}

// 请求处理器返回的响应内容：状态码由处理器决定。
// 上游返回 4xx/5xx 时原样透传，不再一律以 200 返回——
// 否则调用方无法用状态码做重试/降级，监控会把上游故障全部计成成功。
struct HttpReply {
  int status_code = 200;
  std::string content_type = "application/json";
  std::string body;
};

// 请求处理上下文，在一次请求生命周期中传递
// 注意：当前未启用，预留用于请求追踪（request_id + start_time_us）
struct RequestContext {
  int64_t start_time_us = 0; // 请求到达时刻（微秒时间戳）
};

}  // namespace ai_gateway
