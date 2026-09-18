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
//
// 覆盖范围不只是网关自己生成的状态码：上游的 4xx/5xx 是原样透传的
// （见 handle_request），所以上游常见的 401/403/422 等也必须有名有姓，
// 否则响应行会打印成 "HTTP/1.1 413 Unknown"（报告 8.7 第 7 条）。
// 未逐条列出的码按类别兜底，最差也能给出 "4xx Client Error" 这类正确描述。
inline std::string_view status_text(int code) noexcept {
  switch (code) {
    // 2xx
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 206: return "Partial Content";
    // 3xx
    case 301: return "Moved Permanently";
    case 302: return "Found";
    case 304: return "Not Modified";
    case 307: return "Temporary Redirect";
    case 308: return "Permanent Redirect";
    // 4xx
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 402: return "Payment Required";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 406: return "Not Acceptable";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 410: return "Gone";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 414: return "URI Too Long";
    case 415: return "Unsupported Media Type";
    case 422: return "Unprocessable Entity";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    // 5xx
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    case 505: return "HTTP Version Not Supported";
    default: break;
  }
  // 类别兜底：不必为每个码维护文案，但绝不能出现 "Unknown"
  if (code >= 100 && code < 200) return "Informational";
  if (code >= 200 && code < 300) return "Success";
  if (code >= 300 && code < 400) return "Redirection";
  if (code >= 400 && code < 500) return "Client Error";
  if (code >= 500 && code < 600) return "Server Error";
  return "Unknown";
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
