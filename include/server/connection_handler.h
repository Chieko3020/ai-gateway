// 连接处理器：解析 路由 处理 响应
#pragma once

#include <string>

#include "server/request.h"
#include "server/response_writer.h"
#include "server/router.h"

namespace ai_gateway {

// 处理一个完整的 HTTP 请求-响应周期的结果
struct ConnectionResult {
  std::string response;  // 缓冲式响应的完整报文（流式响应时为空）
  // 该连接是否可以在本次响应后继续复用（HTTP/1.1 持久连接）。
  // 这里已经合并了三个判断：请求的版本/Connection 头意愿、handler 是否允许、
  // 以及流式响应是否写完。worker 直接照此执行（kKeepAlive / kClose）
  bool keep_alive = false;
};

class ConnectionHandler {
 public:
  explicit ConnectionHandler(Router& router) : router_(router) {}

  // 处理一次请求：raw_bytes 是 recv 到的原始数据。
  // writer 用于增量写出（流式响应）；缓冲式响应只写 response 字段由调用方发。
  // 返回的 keep_alive 已把"响应是否支持复用"与"请求是否允许复用"合并判断，
  // 且 writer 的 Connection 头与之保持一致
  ConnectionResult process(const char* raw_data, size_t len,
                           ResponseWriter& writer);

 private:
  Router& router_;
};

}  // namespace ai_gateway
