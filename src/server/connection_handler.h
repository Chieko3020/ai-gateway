// 连接处理器：解析 → 路由 → 处理 → 响应（从 HttpServer 分离）
#pragma once

#include <functional>
#include <string>
#include <string_view>

#include "request.h"
#include "router.h"

namespace ai_gateway {

// 处理一个完整的 HTTP 请求-响应周期的结果：要发送的响应数据
struct ConnectionResult {
  std::string response;
  bool should_close = true;
};

class ConnectionHandler {
 public:
  using Handler = std::function<std::string(const std::string&)>;

  explicit ConnectionHandler(Router& router) : router_(router) {}

  // 设置默认 handler（由 main.cpp 注入）
  void set_default_handler(Handler h) { default_handler_ = std::move(h); }

  // 处理一次请求：raw_bytes 是 recv 到的原始数据
  ConnectionResult process(const char* raw_data, size_t len);

 private:
  Router& router_;
  Handler default_handler_;
};

}  // namespace ai_gateway
