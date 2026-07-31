// 连接处理器：解析 路由 处理 响应
#pragma once

#include <string>

#include "server/request.h"
#include "server/router.h"

namespace ai_gateway {

// 处理一个完整的 HTTP 请求-响应周期的结果：要发送的响应数据
struct ConnectionResult {
  std::string response;
};

class ConnectionHandler {
 public:

  explicit ConnectionHandler(Router& router) : router_(router) {}

  // 处理一次请求：raw_bytes 是 recv 到的原始数据
  ConnectionResult process(const char* raw_data, size_t len);

 private:
  Router& router_;
};

}  // namespace ai_gateway
