// URL 路由：将 HTTP path 转 handler 映射
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "common/types.h"

namespace ai_gateway {

// 请求头（method, path） 是否匹配 + 对应 handler
class Router {
 public:
  // 处理器返回完整响应内容（含状态码），而非仅响应体字符串
  using Handler = std::function<HttpReply(const std::string& body)>;

  // 注册路由：method + path
  void add(std::string_view method, std::string_view path, Handler handler);

  // 查找路由：匹配 method + path，返回 handler；未匹配返回 nullptr
  const Handler* find(std::string_view method, std::string_view path) const;

 private:
  // 路由 key 为 "METHOD /path" 格式
  std::unordered_map<std::string, Handler> routes_;
};

}  // namespace ai_gateway
