// URL 路由实现
#include "router.h"

#include <format>

namespace ai_gateway {

void Router::add(std::string_view path, Handler handler) {
  routes_[std::format("POST {}", path)] = std::move(handler);
}

const Router::Handler* Router::find(std::string_view method,
                                     std::string_view path) const {
  auto it = routes_.find(std::format("{} {}", method, path));
  return (it != routes_.end()) ? &it->second : nullptr;
}

}  // namespace ai_gateway
