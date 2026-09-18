// 连接处理器实现
#include "server/connection_handler.h"

#include "common/logger.h"
#include "server/response.h"

namespace ai_gateway {

ConnectionResult ConnectionHandler::process(const char* raw_data,
                                             size_t len) {
  ConnectionResult result;

  // 1. 解析 HTTP 请求
  auto req = parse_request(raw_data, len);
  if (!req.valid) {
    result.response = make_bad_request(R"({"error":"Invalid request"})");
    return result;
  }

  // 2. 仅允许 POST
  if (req.method != "POST") {
    result.response = make_response(405, "application/json",
                                    R"({"error":"Method not allowed"})");
    return result;
  }

  // 3. 路由查找
  auto* handler = router_.find("POST", req.path);
  if (!handler) {
    result.response =
        make_response(404, "application/json", R"({"error":"Not found"})");
    return result;
  }

  // 4. 调用 handler 获取响应
  //    状态码由 handler 决定：上游 4xx/5xx 原样透传给客户端（不再统一 200）
  auto reply = (*handler)(std::string(req.body));
  result.response = make_response(reply.status_code,
                                  std::move(reply.content_type),
                                  std::move(reply.body));
  return result;
}

}  // namespace ai_gateway
