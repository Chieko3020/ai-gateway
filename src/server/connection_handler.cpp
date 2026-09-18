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

  // 2. 路由查找：按 "METHOD /path" 精确匹配。
  //    不能再用"非 POST 一律 405"——那样 GET /metrics 也会被拒（报告 8.7 第 4 条），
  //    而 Prometheus 抓取端点是 GET。既有语义要保住：
  //      请求的 method 没有对应路由 且 不是 POST  -> 405（与旧行为一致）
  //      POST 但路径不存在                        -> 404（与旧行为一致）
  auto* handler = router_.find(req.method, req.path);
  if (!handler && req.method != "POST") {
    result.response = make_response(405, "application/json",
                                    R"({"error":"Method not allowed"})");
    return result;
  }

  // 3. 路径不存在
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
