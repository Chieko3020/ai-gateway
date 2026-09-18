// 连接处理器实现
#include "server/connection_handler.h"

#include "common/logger.h"
#include "server/http_server.h"  // ResponseWriter 完整定义
#include "server/response.h"

namespace ai_gateway {

ConnectionResult ConnectionHandler::process(const char* raw_data, size_t len,
                                            ResponseWriter& writer) {
  ConnectionResult result;

  // 1. 解析 HTTP 请求
  auto req = parse_request(raw_data, len);
  // keep-alive 的客户端意愿：解析失败时无法判定版本，保守按"不复用"处理
  const bool client_keep_alive = req.valid && req.wants_keep_alive();

  if (!req.valid) {
    result.response = make_response(400, "application/json",
                                    R"({"error":"Invalid request"})");
    result.keep_alive = false;
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
    result.keep_alive = client_keep_alive;
    return result;
  }

  // 3. 路径不存在
  if (!handler) {
    result.response =
        make_response(404, "application/json", R"({"error":"Not found"})");
    result.keep_alive = client_keep_alive;
    return result;
  }

  // 4. 调用 handler 获取响应
  //    状态码由 handler 决定：上游 4xx/5xx 原样透传给客户端（不再统一 200）
  HttpRequestInfo info;
  info.keep_alive = client_keep_alive;
  auto reply = (*handler)(std::string(req.body), writer, info);

  // 5. 流式响应已经由 handler 自己写完了（writer.committed()）：
  //    这里不能再发一个报文，否则客户端会读到"两个响应"。
  //    keep-alive 必须仍然遵守客户端的 Connection 头——忘记这一条会让显式
  //    要求 Connection: close 的客户端被挂住，等一个永远不会到来的 EOF
  if (writer.committed()) {
    result.keep_alive = client_keep_alive && !writer.failed();
    return result;
  }

  // 6. 缓冲式响应：套上 Content-Length + Connection 头
  const bool keep_alive = client_keep_alive;
  ResponseHeader h;
  h.status_code = reply.status_code;
  h.content_type = reply.content_type;
  h.content_length = reply.body.size();
  h.keep_alive = keep_alive;
  result.response = build_response_head(h) + reply.body;
  result.keep_alive = keep_alive;
  return result;
}

}  // namespace ai_gateway
