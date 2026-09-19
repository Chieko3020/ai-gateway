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

  // 4.5 handler 回写"这个响应是否允许复用"（报告 M3）。
  //     生产管道按响应内容决定：输入被拒 400 / 输出被拦 502 / 上游 5xx /
  //     流式请求走非流式兜底 → false。它必须与客户端意愿取合取，否则响应头里
  //     写的 Connection 与"是否真的关连接"会脱节：
  //     旧实现把它丢掉、用 client_keep_alive 重算，于是 4 处显式决策全部失效。
  //     handler 不改写时保持客户端意愿不变（网关自行生成的其它路由如 /metrics
  //     就属于这一类）
  const bool server_ok_with_reuse = info.keep_alive;
  info.keep_alive = client_keep_alive && server_ok_with_reuse;

  // 5. 流式响应已经由 handler 自己写完了（writer.committed()）：
  //    这里不能再发一个报文，否则客户端会读到"两个响应"。
  //    这条路径上响应头（含 Connection）已经在管道里发出去了，因此这里**只能**
  //    采用管道回写的值（再与 writer.failed() 取合取），不能自己重算：否则会出现
  //    "响应头写了 keep-alive、连接却被关掉"（或反之）
  if (writer.committed()) {
    result.keep_alive = info.keep_alive && !writer.failed();
    return result;
  }

  // 6. 缓冲式响应：套上 Content-Length + Connection 头
  //    响应头里的 Connection 与 result.keep_alive 用的是同一个值（同一处产生）
  const bool keep_alive = info.keep_alive;
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
