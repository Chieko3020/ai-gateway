// 连接处理器实现
#include "connection_handler.h"

#include "logger.h"
#include "response.h"

namespace ai_gateway {

ConnectionResult ConnectionHandler::process(const char* raw_data,
                                             size_t len) {
  ConnectionResult result;

  // 1. 解析 HTTP 请求
  LOG_INFO("http raw_len={} cl_pos={}", len, std::string_view(raw_data, len).find("Content-Length"));
  auto req = parse_request(raw_data, len);
  LOG_INFO("http parsed valid={} cl={} body_len={}", req.valid, req.content_length, req.body.size());
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
  std::string response_body = (*handler)(std::string(req.body));
  result.response = make_ok_json(std::move(response_body));
  return result;
}

}  // namespace ai_gateway
