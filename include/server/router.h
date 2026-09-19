// URL 路由：将 HTTP path 转 handler 映射
#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "common/types.h"
#include "server/response_writer.h"

namespace ai_gateway {

// 请求里"handler 可能需要、但不在 body 里"的部分。
// 目前只有一项：客户端的连接复用意愿（由版本 + Connection 头决定）。
// 为什么必须传给 handler：流式响应由 handler 自己写响应头，Connection 头必须
// 如实反映这个意愿，否则客户端会被挂死在"等下一个响应"上
//
// **这是双向字段**（报告 M3）：入参是客户端意愿，出参是"handler 认为这个响应
// 是否允许复用"。生产管道（gateway/pipeline.cpp）会按响应内容改写它——输入被拒
// 400、输出被拦 502、上游 5xx、流式请求走非流式兜底这四类都不允许复用；连接
// 处理器在这之后读它，并与客户端意愿取合取（connection_handler.cpp 第 4 步之后）。
// 旧实现在管道里算出了这个决策（HandleOutcome 的第二个字段）却在路由适配层丢弃，
// 于是"上游 5xx 不复用""流式兜底不复用"两条注释与实现不一致：实测 stream_fallback
// 仍回 `Connection: keep-alive` 且连接真被复用。
//
// 注：写死线的口径（流式用空闲超时 / 缓冲式用总死线）**不在这里**传递——
// 它由 worker 层（http_server.cpp）从请求体里直接预判 `"stream"` 子串得到。
// 理由：那是连接层写路径自己的参数，路由 handler 从不需要读它；为省一次子串
// 扫描而把它做成跨层字段，只会让"谁决定死线口径"分散在两处
struct HttpRequestInfo {
  bool keep_alive = false;
};

// 请求头（method, path） 是否匹配 + 对应 handler
class Router {
 public:
  // 处理器返回完整响应内容（含状态码），而非仅响应体字符串。
  // 第二个参数是响应写出器：缓冲式响应完全用不到它（只 return），
  // 需要边收边发的 SSE 透传才直接往 writer 写
  using Handler = std::function<HttpReply(
      const std::string& body, ResponseWriter& writer,
      HttpRequestInfo& info)>;

  // 注册路由：method + path
  void add(std::string_view method, std::string_view path, Handler handler);

  // 查找路由：匹配 method + path，返回 handler；未匹配返回 nullptr
  const Handler* find(std::string_view method, std::string_view path) const;

 private:
  // 路由 key 为 "METHOD /path" 格式
  std::unordered_map<std::string, Handler> routes_;
};

}  // namespace ai_gateway
