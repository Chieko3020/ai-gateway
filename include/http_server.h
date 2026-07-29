// HTTP 服务器核心：epoll ET + 非阻塞 IO，主线程同步处理
//
// 架构：
//   主线程 accept + epoll_wait + ConnectionHandler 处理数据 + send 响应
//

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "config.h"
#include "connection_handler.h"
#include "router.h"

namespace ai_gateway {

// 请求处理器签名：接收请求体 JSON，返回响应体 JSON
using RequestHandler = std::function<std::string(const std::string& request_body)>;

class HttpServer {
 public:
  explicit HttpServer(const ServerConfig& config);
  ~HttpServer();

  // 禁止拷贝/移动（管理 fd 等资源）
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // 注册路由处理器（默认注册到 POST /v1/chat/completions）
  void set_handler(RequestHandler handler);

  // 注册自定义路由
  void add_route(std::string_view path, RequestHandler handler);

  // 启动服务（阻塞当前线程直到 stop() 被调用）
  void run();

  // 停止服务
  void stop();

 private:
  // 创建非阻塞监听 socket
  int create_listen_socket();

  // 处理单个客户端连接
  void handle_client(int client_fd);

  // 设置 fd 为非阻塞
  static void set_nonblocking(int fd);

  ServerConfig config_;
  int listen_fd_ = -1;
  int epoll_fd_ = -1;
  std::atomic<bool> running_{false};
  Router router_;
  ConnectionHandler conn_handler_{router_};
};

}  // namespace ai_gateway
