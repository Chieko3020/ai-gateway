// HTTP 服务器核心：epoll ET + 非阻塞 IO + 线程池
//
// 架构：
//   主线程 accept + epoll_wait 收事件 → 线程池处理业务逻辑
//
// 设计决策：
//   - 使用 ET 模式减少 epoll 事件通知次数
//   - 线程池和 epoll 在同一线程模型下配合（主线程负责 IO，工作线程负责处理）
//   - 不使用 Boost/Muduo：复用 HttpFramework 已验证的架构，保持零重依赖
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/config.h"

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

  // 注册请求处理函数
  void set_handler(RequestHandler handler);

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
  RequestHandler handler_;
  std::vector<std::thread> workers_;
  bool running_ = false;
};

}  // namespace ai_gateway
