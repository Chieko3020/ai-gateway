// HTTP 服务器核心实现
#include "http_server.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <format>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sstream>

#include "common/logger.h"
#include "common/types.h"
#include "request.h"
#include "response.h"
#include "router.h"

namespace ai_gateway {

namespace {
constexpr int kMaxEvents = 64;
constexpr int kBacklog = 128;
constexpr size_t kBufSize = 65536;  // 64KB 缓冲区，足够大多数 HTTP 请求
}  // namespace

HttpServer::HttpServer(const ServerConfig& config)
    : config_(config) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::set_handler(RequestHandler handler) {
  handler_ = std::move(handler);
  router_.add("/v1/chat/completions", handler_);
}

void HttpServer::add_route(std::string_view path, RequestHandler handler) {
  router_.add(path, std::move(handler));
}

void HttpServer::set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int HttpServer::create_listen_socket() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    LOG_ERROR("socket() failed: {}", std::strerror(errno));
    return -1;
  }

  // SO_REUSEADDR 允许快速重启（避免 TIME_WAIT 阻塞 bind）
  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  set_nonblocking(fd);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(config_.port));

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    LOG_ERROR("bind() failed: {}", std::strerror(errno));
    close(fd);
    return -1;
  }

  if (listen(fd, kBacklog) < 0) {
    LOG_ERROR("listen() failed: {}", std::strerror(errno));
    close(fd);
    return -1;
  }

  LOG_INFO("listening on port {}", config_.port);
  return fd;
}

void HttpServer::run() {
  listen_fd_ = create_listen_socket();
  if (listen_fd_ < 0) return;

  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ < 0) {
    LOG_ERROR("epoll_create1 failed: {}", std::strerror(errno));
    close(listen_fd_);
    return;
  }

  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;  // 边缘触发
  ev.data.fd = listen_fd_;
  epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev);

  running_ = true;  // 工作线程已移除，但标志位仍需设置
  epoll_event events[kMaxEvents];

  while (running_) {
    // 短暂超时以便检查 running_ 标志
    int nfds = epoll_wait(epoll_fd_, events, kMaxEvents, 100);
    if (nfds < 0) {
      if (errno == EINTR) continue;
      LOG_ERROR("epoll_wait failed: {}", std::strerror(errno));
      break;
    }

    for (int i = 0; i < nfds; ++i) {
      int fd = events[i].data.fd;

      if (fd == listen_fd_) {
        // ---- 新连接 ----
        // ET 模式下需要循环 accept 直到 EAGAIN
        while (true) {
          sockaddr_in client_addr{};
          socklen_t addr_len = sizeof(client_addr);
          int client_fd = accept4(listen_fd_,
                                   reinterpret_cast<sockaddr*>(&client_addr),
                                   &addr_len, SOCK_NONBLOCK);
          if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            LOG_WARN("accept failed: {}", std::strerror(errno));
            break;
          }

          ev.events = EPOLLIN | EPOLLET;
          ev.data.fd = client_fd;
          epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev);
        }
      } else {
        // ---- 客户端数据 ----
        handle_client(fd);
        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
        close(fd);
      }
    }
  }

  close(epoll_fd_);
  close(listen_fd_);
}

void HttpServer::stop() {
  running_ = false;
}

void HttpServer::handle_client(int client_fd) {
  // Phase 1 简化实现：一次读 + 一次写
  // 假设请求足够小，一次 recv 能读完
  // TODO(Phase2): 改为循环读取直到 EAGAIN，处理大请求和分片到达

  char buf[kBufSize];
  ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
  if (n <= 0) return;
  buf[n] = '\0';

  auto req = parse_request(buf, static_cast<size_t>(n));
  if (!req.valid) {
    auto resp = make_bad_request(R"({"error":"Invalid request"})");
    send(client_fd, resp.data(), resp.size(), MSG_NOSIGNAL);
    return;
  }

  // 仅允许 POST 方法（OpenAI 兼容 API）
  if (req.method != "POST") {
    auto resp = make_response(405, "application/json",
                              R"({"error":"Method not allowed"})");
    send(client_fd, resp.data(), resp.size(), MSG_NOSIGNAL);
    return;
  }

  // 使用 Router 查找处理器
  auto* handler = router_.find("POST", req.path);
  if (!handler) {
    auto resp = make_response(404, "application/json",
                              R"({"error":"Not found"})");
    send(client_fd, resp.data(), resp.size(), MSG_NOSIGNAL);
    return;
  }

  // 调用注册的 handler（由 main.cpp 注入转发逻辑）
  if (!handler_) {
    auto resp = make_service_unavailable(R"({"error":"No handler registered"})");
    send(client_fd, resp.data(), resp.size(), MSG_NOSIGNAL);
    return;
  }

  std::string request_body(req.body);
  std::string response_body = (*handler)(request_body);

  auto resp = make_ok_json(std::move(response_body));
  send(client_fd, resp.data(), resp.size(), MSG_NOSIGNAL);
}

}  // namespace ai_gateway
