// HTTP 服务器核心实现
#include "http_server.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>

#include "logger.h"
#include "connection_handler.h"

namespace ai_gateway {

namespace {
constexpr int kMaxEvents = 64;
constexpr int kBacklog = 128;
constexpr size_t kBufSize = 65536;  // 64KB 缓冲区，足够大多数 HTTP 请求
}  // namespace

HttpServer::HttpServer(const ServerConfig& config)
    : config_(config) {}

HttpServer::~HttpServer() {
  stop();
  if (listen_fd_ >= 0) close(listen_fd_);
  if (epoll_fd_ >= 0) close(epoll_fd_);
}

void HttpServer::set_handler(RequestHandler handler) {
  router_.add("/v1/chat/completions", std::move(handler));
}

void HttpServer::add_route(std::string_view path, RequestHandler handler) {
  router_.add(path, std::move(handler));
}

void HttpServer::set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return;
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int HttpServer::create_listen_socket() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    LOG_ERROR("socket() failed: {}", std::strerror(errno));
    return -1;
  }

  // SO_REUSEADDR
  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    LOG_WARN("setsockopt SO_REUSEADDR failed: {}", std::strerror(errno));
  }

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
          if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev) < 0) {
            close(client_fd);
            continue;
          }
        }
      } else {
        // ---- 客户端数据提交到线程池处理 ----
        handle_client(fd);
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
  char buf[kBufSize];
  ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
  buf[n] = '\0';

  // Non-blocking socket may require multiple recv() calls to assemble
  // a complete HTTP request (TCP segmentation). Do one more recv with
  // a short timeout to collect trailing data.
  {
    struct timeval tv = {0, 100000};  // 100ms
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ssize_t n2 = recv(client_fd, buf + n, sizeof(buf) - 1 - n, 0);
    if (n2 > 0) { n += n2; buf[n] = '\0'; }
  }
  std::string request(buf, n);
  pool_.execute([this, client_fd, req = std::move(request), n] {
    auto result = conn_handler_.process(req.data(), n);
    const char* p = result.response.data();
    size_t remaining = result.response.size();
    while (remaining > 0) {
      ssize_t sent = send(client_fd, p, remaining, MSG_NOSIGNAL);
      if (sent <= 0) break;
      p += sent;
      remaining -= sent;
    }
    close(client_fd);
  });
}

}  // namespace ai_gateway
