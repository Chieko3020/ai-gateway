// HTTP 服务器核心实现
#include "server/http_server.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <charconv>
#include <cstring>
#include <string_view>

#include "common/logger.h"
#include "server/connection_handler.h"
#include "server/response.h"

namespace ai_gateway {

namespace {
constexpr int kMaxEvents = 64;
constexpr int kBacklog = 128;
constexpr size_t kReadChunk = 4096;    // 每次 recv 的读取块大小
constexpr size_t kMaxHeaderBytes = 65536;  // 头部最大 64KB，防止 slowloris
}  // namespace

HttpServer::HttpServer(const ServerConfig& config)
    : config_(config) {}

HttpServer::~HttpServer() {
  stop();
  // 关闭所有残留的连接缓冲
  for (auto& [fd, buf] : conn_buffers_) {
    (void)buf;
    close(fd);
  }
  conn_buffers_.clear();
  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }
  if (epoll_fd_ >= 0) {
    close(epoll_fd_);
    epoll_fd_ = -1;
  }
}

void HttpServer::set_handler(RequestHandler handler) {
  router_.add("POST", "/v1/chat/completions", std::move(handler));
}

void HttpServer::add_route(std::string_view path, RequestHandler handler) {
  router_.add("POST", path, std::move(handler));
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

void HttpServer::run(std::atomic<bool>* external_shutdown) {
  listen_fd_ = create_listen_socket();
  if (listen_fd_ < 0) return;

  epoll_fd_ = epoll_create1(0);
  if (epoll_fd_ < 0) {
    LOG_ERROR("epoll_create1 failed: {}", std::strerror(errno));
    close(listen_fd_);
    listen_fd_ = -1;
    return;
  }

  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;  // 边缘触发
  ev.data.fd = listen_fd_;
  epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev);

  running_ = true;
  epoll_event events[kMaxEvents];

  while (running_) {
    // 检查外部关闭信号（例如来自信号处理器）
    if (external_shutdown &&
        external_shutdown->load(std::memory_order_acquire)) {
      break;
    }

    // 短暂超时以便检查 running_ 和 external_shutdown
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
        // ET 模式下循环 accept，限流最多 16 个防主线程饥饿
        for (int accepted = 0; accepted < 16; ++accepted) {
          sockaddr_in client_addr{};
          socklen_t addr_len = sizeof(client_addr);
          int client_fd = accept4(listen_fd_,
                                   reinterpret_cast<sockaddr*>(&client_addr),
                                   &addr_len, SOCK_NONBLOCK);
          if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;  // 信号中断，继续尝试 accept
            LOG_WARN("accept failed: {}", std::strerror(errno));
            break;
          }

          ev.events = EPOLLIN | EPOLLET;
          ev.data.fd = client_fd;
          if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev) < 0) {
            LOG_WARN("epoll_ctl ADD failed: {}", std::strerror(errno));
            close(client_fd);
            continue;
          }
        }
      } else {
        // ---- 客户端数据：读入累积缓冲区，判断请求是否完整 ----
        handle_client(fd);
      }
    }
  }

  if (epoll_fd_ >= 0) {
    close(epoll_fd_);
    epoll_fd_ = -1;
  }
  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }
}

void HttpServer::stop() {
  running_ = false;
}

// 非阻塞循环 recv，把当前所有可读数据追加到缓冲区
// 返回 false 表示连接出错需关闭；true 表示正常（可能 EOF 或 EAGAIN）
bool HttpServer::read_into_buffer(int client_fd, std::string& buf) {
  char tmp[kReadChunk];
  while (true) {
    ssize_t n = recv(client_fd, tmp, sizeof(tmp), 0);
    if (n > 0) {
      buf.append(tmp, n);
      continue;
    }
    if (n == 0) {
      return true;  // 对端关闭写端，交由上层用已有缓冲区判断
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return true;  // 本轮可读数据读完
    }
    if (errno == EINTR) {
      continue;
    }
    return false;  // 真实错误
  }
}

// 从头部区段解析 Content-Length，失败返回 0
static size_t parse_content_length(std::string_view header_section) {
  // 大小写不敏感查找 "Content-Length:"
  size_t pos = 0;
  while (pos < header_section.size()) {
    auto line_end = header_section.find("\r\n", pos);
    if (line_end == std::string_view::npos) line_end = header_section.size();
    auto line = header_section.substr(pos, line_end - pos);
    auto colon = line.find(':');
    if (colon != std::string_view::npos) {
      auto key = line.substr(0, colon);
      // 简单大小写不敏感比较
      if (key.size() == 14) {  // "Content-Length" 长度
        bool match = true;
        for (size_t i = 0; i < 14; ++i) {
          char a = key[i];
          char b = "content-length"[i];
          if (a >= 'A' && a <= 'Z') a = a - 'A' + 'a';
          if (a != b) { match = false; break; }
        }
        if (match) {
          auto val_start = colon + 1;
          while (val_start < line.size() && line[val_start] == ' ') ++val_start;
          auto val = line.substr(val_start);
          size_t out = 0;
          auto [ptr, ec] = std::from_chars(val.data(), val.data() + val.size(), out);
          if (ec == std::errc() && ptr == val.data() + val.size()) return out;
          return 0;
        }
      }
    }
    if (line_end == header_section.size()) break;
    pos = line_end + 2;
  }
  return 0;
}

void HttpServer::handle_client(int client_fd) {
  std::string& buf = conn_buffers_[client_fd];
  if (!read_into_buffer(client_fd, buf)) {
    LOG_WARN("recv failed: {}", std::strerror(errno));
    conn_buffers_.erase(client_fd);
    close(client_fd);
    return;
  }
  if (buf.empty()) {
    conn_buffers_.erase(client_fd);
    close(client_fd);
    return;
  }

  // 1. 定位头部结束 \r\n\r\n
  auto header_end = buf.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    // 头部未收齐；超过头部上限则断开防 slowloris
    if (buf.size() > kMaxHeaderBytes) {
      LOG_WARN("header too large: {} bytes, closing", buf.size());
      conn_buffers_.erase(client_fd);
      close(client_fd);
      return;
    }
    // ET 模式下保持注册，等待下一批数据到达
    return;
  }

  // 2. 解析 Content-Length
  std::string_view header_section(buf.data(), header_end);
  size_t content_length = parse_content_length(header_section);
  if (content_length > config_.max_body_bytes) {
    LOG_WARN("body too large: {} > {} bytes, closing",
             content_length, config_.max_body_bytes);
    auto resp = make_payload_too_large(R"({"error":"Payload too large"})");
    const char* p = resp.data();
    size_t remaining = resp.size();
    while (remaining > 0) {
      ssize_t sent = send(client_fd, p, remaining, MSG_NOSIGNAL);
      if (sent <= 0) break;
      p += sent;
      remaining -= sent;
    }
    conn_buffers_.erase(client_fd);
    close(client_fd);
    return;
  }

  // 3. 判断 body 是否收齐
  size_t body_start = header_end + 4;
  if (buf.size() < body_start + content_length) {
    // body 未收齐，ET 模式下保持注册等待更多数据
    return;
  }

  // 4. 请求完整：从 epoll 移除，提交线程池处理
  epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr);
  std::string request = buf.substr(0, body_start + content_length);
  conn_buffers_.erase(client_fd);

  pool_.execute([this, client_fd, req = std::move(request)] {
    auto result = conn_handler_.process(req.data(), req.size());
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
