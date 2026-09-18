// HTTP 服务器核心实现
#include "server/http_server.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
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
constexpr int kMaxAcceptBatch = 512;   // 单次 epoll 事件内最多 accept 的连接数

}  // namespace

HttpServer::HttpServer(const ServerConfig& config)
    : config_(config) {}

HttpServer::~HttpServer() {
  stop();
  // 关闭所有残留的连接
  for (auto& [fd, conn] : conns_) {
    (void)conn;
    close(fd);
  }
  conns_.clear();
  conn_index_.clear();
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

void HttpServer::add_route(std::string_view method, std::string_view path,
                           RequestHandler handler) {
  router_.add(method, path, std::move(handler));
}

void HttpServer::set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return;
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    LOG_WARN("fcntl F_SETFL failed: {}", std::strerror(errno));
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

  // port=0 时由内核分配端口：回读实际端口供日志与测试使用
  sockaddr_in bound{};
  socklen_t bound_len = sizeof(bound);
  if (getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &bound_len) == 0)
    listen_port_ = ntohs(bound.sin_port);
  else
    listen_port_ = config_.port;

  LOG_INFO("listening on port {}", listen_port_);
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
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev) < 0) {
    LOG_ERROR("epoll_ctl ADD listen_fd failed: {}", std::strerror(errno));
    close(listen_fd_);
    listen_fd_ = -1;
    return;
  }

  running_ = true;
  epoll_event events[kMaxEvents];

  while (running_) {
    // 检查外部关闭信号（例如来自信号处理器）
    if (external_shutdown &&
        external_shutdown->load(std::memory_order_acquire)) {
      break;
    }

    // 短暂超时：既用于检查 running_/external_shutdown，也是空闲连接的扫描节拍
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
        // ET 模式下循环 accept 直到 EAGAIN：必须把 accept 队列排空，否则
        // max_connections 只会约束"已 accept 的表"，队列里排队的连接照样能完成
        // 三次握手并占用 backlog 槽位，上限形同虚设。单次上限仅用于防主线程饥饿。
        for (int accepted = 0; accepted < kMaxAcceptBatch; ++accepted) {
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

          // 连接数上限：直接拒绝而不是先收进来。没有这个上限时，fd 耗尽会让
          // accept 全面失败（含健康检查），整个服务不可用（报告 H5）。
          if (config_.max_connections > 0 &&
              conns_.size() >= config_.max_connections) {
            LOG_WARN("connection limit reached ({}), rejecting new connection",
                     config_.max_connections);
            // 注意：这里是 reactor 线程，绝不能用会阻塞的 send_all()——
            // 拒绝响应只有几十字节、socket 发送缓冲为空，正常不会 EAGAIN。
            // 大响应（唯一会真正撞上 EAGAIN 的场景）在 worker 里走 send_all()
            auto resp = make_service_unavailable(
                R"({"error":"Too many connections"})");
            const char* p = resp.data();
            size_t remaining = resp.size();
            while (remaining > 0) {
              ssize_t sent = send(client_fd, p, remaining, MSG_NOSIGNAL);
              if (sent <= 0) break;
              p += sent;
              remaining -= static_cast<size_t>(sent);
            }
            close(client_fd);
            continue;
          }

          ev.events = EPOLLIN | EPOLLET;
          ev.data.fd = client_fd;
          if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev) < 0) {
            LOG_WARN("epoll_ctl ADD failed: {}", std::strerror(errno));
            close(client_fd);
            continue;
          }
          auto it = conns_.emplace(conns_.end(), client_fd, Connection{});
          it->second.last_activity = std::chrono::steady_clock::now();
          conn_index_[client_fd] = it;
        }
      } else {
        // ---- 客户端数据：读入累积缓冲区，判断请求是否完整 ----
        handle_client(fd);
      }
    }

    // 每个 epoll 节拍做一次连接维护：
    //   - 主动读一遍所有连接（ET 模式下"对端只发 FIN、不再发数据"不一定产生新的
    //     EPOLLIN 边沿，实测在低延迟环回上会漏；主动读才能保证半关闭立刻被发现）
    //   - 关闭空闲超时的连接（slowloris / 半关闭驻留）
    maintain_connections();
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

void HttpServer::drain() {
  // run() 已退出：不再有新连接与新任务提交（worker 之间也不会再 execute），
  // 这里只需等在途请求跑完，之后的统计/落盘才不会与它们并发
  pool_.wait_idle();
}

// 自由函数（不是 HttpServer 的私有方法）：这样可以脱离 HTTP 服务直接用
// socketpair 做确定性单测（见 test_http_server 第 6 段），不必依赖 TCP 时序
bool send_all_with_deadline(int client_fd, const char* data, size_t len,
                            int write_deadline_ms,
                            std::atomic<uint64_t>* eagain_count) {
  // 旧实现在这里 `if (sent <= 0) break`：非阻塞 fd 上 send() 返回 EAGAIN 时
  // 会把剩余字节直接丢掉，慢客户端读大响应只会拿到前半截（报告 8.7 第 3 条）。
  //
  // 这里改成"等到能写为止"：EAGAIN 时用 poll(POLLOUT) 阻塞等待可写再续发，
  // 并用**总 deadline**（不是每次 poll 各自计时）兜住慢客户端，
  // 避免一个连接把 worker 永久占住。
  //
  // 为什么不做 EPOLLOUT 写缓冲：那需要把未发完的数据从 worker 交还给 reactor
  // （跨线程写队列 + eventfd 唤醒 + 重新注册 EPOLLOUT + 所有权转移），
  // 在本项目"worker 独占 fd 并在结束时 close"的架构下改动面大、竞态风险高；
  // poll 方案只影响单个 worker 且行为可测（见 test_http_server 的慢客户端用例）。
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(write_deadline_ms);
  size_t offset = 0;
  while (offset < len) {
    ssize_t sent = send(client_fd, data + offset, len - offset, MSG_NOSIGNAL);
    if (sent > 0) {
      offset += static_cast<size_t>(sent);
      continue;
    }
    if (sent < 0 && errno == EINTR) continue;
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      // 计数供回归测试断言"确实走到了等待可写的分支"（否则用例可能没触发 EAGAIN）
      if (eagain_count)
        eagain_count->fetch_add(1, std::memory_order_relaxed);
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now())
                            .count();
      if (left <= 0) {
        LOG_WARN("send: write deadline ({}ms) exceeded, {} of {} bytes unsent",
                 write_deadline_ms, len - offset, len);
        return false;
      }
      pollfd w{client_fd, POLLOUT, 0};
      int rc = ::poll(&w, 1, static_cast<int>(left));
      if (rc < 0 && errno != EINTR) {
        LOG_WARN("send: poll failed: {}", std::strerror(errno));
        return false;
      }
      if (rc == 0) {
        LOG_WARN("send: write deadline ({}ms) exceeded, {} of {} bytes unsent",
                 write_deadline_ms, len - offset, len);
        return false;
      }
      continue;  // 可写了，续发
    }
    // 真实错误：EPIPE / ECONNRESET 等（客户端已经走了，没什么可补救的）
    if (sent < 0)
      LOG_DEBUG("send error: {} ({} of {} bytes sent)", std::strerror(errno),
                offset, len);
    return false;
  }
  return true;
}

bool HttpServer::send_all(int client_fd, const char* data, size_t len,
                          int write_deadline_ms) {
  return send_all_with_deadline(client_fd, data, len, write_deadline_ms,
                                &send_eagain_count_);
}

void HttpServer::close_connection(int client_fd) {
  if (epoll_fd_ >= 0)
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr);
  auto it = conn_index_.find(client_fd);
  if (it != conn_index_.end()) {
    conns_.erase(it->second);
    conn_index_.erase(it);
  }
  close(client_fd);
}

void HttpServer::maintain_connections() {
  // 收集本轮要处理的 fd：处理过程中会 erase 连接，不能直接边遍历边改
  std::vector<int> fds;
  fds.reserve(conns_.size());
  for (auto& [fd, conn] : conns_) fds.push_back(fd);

  const bool has_timeout = config_.idle_timeout_seconds > 0;
  const auto now = std::chrono::steady_clock::now();
  const auto limit = std::chrono::seconds(config_.idle_timeout_seconds);

  size_t closed_idle = 0;
  for (int fd : fds) {
    auto it = conn_index_.find(fd);
    if (it == conn_index_.end()) continue;  // 同一轮里已被处理掉

    // 主动读一次：把因 ET 边沿丢失而滞留在内核缓冲区（含 FIN）的数据取出来。
    // 这里不刷新 last_activity —— 空闲超时必须按"客户端最后一次发字节"计时，
    // 否则维护本身会把连接续命。
    ReadState state = read_into_buffer(fd, it->second->second.buf);
    if (state == ReadState::kError) {
      LOG_WARN("recv failed: {}", std::strerror(errno));
      close_connection(fd);
      continue;
    }
    if (state == ReadState::kData) {
      it->second->second.last_activity = now;
      if (handle_buffer(fd, /*peer_closed=*/false)) continue;
    } else if (state == ReadState::kEof) {
      if (handle_buffer(fd, /*peer_closed=*/true)) continue;
    }

    // 仍不完整的连接：检查空闲超时
    it = conn_index_.find(fd);
    if (it == conn_index_.end()) continue;
    if (has_timeout && now - it->second->second.last_activity >= limit) {
      LOG_WARN("idle timeout ({}s): closing connection, buffered {} bytes",
               config_.idle_timeout_seconds, it->second->second.buf.size());
      close_connection(fd);
      ++closed_idle;
    }
  }
  if (closed_idle > 0)
    LOG_DEBUG("idle timeout: closed {} connection(s)", closed_idle);
}

// 非阻塞循环 recv，把当前所有可读数据追加到缓冲区。
// 三态返回：kData/kEof 表示本次读到过东西（含对端关闭），kAgain 表示当前无数据，
// kError 表示真实错误。旧实现只返回 bool，把"对端 FIN"与"暂时无数据"都当成 true，
// 于是半关闭连接会被永久留在 epoll 里等一个永不到来的可读事件（报告 H5）。
HttpServer::ReadState HttpServer::read_into_buffer(int client_fd,
                                                   std::string& buf) {
  char tmp[kReadChunk];
  bool got_data = false;
  while (true) {
    ssize_t n = recv(client_fd, tmp, sizeof(tmp), 0);
    if (n > 0) {
      buf.append(tmp, static_cast<size_t>(n));
      got_data = true;
      continue;
    }
    if (n == 0) {
      // 对端关闭写端（FIN）：缓冲区里已是全部可用数据，交由上层判断是否完整
      return got_data ? ReadState::kData : ReadState::kEof;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return got_data ? ReadState::kData : ReadState::kAgain;
    }
    if (errno == EINTR) {
      continue;
    }
    return ReadState::kError;
  }
}

// 从头部区段解析 Content-Length，失败返回 0
static size_t parse_content_length_caseless(std::string_view header_section) {
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
  auto idx_it = conn_index_.find(client_fd);
  if (idx_it == conn_index_.end()) {
    // 已被 worker 关闭 / 已被空闲超时清理：fd 号里已无我们的状态
    return;
  }
  auto& conn = idx_it->second->second;

  ReadState state = read_into_buffer(client_fd, conn.buf);
  if (state == ReadState::kError) {
    LOG_WARN("recv failed: {}", std::strerror(errno));
    close_connection(client_fd);
    return;
  }
  // kAgain 表示"当前无数据"——不刷新活动时间，让空闲超时能真正生效；
  // kData/kEof 都表示本轮读到了字节（或对端已关闭），刷新活动时间
  if (state == ReadState::kData)
    conn.last_activity = std::chrono::steady_clock::now();

  if (conn.buf.empty()) {
    // 无数据：只有对端已关闭（FIN）时才立即回收，EAGAIN 时保留连接等下一批数据
    if (state == ReadState::kEof) close_connection(client_fd);
    return;
  }

  handle_buffer(client_fd, state == ReadState::kEof);
}

// 用累积缓冲区判断请求是否完整：
//   完整 -> 提交线程池并注销连接（返回 true）
//   不完整且对端已 FIN -> 立即关闭并回收 fd（返回 true，报告 H5 的核心修复）
//   不完整且对端仍开着 -> 保留（返回 false，等后续数据或空闲超时）
bool HttpServer::handle_buffer(int client_fd, bool peer_closed) {
  auto idx_it = conn_index_.find(client_fd);
  if (idx_it == conn_index_.end()) return true;
  std::string& buf = idx_it->second->second.buf;

  // 1. 定位头部结束 \r\n\r\n
  auto header_end = buf.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    // 头部未收齐：超过头部上限断开防 slowloris；对端已 FIN 也立即回收
    if (buf.size() > config_.max_header_bytes || peer_closed) {
      LOG_WARN("incomplete header ({} bytes, eof={}), closing", buf.size(),
               peer_closed);
      close_connection(client_fd);
      return true;
    }
    return false;
  }

  // 2. 解析 Content-Length
  std::string_view header_section(buf.data(), header_end);
  size_t content_length = parse_content_length_caseless(header_section);
  if (content_length > config_.max_body_bytes) {
    LOG_WARN("body too large: {} > {} bytes, closing",
             content_length, config_.max_body_bytes);
    // 同样在 reactor 线程内，响应只有几十字节（见上面 503 分支的说明）
    auto resp = make_payload_too_large(R"({"error":"Payload too large"})");
    const char* p = resp.data();
    size_t remaining = resp.size();
    while (remaining > 0) {
      ssize_t sent = send(client_fd, p, remaining, MSG_NOSIGNAL);
      if (sent <= 0) {
        if (sent < 0) LOG_DEBUG("send error: {}", std::strerror(errno));
        break;
      }
      p += sent;
      remaining -= static_cast<size_t>(sent);
    }
    close_connection(client_fd);
    return true;
  }

  // 3. 判断 body 是否收齐
  size_t body_start = header_end + 4;
  if (buf.size() < body_start + content_length) {
    // body 未收齐：对端已经不会再发数据（FIN）→ 立刻回收 fd 与缓冲区；
    // 这正是报告 H5 的泄漏场景（旧实现在这里直接 return，连接永久驻留）
    if (peer_closed) {
      LOG_WARN("client half-closed with incomplete body ({} < {}), closing",
               buf.size() - body_start, content_length);
      close_connection(client_fd);
      return true;
    }
    return false;
  }

  // 4. 请求完整：从 epoll 移除，提交线程池处理
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr) < 0)
    LOG_DEBUG("epoll_ctl DEL failed: {}", std::strerror(errno));
  std::string request = buf.substr(0, body_start + content_length);
  conns_.erase(idx_it->second);
  conn_index_.erase(client_fd);

  pool_.execute([this, client_fd, req = std::move(request)] {
    try {
      auto result = conn_handler_.process(req.data(), req.size());
      // 大响应 + 慢客户端：必须写到底（或到写超时），不能因为 EAGAIN 就截断
      send_all(client_fd, result.response.data(), result.response.size(),
               config_.write_timeout_seconds * 1000);
    } catch (const std::exception& e) {
      // 任何异常都不能穿越线程池边界（会 std::terminate 整个进程）
      LOG_ERROR("worker task threw: {}", e.what());
    } catch (...) {
      LOG_ERROR("worker task threw non-std exception");
    }
    close(client_fd);
  });
  return true;
}

}  // namespace ai_gateway
