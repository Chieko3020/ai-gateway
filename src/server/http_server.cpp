// HTTP 服务器核心实现
#include "server/http_server.h"

#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
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
constexpr size_t kReadChunk = 4096;   // 每次 recv 的读取块大小
constexpr int kMaxAcceptBatch = 512;  // 单次 epoll 事件内最多 accept 的连接数
}  // namespace

// ===========================================================================
// 写路径：非阻塞 fd 上"写到底"
// ===========================================================================
WriteStatus send_all_with_deadline_until(
    int client_fd, const char* data, size_t len, int write_deadline_ms,
    std::atomic<uint64_t>* eagain_count,
    std::chrono::steady_clock::time_point total_deadline,
    std::chrono::steady_clock::time_point idle_deadline) {
  // 旧实现在这里 `if (sent <= 0) break`：非阻塞 fd 上 send() 返回 EAGAIN 时
  // 会把剩余字节直接丢掉，慢客户端读大响应只会拿到前半截（报告 8.7 第 3 条）。
  //
  // 这里改成"等到能写为止"：EAGAIN 时用 poll(POLLOUT) 阻塞等待可写再续发，
  // 避免一个连接把 worker 永久占住靠的是两条死线：
  //   total_deadline —— **整条响应**的绝对上限（缓冲式响应用它）
  //   idle_deadline  —— 单次停顿的上限（流式响应用它）。
  //     语义差别是本质的：total 把"回答有多长"也框住了，idle 只框"客户端是否还在读"。
  //     流式长回答必须用后者，否则输出超过 total 的流会被拦腰切断
  //
  // 为什么不做 EPOLLOUT 写缓冲：那需要把未发完的数据从 worker 交还给 reactor
  // （跨线程写队列 + eventfd 唤醒 + 重新注册 EPOLLOUT + 所有权转移），
  // 在本项目"worker 独占 fd 直到写完"的架构下改动面大、竞态风险高；
  // poll 方案只影响单个 worker 且行为可测（见 test_http_server 的慢客户端用例）。
  const auto now = std::chrono::steady_clock::now();
  auto call_deadline =
      now + std::chrono::milliseconds(write_deadline_ms > 0 ? write_deadline_ms : 0);
  // 单次写调用的上限 = min(单次写死线, 整条响应的总死线)
  const auto deadline = std::min(call_deadline, total_deadline);
  // 一次停顿的上限再取一次 min（idle 通常就是总死线本身；流式下 total 是 max）
  const auto pause_deadline = std::min(deadline, idle_deadline);

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
                            pause_deadline - std::chrono::steady_clock::now())
                            .count();
      if (left <= 0) {
        LOG_WARN("send: write deadline exceeded, {} of {} bytes unsent",
                 len - offset, len);
        return WriteStatus::kDeadline;
      }
      pollfd w{client_fd, POLLOUT, 0};
      int rc = ::poll(&w, 1, static_cast<int>(left));
      if (rc < 0 && errno != EINTR) {
        LOG_WARN("send: poll failed: {}", std::strerror(errno));
        return WriteStatus::kSocket;
      }
      if (rc == 0) {
        LOG_WARN("send: write deadline exceeded, {} of {} bytes unsent",
                 len - offset, len);
        return WriteStatus::kDeadline;
      }
      continue;  // 可写了，续发
    }
    // 真实错误：EPIPE / ECONNRESET 等（客户端已经走了，没什么可补救的）
    if (sent < 0)
      LOG_DEBUG("send error: {} ({} of {} bytes sent)", std::strerror(errno),
                offset, len);
    return WriteStatus::kSocket;
  }
  return WriteStatus::kOk;
}

// ===========================================================================
// ResponseWriter
// ===========================================================================
WriteStatus send_all_with_deadline(
    int client_fd, const char* data, size_t len, int write_deadline_ms,
    std::atomic<uint64_t>* eagain_count,
    std::chrono::steady_clock::time_point total_deadline) {
  return send_all_with_deadline_until(client_fd, data, len, write_deadline_ms,
                                      eagain_count, total_deadline,
                                      total_deadline);
}

std::chrono::steady_clock::time_point ResponseWriter::current_total_deadline()
    const {
  // 流式响应**没有**总死线：时长由上游回答长度决定（输出 4K token 就是几十秒到
  // 几分钟），给它一个总时限等于"回答超过 X 秒就被切断"——正是本轮要修的缺陷。
  // 防慢客户端改由空闲死线负责（见 current_idle_deadline）
  return chunked_ ? std::chrono::steady_clock::time_point::max()
                  : total_deadline_;
}

std::chrono::steady_clock::time_point ResponseWriter::current_idle_deadline()
    const {
  return chunked_ ? stream_idle_deadline_ : total_deadline_;
}

bool ResponseWriter::send_raw(std::string_view data) {
  if (failed_) return false;
  if (data.empty()) return true;

  // 小数据直接写：HTTP 头与 SSE 事件通常几十到几百字节，避免为它们多走一次
  // 缓冲拷贝。写不完的部分再进缓冲（后续 flush 续发）。
  WriteStatus st = WriteStatus::kOk;
  if (out_.empty()) {
    st = send_all_with_deadline_until(
        fd_, data.data(), data.size(), write_deadline_ms_, eagain_count_,
        current_total_deadline(), current_idle_deadline());
    if (st == WriteStatus::kOk) {
      bytes_sent_ += data.size();
      // 流式：写出去了就说明客户端在跟读，把空闲死线向前推（这就是"续期"）
      if (chunked_)
        stream_idle_deadline_ = std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(stream_idle_ms_);
      return true;
    }
    failed_ = true;
    // 总死线与空闲死线要分开报：前者是"这条响应整体写太久了"（缓冲式才可能），
    // 后者是"客户端在流中途不读了"。排查时两者的处置完全不同
    abort_ = (st == WriteStatus::kDeadline)
                 ? (chunked_ ? AbortReason::kIdle : AbortReason::kDeadline)
                 : AbortReason::kClientGone;
    return false;
  }

  // 已有积压：先追加再整体 flush（保持字节顺序）
  out_.append(data);
  return flush();
}

bool ResponseWriter::flush() {
  if (failed_) return false;
  if (out_.empty()) return true;
  const size_t n = out_.size();
  WriteStatus st = send_all_with_deadline_until(
      fd_, out_.data(), n, write_deadline_ms_, eagain_count_,
      current_total_deadline(), current_idle_deadline());
  if (st == WriteStatus::kOk) {
    bytes_sent_ += n;
    out_.clear();
    if (chunked_)
      stream_idle_deadline_ = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(stream_idle_ms_);
    return true;
  }
  failed_ = true;
  abort_ = (st == WriteStatus::kDeadline)
               ? (chunked_ ? AbortReason::kIdle : AbortReason::kDeadline)
               : AbortReason::kClientGone;
  return false;
}

bool ResponseWriter::write_head(int status_code, std::string_view content_type,
                                size_t content_length, bool keep_alive) {
  if (committed_ || failed_) return false;
  committed_ = true;
  chunked_ = false;
  ResponseHeader h;
  h.status_code = status_code;
  h.content_type = std::string(content_type);
  h.content_length = content_length;
  h.keep_alive = keep_alive;
  h.chunked = false;
  return send_raw(build_response_head(h));
}

bool ResponseWriter::write_stream_head(int status_code,
                                       std::string_view content_type,
                                       bool keep_alive) {
  if (committed_ || failed_) return false;
  committed_ = true;
  chunked_ = true;
  ResponseHeader h;
  h.status_code = status_code;
  h.content_type = std::string(content_type);
  h.content_length = kChunkedLength;
  h.chunked = true;
  h.keep_alive = keep_alive;
  // 起点：空闲死线从"响应头发出去"这一刻开始算。之后每次 write_body 成功都会
  // 把它推到 now + 空闲值。上游迟迟不吐第一个 token（"首包很慢"）这段时间
  // 也受同一个空闲值约束——否则一个连上就不发数据的上游能把 worker 挂死
  stream_idle_deadline_ =
      std::chrono::steady_clock::now() +
      std::chrono::milliseconds(stream_idle_ms_);
  // 流式响应头必须**立刻**发出去：客户端（curl -N / SSE 解析器）要看到 200 +
  // text/event-stream 才开始处理后续事件。缓冲到第一次 body 再发会让"上游迟迟
  // 不吐第一个 token"的场景表现为网关不响应
  return send_raw(build_response_head(h));
}

bool ResponseWriter::write_body(std::string_view data) {
  if (!committed_ || failed_) return false;
  if (data.empty()) return true;
  if (chunked_) {
    // chunked 分帧：每个 chunk 自带长度。0 长度块专用作终止（见 finish_stream），
    // 因此空块在这里被跳过
    std::string framed = encode_chunk(data);
    return send_raw(framed);
  }
  return send_raw(data);
}

bool ResponseWriter::finish_stream() {
  if (!committed_ || failed_ || !chunked_) return !failed_;
  return send_raw(kChunkedTerminator);
}

// ===========================================================================
// HttpServer
// ===========================================================================
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
  {
    std::lock_guard lock(io_mutex_);
    for (auto& r : io_returns_) close(r.fd);
    io_returns_.clear();
  }
  if (wake_fd_ >= 0) {
    close(wake_fd_);
    wake_fd_ = -1;
  }
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

  // 唤醒 fd：worker 写完一个 keep-alive 连接后要把它交还 reactor 重新登记，
  // 不能等最多 100ms 的 epoll 超时（那会给下一个请求平白加上几十毫秒）。
  // eventfd 的 8 字节计数只当信号用，不承载数据（fd 本身走 io_returns_ 队列）
  wake_fd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (wake_fd_ < 0) {
    LOG_WARN("eventfd() failed: {} (keep-alive 归还最迟 100ms 生效)",
             std::strerror(errno));
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
  if (wake_fd_ >= 0) {
    ev.events = EPOLLIN | EPOLLET;
    ev.data.fd = wake_fd_;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, wake_fd_, &ev) < 0)
      LOG_WARN("epoll_ctl ADD wake_fd failed: {}", std::strerror(errno));
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

          accepted_connections_.fetch_add(1, std::memory_order_relaxed);
          ev.events = EPOLLIN | EPOLLET;
          ev.data.fd = client_fd;
          if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &ev) < 0) {
            LOG_WARN("epoll_ctl ADD failed: {}", std::strerror(errno));
            close(client_fd);
            continue;
          }
          auto it = conns_.emplace(conns_.end(), client_fd,
                                   Connection{std::chrono::steady_clock::now()});
          conn_index_[client_fd] = it;
        }
      } else if (fd == wake_fd_) {
        // ---- worker 归还的 fd ----
        uint64_t v = 0;
        while (read(wake_fd_, &v, sizeof(v)) > 0) {}  // 清空计数（ET 模式）
        process_returned_fds(/*stop_requested=*/false);
      } else {
        // ---- 客户端数据：读入累积缓冲区，判断请求是否完整 ----
        handle_client(fd);
      }
    }

    // 每个 epoll 节拍做一次连接维护：
    //   - 主动读一遍所有连接（ET 模式下"对端只发 FIN、不再发数据"不一定产生新的
    //     EPOLLIN 边沿，实测在低延迟环回上会漏；主动读才能保证半关闭立刻被发现）
    //   - 关闭空闲超时的连接（slowloris / 半关闭驻留 / keep-alive 空闲连接）
    //   - 处理上一轮里 worker 归还的 fd（不依赖 eventfd，做兜底）
    maintain_connections();
  }

  // 停机：把所有还借在 worker 手里的连接按 close 处理（它们在 drain() 之后
  // 才归还，这里先不处理；drain() 里会再收一次尾）
  process_returned_fds(/*stop_requested=*/true);

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
  // worker 一律不自己 close(fd)，因此这里必须把归还队列收尾——
  // 否则停机时这些 fd 既不在 conns_（不归 reactor 管）也没被 close
  process_returned_fds(/*stop_requested=*/true);
}

// ---------------------------------------------------------------------------
// fd 所有权：worker -> reactor 的归还路径
// ---------------------------------------------------------------------------
void HttpServer::grant_borrow(uint64_t token) {
  std::lock_guard lock(io_mutex_);
  borrowed_tokens_.insert(token);
}

bool HttpServer::try_claim_borrow(uint64_t token) {
  std::lock_guard lock(io_mutex_);
  return borrowed_tokens_.erase(token) > 0;
}

void HttpServer::return_fd(int fd, FdAction action, uint64_t token) {
  {
    std::lock_guard lock(io_mutex_);
    // 令牌可能已被 reactor 单方面撤销（try_claim_borrow 失败时不会归还），
    // 这里统一清掉，避免集合无限增长
    borrowed_tokens_.erase(token);
    io_returns_.push_back(FdReturn{fd, action, token});
  }
  if (wake_fd_ >= 0) {
    uint64_t one = 1;
    ssize_t n = write(wake_fd_, &one, sizeof(one));
    (void)n;  // 失败也没关系：maintain_connections() 每 100ms 兜底处理一次
  }
}

void HttpServer::process_returned_fds(bool stop_requested) {
  std::vector<FdReturn> batch;
  {
    std::lock_guard lock(io_mutex_);
    batch.swap(io_returns_);
  }
  for (const auto& r : batch) {
    // 连接项仍然由 reactor 持有（worker 只是"借用"），因此这里统一回收
    if (stop_requested || r.action == FdAction::kClose) {
      close_connection(r.fd);
      continue;
    }
    rearm_keep_alive(r.fd, r.token);
  }
}

void HttpServer::rearm_keep_alive(int fd, uint64_t token) {
  auto it = conn_index_.find(fd);
  if (it == conn_index_.end()) {
    // 连接项已经没了：若 fd 仍然有效，说明它已经不属于我们（正常路径上
    // 不会发生——reactor 只在归还流程里删连接项），保守关闭避免 fd 泄漏
    LOG_DEBUG("keep-alive: fd {} has no registered connection, closing", fd);
    ::close(fd);
    return;
  }
  auto& conn = it->second->second;
  if (conn.borrow_token != token) {
    // fd 号已被复用（旧连接在借用期间被关闭，新连接拿到了同一个号码）。
    // 这时既不能 close 也不能重新登记：那个 fd 现在是别人的连接
    LOG_DEBUG("keep-alive: stale return for fd {} (token {} != {}), ignoring", fd,
              token, conn.borrow_token);
    return;
  }

  // 同一个连接上的第 2 个及以后的请求：计数供测试证明复用真的发生了
  // （而不是"客户端碰巧连了两次"，那个场景下每条连接的 requests_served 都是 1）
  if (++conn.requests_served > 1)
    reused_connections_.fetch_add(1, std::memory_order_relaxed);
  keep_alive_rearms_.fetch_add(1, std::memory_order_relaxed);

  // worker 借用期间 reactor 不读该 fd，因此缓冲区里只可能有上一个请求之后
  // 多收到的字节（pipelining）：留给下一轮解析，别丢
  conn.worker_owned.store(false);
  conn.last_activity = std::chrono::steady_clock::now();

  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ev.data.fd = fd;
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, fd, &ev) < 0) {
    close_connection(fd);
    return;
  }

  // 缓冲里已经有下一个请求（客户端 pipelining）时立即推进状态机，
  // 不必等下一次可读事件——ET 模式下那些字节不会再有新的边沿
  if (!conn.buf.empty()) handle_buffer(fd, /*peer_closed=*/false);
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
  // 先处理 worker 归还的 fd（不依赖 eventfd 的兜底路径）
  process_returned_fds(/*stop_requested=*/false);

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

    // 借给 worker 的连接：所有权不在 reactor 手里，绝不能读或关它
    // （读会与 worker 的响应写/客户端并发行为抢数据，关会让 worker 的
    //   send 落到一个已被复用出去的 fd 上）
    if (it->second->second.worker_owned.load())
      continue;

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
    if (it->second->second.worker_owned.load())
      continue;
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
    // 已被空闲超时清理 / 已被归还流程处理：fd 号里已无我们的状态
    return;
  }
  // 借给 worker 的连接：worker 正在写响应，reactor 不能读它的 socket
  if (idx_it->second->second.worker_owned.load())
    return;
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
//   完整 → 标记 worker_owned 后提交线程池（返回 true）
//   不完整且对端已 FIN → 立即关闭并回收 fd（返回 true，报告 H5 的核心修复）
//   不完整且对端仍开着 → 保留（返回 false，等后续数据或空闲超时）
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

  // 4. 请求完整：从 epoll 移除并**借出**给线程池
  //
  // 这里不 erase 连接项：fd 的数量上限（max_connections）必须把"正在处理的
  // 请求"也算进去，否则并发在上限之上时 fd 会被打穿。改为打 worker_owned 标记，
  // reactor 在 handle_client/maintain_connections 里跳过它。
  if (epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr) < 0)
    LOG_DEBUG("epoll_ctl DEL failed: {}", std::strerror(errno));

  std::string request = buf.substr(0, body_start + content_length);
  // 请求之后可能已经跟着下一个请求（pipelining）：把已消费的部分从缓冲区摘掉，
  // 剩余字节留给归还后重新登记时继续解析
  buf.erase(0, body_start + content_length);
  idx_it->second->second.worker_owned.store(true);
  const uint64_t borrow_token = ++idx_it->second->second.borrow_token;
  grant_borrow(borrow_token);

  pool_.execute([this, client_fd, borrow_token, req = std::move(request)] {
    // 借用认领：这个 fd 号在"提交任务"到"worker 真正开始跑"之间可能已经被
    // reactor 关闭并复用（空闲超时 / 半关闭），此时绝不能再碰它
    if (!try_claim_borrow(borrow_token)) {
      LOG_DEBUG("worker: fd {} borrow revoked before start, dropping task",
                client_fd);
      // 这次借出已被撤销：fd 要么已经被 reactor 关闭、要么已经是别人的连接，
      // 两种情况下都不能 close，直接放弃任务
      return;
    }

    bool keep_alive = false;
    try {
      // 写死线的两种口径（见 http_server.h 的 ResponseWriter 说明）：
      //   缓冲式：write_timeout_seconds 是**整条响应**的总死线
      //   流式  ：不设总死线（时长由回答长度决定），改用"两次写入之间的空闲超时"
      // 是否流式由连接处理器扫请求体得到的 may_stream 预判（不解析 JSON）；
      // 真正的流式判定仍在 handle_request 里由 wants_stream() 做
      const bool may_stream = req.find("\"stream\"") != std::string::npos;
      const auto total_deadline =
          may_stream ? std::chrono::steady_clock::time_point::max()
                     : std::chrono::steady_clock::now() +
                           std::chrono::seconds(config_.write_timeout_seconds);
      ResponseWriter writer(client_fd, config_.write_timeout_seconds * 1000,
                            total_deadline, &send_eagain_count_,
                            config_.stream_idle_timeout_seconds * 1000);
      auto result = conn_handler_.process(req.data(), req.size(), writer);
      if (!result.response.empty()) {
        // 缓冲式响应：一次性写到底（大响应 + 慢客户端走 send_all 的等待路径）
        WriteStatus st = send_all(client_fd, result.response.data(),
                                  result.response.size(),
                                  config_.write_timeout_seconds * 1000,
                                  total_deadline);
        keep_alive = result.keep_alive && st == WriteStatus::kOk;
        if (st != WriteStatus::kOk)
          LOG_DEBUG("response write aborted ({} bytes)", result.response.size());
      } else {
        // 流式响应：handler 已自行写完（含 chunked 终止块）
        keep_alive = result.keep_alive && !writer.failed();
      }
    } catch (const std::exception& e) {
      // 任何异常都不能穿越线程池边界（会 std::terminate 整个进程）
      LOG_ERROR("worker task threw: {}", e.what());
      keep_alive = false;
    } catch (...) {
      LOG_ERROR("worker task threw non-std exception");
      keep_alive = false;
    }
    // 所有权归还：worker 绝不自己 close(fd)（见 http_server.h 顶部说明）。
    // 归还时带上借用令牌，reactor 据此丢弃"fd 已被复用"的过期归还
    return_fd(client_fd, keep_alive ? FdAction::kKeepAlive : FdAction::kClose,
              borrow_token);
  });
  return true;
}

WriteStatus HttpServer::send_all(
    int client_fd, const char* data, size_t len, int write_deadline_ms,
    std::chrono::steady_clock::time_point total_deadline) {
  return send_all_with_deadline(client_fd, data, len, write_deadline_ms,
                                &send_eagain_count_, total_deadline);
}

}  // namespace ai_gateway
