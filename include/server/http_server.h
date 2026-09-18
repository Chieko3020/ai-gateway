// HTTP 服务器核心：epoll ET + 非阻塞 IO + 线程池
// 主线程 accept + epoll_wait 提交到线程池 worker 处理 + 写 + 归还/close
//     epoll_wait (主线程)
//       ├─ listen_fd EPOLLIN accept 后 EPOLL_CTL_ADD 新 client_fd
//       └─ client_fd EPOLLIN 然后 EPOLL_CTL_DEL 然后 handle_client
//                                                      ↓
//           handle_client: 非阻塞循环 recv 追加到累积缓冲区 conns_[fd].buf
//                          解析 Content-Length 判断 body 是否完整
//                          完整 → 连接留在 conns_ 但标记 worker_owned，
//                                  pool_.execute 里 process + 写 + 归还/close
//                          不完整 → 保持注册等待更多数据（受空闲超时约束）
//
// 连接生命周期（报告 H5 + 本轮 keep-alive）：
//   - recv 三态：收到数据 / 对端 EOF / 错误或"当前无数据"（EAGAIN）
//     半关闭（收到 FIN）且请求不完整 → 立即关闭，不再等一个永远不来的可读事件
//   - 每连接空闲超时 idle_timeout_seconds：epoll_wait 的 100ms 节拍上扫描，
//     超时立即关闭。keep-alive 复用后同一个上限同时承担"空闲长连接回收"
//   - max_connections 上限：超出时对新连接直接回 503 并关闭
//   - **fd 的所有权只有一个**：连接要么在 reactor（conns_）手里，要么借给某个
//     worker。worker 处理期间连接项留在 conns_ 里但带 worker_owned 标记
//     （连接上限仍然把它算进去），reactor 跳过它、不读不关。
//     worker 结束后一律**不自己 close**，而是把 fd 投进 io_returns_ 队列
//     （eventfd 唤醒 reactor），由 reactor 统一 close 或重新登记。
//     这条规则消除了"worker close 之后 fd 号被 reaccept 复用、而陈旧登记还没
//     清掉"的窗口——旧实现在 worker 里直接 close(fd) 就出在这里。
//   - 流式响应 + keep-alive：流式响应用 Transfer-Encoding: chunked 承载
//     （见 response.h 的 build_response_head），因此流式响应结束后连接同样可以
//     复用。为什么不用"流式强制 Connection: close"：那会让每个 SSE 请求都重建
//     一次 TCP 连接，而 SSE 正是长连接场景；chunked 是 HTTP/1.1 为此设计的标准
//     机制，curl / 浏览器 / OpenAI SDK 都原生支持。
//     所有响应都不再使用"无长度 + close 界定"的第三种语义（HTTP/1.0 风格），
//     因为在 keep-alive 下它是歧义的。
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/config.h"
#include "server/connection_handler.h"
#include "server/router.h"
#include "server/response.h"
#include "common/thread_pool.h"

// 注意：pool_ 的声明顺序必须在 conn_handler_ 之后（析构顺序相反）
// pool_ 先析构 join 所有任务 此时 conn_handler_ 仍有效 线程池执行execute任务传入lambda时使用[this]捕获是安全的

namespace ai_gateway {

// 可移动的原子布尔：std::atomic 不可拷贝/移动，而存储容器的元素构造要求
// 可移动（否则只能用已经在别处构造好的对象去 move 构造）。这里的 move 语义
// 只用于容器内部迁移，因此"搬走即清空源"是安全的
class MovableFlag {
 public:
  MovableFlag() = default;
  explicit MovableFlag(bool v) : v_(v) {}
  MovableFlag(const MovableFlag&) = delete;
  MovableFlag& operator=(const MovableFlag&) = delete;
  MovableFlag(MovableFlag&& o) noexcept : v_(o.v_.load()) { o.v_.store(false); }
  MovableFlag& operator=(MovableFlag&& o) noexcept {
    v_.store(o.v_.load());
    o.v_.store(false);
    return *this;
  }
  void store(bool v) { v_.store(v); }
  bool load() const { return v_.load(); }

 private:
  std::atomic<bool> v_{false};
};

// 写路径的结果（send_all_with_deadline 的返回值，也是 ResponseWriter 的内部状态）
enum class WriteStatus {
  kOk = 0,    // 全部写完
  kDeadline,  // 超过写死线（慢客户端）
  kSocket,    // 对端关闭 / EPIPE / ECONNRESET 等真实套接字错误
};

// 写路径失败原因的对外口径（映射到 llm_client 的 StreamAbortReason 语义）
enum class AbortReason {
  kNone = 0,
  kClientGone,
  kDeadline,  // 响应的**总死线**到点（只可能出现在缓冲式响应上）
  kIdle,      // 流式响应的**空闲死线**到点：两次成功写入之间太久没有进展
};

// 非阻塞 fd 上"写到底"：遇 EAGAIN 用 poll(POLLOUT) 等可写再续发，直到写完 /
// 超过死线 / 出现真实错误。
//   write_deadline_ms —— 单次调用的写死线（相对值）
//   total_deadline    —— **整条响应**的绝对死线（绝对 time_point）。
//                        对流式响应必须传 time_point::max()：流式响应的时长由
//                        回答长度决定（输出几千 token 就是几十秒到几分钟），
//                        任何总时限都等于给回答长度设上限。流式靠"写入之间的
//                        空闲超时"（ResponseWriter 内部续期）防住慢客户端。
//                        缓冲式响应则需要它：把 N 次 send 累计起来仍受限，否则
//                        一个"每次只读一点点"的客户端能让这条连接无限期占用 worker
//   idle_deadline     —— 单次停顿允许的最长等待时间（绝对 time_point）。
//                        poll 等到可写后循环从头再试，因此这是"一次停顿"的上限
//                        而不是整条响应的上限——即"空闲超时"。默认 max()
// 自由函数以便脱离 HTTP 服务单测（socketpair，见 test_http_server 第 6 段）
WriteStatus send_all_with_deadline_until(
    int client_fd, const char* data, size_t len, int write_deadline_ms,
    std::atomic<uint64_t>* eagain_count,
    std::chrono::steady_clock::time_point total_deadline,
    std::chrono::steady_clock::time_point idle_deadline);

// 兼容入口（缓冲式调用方）：idle_deadline 缺省 = total_deadline
WriteStatus send_all_with_deadline(
    int client_fd, const char* data, size_t len, int write_deadline_ms,
    std::atomic<uint64_t>* eagain_count,
    std::chrono::steady_clock::time_point total_deadline =
        std::chrono::steady_clock::time_point::max());

// 增量响应写出器：按顺序把"响应头 + 若干块正文"写到客户端。
// 两种长度语义：Content-Length（普通响应）与 chunked（流式响应）。
// 失败后自身进入 error 状态，后续 write_* 全部直接返回失败（不再怼 socket）。
//
// 两种死线语义（这是"长流不被切断"的关键）：
//   缓冲式（write_head）：整条响应受 total_deadline_ 约束
//   流式（write_stream_head）：**不设总死线**，改为"两次成功写入之间的空闲超时"
//     stream_idle_ms_ —— 每次成功写出数据后把空闲死线推到 now + 空闲值。
//     只要客户端在跟读，流要多长都行；客户端卡住不读超过空闲值就中停上游
class ResponseWriter {
 public:
  ResponseWriter(int client_fd, int write_deadline_ms,
                 std::chrono::steady_clock::time_point total_deadline,
                 std::atomic<uint64_t>* eagain_count,
                 int stream_idle_ms = 0)
      : fd_(client_fd),
        write_deadline_ms_(write_deadline_ms),
        total_deadline_(total_deadline),
        stream_idle_ms_(stream_idle_ms > 0 ? stream_idle_ms : write_deadline_ms),
        eagain_count_(eagain_count) {}

  // 写响应头（含 Content-Length）。keep_alive 只影响 Connection 头的声明，
  // 是否真的复用由调用方决定。返回 false = 写失败或已经写过响应头
  bool write_head(int status_code, std::string_view content_type,
                  size_t content_length, bool keep_alive);
  // 流式响应头：Transfer-Encoding: chunked
  bool write_stream_head(int status_code, std::string_view content_type,
                         bool keep_alive);
  // 写正文（内部按需分包；chunked 模式下自动补长度字段）
  bool write_body(std::string_view data);
  // 结束流式响应（写 0 长度终止块）。非流式模式下是 no-op
  bool finish_stream();
  // 把缓冲区里还没发出去的数据发完
  bool flush();

  bool committed() const { return committed_; }  // 响应头已经发出去了
  bool failed() const { return failed_; }
  bool chunked() const { return chunked_; }
  AbortReason abort_reason() const { return abort_; }
  uint64_t bytes_sent() const { return bytes_sent_; }

 private:
  // 发送 data：按当前模式选取死线（流式 = 空闲死线，缓冲式 = 总死线）。
  // 成功且是流式时把空闲死线推到 now + stream_idle_ms_（这就是"续期"）
  bool send_raw(std::string_view data);
  std::chrono::steady_clock::time_point current_total_deadline() const;
  std::chrono::steady_clock::time_point current_idle_deadline() const;

  int fd_;
  int write_deadline_ms_;
  std::chrono::steady_clock::time_point total_deadline_;
  int stream_idle_ms_;
  // 流式（chunked）模式下的空闲死线：**绝对时刻**，每次成功写出后向前推。
  // 只有 chunked_ 为真时才有意义
  std::chrono::steady_clock::time_point stream_idle_deadline_{};
  std::atomic<uint64_t>* eagain_count_;

  std::string out_;  // 未发完的待发缓冲
  bool committed_ = false;
  bool chunked_ = false;
  bool failed_ = false;
  AbortReason abort_ = AbortReason::kNone;
  uint64_t bytes_sent_ = 0;
};

// 请求处理器签名：接收请求体 JSON + 响应写出器 + 请求附加信息，返回响应内容。
// 绝大多数 handler 只用返回值（缓冲式响应）；需要边收边发的 handler（SSE 透传）
// 直接用 writer 写，用 writer.committed() 表达"已自行完成"。
// info 里目前只有客户端的连接复用意愿——流式 handler 需要它来正确写 Connection 头
using RequestHandler = std::function<HttpReply(
    const std::string& request_body, ResponseWriter& writer,
    const HttpRequestInfo& info)>;

class HttpServer {
 public:
  explicit HttpServer(const ServerConfig& config);
  ~HttpServer();

  // 禁止拷贝/移动（管理 fd 等资源）
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  // 注册路由处理器（默认注册到 POST /v1/chat/completions）
  void set_handler(RequestHandler handler);

  // 注册自定义路由（默认 POST）
  void add_route(std::string_view path, RequestHandler handler);

  // 注册指定 method 的自定义路由（例如 GET /metrics）。
  // 路由 key 是 "METHOD /path"，因此新增 GET 端点不会影响既有的 POST 路由
  void add_route(std::string_view method, std::string_view path,
                 RequestHandler handler);

  // 启动服务（阻塞当前线程直到 stop() 被调用）
  // external_shutdown: 可选的原子标志，信号处理器等外部代码可通过它触发优雅关闭
  void run(std::atomic<bool>* external_shutdown = nullptr);

  // 停止服务（只停 reactor；在途请求由 drain() 等待）
  void stop();

  // 关闭流程第二阶段：等待线程池里所有在途请求执行完再返回。
  // 必须在 run() 返回后调用——否则 main 的统计输出/落盘会与仍在运行的 worker
  // 并发访问 LruStore / CacheEngine / Stats（报告 M12）。
  // worker 一律不自己 close(fd)，因此这里还要把 io_returns_ 里剩下的 fd 处理掉，
  // 否则进程退出会漏掉这些 fd
  void drain();

  // 实际监听端口（config.port 为 0 时由内核分配）
  int listen_port() const { return listen_port_; }

  // 当前登记中的连接数（用于观测与测试：确认半关闭/超时连接确实被回收）。
  // 含借给 worker 的连接（worker_owned），因此空闲超时/回收断言的口径不变
  size_t active_connections() const { return conns_.size(); }

  // 线程池观测（/metrics 用）：两者都由 ThreadPool 的互斥量保护，可从任意线程调用。
  // 注意**不要**在 worker 里读 active_connections()——conns_ 由 reactor 独占，
  // 那是数据竞争
  size_t pending_tasks() const { return pool_.pending(); }
  size_t active_tasks() const { return pool_.active(); }
  size_t worker_threads() const { return pool_.size(); }

  // 写路径上真正遇到 EAGAIN 的次数（原子计数，任意线程可读）。
  // 用途：慢客户端回归测试用它证明"确实走到了非阻塞写要等待可写的分支"，
  // 否则用例可能只是碰巧没触发 EAGAIN 而"通过"
  uint64_t send_eagain_count() const {
    return send_eagain_count_.load(std::memory_order_relaxed);
  }

  // keep-alive 复用次数（同一连接上第 2 个及以后的请求各计一次）。
  // 用它证明"同一连接确实被复用了"，而不只是"客户端碰巧连了两次"
  uint64_t reused_connection_count() const {
    return reused_connections_.load(std::memory_order_relaxed);
  }

  // 累计 accept 成功的连接数（诊断/基准用）：它直接等于"服务端执行的 TCP
  // 握手次数"，是 keep-alive 收益最客观的度量——同一连接上 N 个请求只算 1 次
  uint64_t accepted_connections() const {
    return accepted_connections_.load(std::memory_order_relaxed);
  }

  // 连接被重新登记为可复用空闲连接的次数（每次归还都算）。
  // 测试用它做同步点：等这条连接确实挂回 epoll 之后再发下一个请求，
  // 否则客户端可能在服务端重新登记之前发字节，那一瞬间没有 epoll 事件来源
  uint64_t keep_alive_rearms() const {
    return keep_alive_rearms_.load(std::memory_order_relaxed);
  }

 private:
  // 创建非阻塞监听 socket
  int create_listen_socket();

  // 处理单个客户端连接
  void handle_client(int client_fd);

  // recv 的结果语义（报告 H5：旧实现用 bool，把"对端关闭"与"本轮无数据"混在一起）
  enum class ReadState {
    kData,   // 读到了数据（可能后续还有，调用方按 ET 循环已读到 EAGAIN）
    kEof,    // 对端关闭写端（收到 FIN）
    kAgain,  // 当前无数据可读（EAGAIN/EWOULDBLOCK）
    kError,  // 真实错误
  };
  ReadState read_into_buffer(int client_fd, std::string& buf);

  // 关闭并清理一个连接（从 epoll 摘除 + erase 状态 + close）。
  // 只能在 reactor 线程调用；worker 走 return_fd() 归还所有权
  void close_connection(int client_fd);

  // 非阻塞 fd 上"写到底"。必须在 worker 里调用：可能阻塞（这正是慢客户端
  // 只会拖住一个 worker 的代价）
  WriteStatus send_all(int client_fd, const char* data, size_t len,
                       int write_deadline_ms,
                       std::chrono::steady_clock::time_point total_deadline);

  // 用累积缓冲区判断请求是否完整并推进状态机；
  // 返回 true 表示该连接已被消费（提交线程池或已关闭），false 表示仍需等待更多数据
  bool handle_buffer(int client_fd, bool peer_closed);

  // 每轮 epoll 节拍上的连接维护：主动读取所有连接（ET 边沿可能丢掉"只发 FIN"
  // 的可读事件）+ 关闭空闲超时的连接 + 处理 worker 归还的 fd
  void maintain_connections();

  // worker 写完后的归还动作
  enum class FdAction : uint8_t {
    kClose = 0,  // 关闭（非 keep-alive / 出错 / 已请求停机）
    kKeepAlive,  // 重新登记进 conns_ 并挂回 epoll
  };
  struct FdReturn {
    int fd;
    FdAction action;
    uint64_t token = 0;  // 借用令牌：reactor 用它识别过期归还
  };
  // 任何线程可调用：把 fd 的所有权交还 reactor
  void return_fd(int fd, FdAction action, uint64_t token);
  // reactor：登记一次借出（worker 用 try_claim_borrow 原子认领）
  void grant_borrow(uint64_t token);
  // worker：认领借出。false = 这次借出已被撤销（fd 已被 reactor 关掉/复用）
  bool try_claim_borrow(uint64_t token);
  // reactor 线程：处理归还队列。stop_requested=true 时全部按 close 处理
  void process_returned_fds(bool stop_requested);
  // reactor 线程：把 fd 重新登记为可复用的空闲连接。
  // token 必须与该连接当前的 borrow_token 一致，否则说明这个 fd 号已被复用，
  // 本次归还作废（不能动新连接的状态）
  void rearm_keep_alive(int fd, uint64_t token);

  // 设置 fd 为非阻塞
  static void set_nonblocking(int fd);

  ServerConfig config_;
  int listen_fd_ = -1;
  int listen_port_ = 0;
  int epoll_fd_ = -1;
  int wake_fd_ = -1;  // eventfd：worker 归还 fd 时唤醒 reactor
  std::atomic<bool> running_{false};
  Router router_;
  ConnectionHandler conn_handler_{router_};
  ThreadPool pool_{4};  // 单 reactor + 线程池架构

  // 每个连接的累积接收缓冲与最近活动时间。
  // reactor 线程独占访问 conns_/conn_index_；worker 只通过 return_fd() 归还，
  // 从不直接读写这两张表——因此结构本身不需要同步。
  struct Connection {
    Connection() = default;
    // 含 atomic 成员，因此不能拷贝/移动；用显式构造在 emplace 时就地初始化
    explicit Connection(std::chrono::steady_clock::time_point t)
        : last_activity(t) {}

    std::string buf;
    std::chrono::steady_clock::time_point last_activity;
    // 该连接当前是否借给了某个 worker：reactor 见到 true 就跳过（不读不关），
    // 由 worker 通过归还队列交还所有权。worker 写、reactor 读，因此是原子量
    MovableFlag worker_owned;
    // 这条连接上已经处理过的请求数（reactor 独占访问）。>1 即发生了 keep-alive
    // 复用，reused_connections_ 据此累加——用它区分"真复用"与"客户端恰好连了两次"
    uint64_t requests_served = 0;
    // fd 借用令牌（"第几轮接管"）。每登记一条新连接时 +1，worker 归还时核对：
    // 期间这个 fd 号被 close 后又被 accept 复用的话，令牌已经变了，旧的归还请求
    // 必须被丢弃。没有这道校验，"worker 借 fd -> reactor 因空闲超时关闭它 ->
    // 新连接恰好拿到同一个 fd 号 -> 旧 worker 的归还作用到新连接上"会造成
    // 新连接被莫名重新登记/关闭（本轮实测到过：keep-alive 复用因此完全失效）
    uint64_t borrow_token = 0;
  };
  // 用 deque 而非 unordered_map：扫描时顺序遍历性能更好。
  // 不用 list 的原因：Connection 含原子量，list 的 emplace 需要由"已构造好的
  // 元素"移动构造节点（pair 构造），而 deque 的 emplace_back 是就地构造。
  // 元素地址在两端增删时不失效这点两者一致：worker 只持 fd 号，不持迭代器
  std::deque<std::pair<int, Connection>> conns_;
  std::unordered_map<int, std::deque<std::pair<int, Connection>>::iterator>
      conn_index_;

  // worker -> reactor 的 fd 归还队列（归还 = close 或重新登记）
  std::mutex io_mutex_;
  std::vector<FdReturn> io_returns_;
  // 已"借出"给 worker 的借用令牌集合（受 io_mutex_ 保护）。
  // 为什么需要一张单独的注册表而不是让 worker 直接查 conn_index_：
  // conn_index_ 由 reactor 独占，worker 读它本身就是数据竞争（reactor 可能在
  // 并发 erase）。worker 只在这里做一次"我的令牌还在不在"的原子判定，
  // 不在就说明 fd 已被 reactor 关闭/复用，任务必须放弃
  std::unordered_set<uint64_t> borrowed_tokens_;

  // 写路径遇到 EAGAIN 的次数（worker 写，测试/metrics 读 -> 必须原子）
  std::atomic<uint64_t> send_eagain_count_{0};
  // keep-alive 复用次数（同一个连接上第 2 个及以后的请求）
  std::atomic<uint64_t> reused_connections_{0};
  // 重新登记次数（测试同步点，见 keep_alive_rearms()）
  std::atomic<uint64_t> keep_alive_rearms_{0};
  // 累计 accept 次数（诊断/基准用）
  std::atomic<uint64_t> accepted_connections_{0};
};

}  // namespace ai_gateway
