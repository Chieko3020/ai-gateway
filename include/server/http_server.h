// HTTP 服务器核心：epoll ET + 非阻塞 IO + 线程池
// 主线程 accept + epoll_wait 提交到线程池 worker 处理 + send + close
//     epoll_wait (主线程)
//       ├─ listen_fd EPOLLIN accept 后 EPOLL_CTL_ADD 新 client_fd
//       └─ client_fd EPOLLIN 然后 EPOLL_CTL_DEL 然后 handle_client
//                                                      ↓
//           handle_client: 非阻塞循环 recv 追加到累积缓冲区 conns_[fd].buf
//                          解析 Content-Length 判断 body 是否完整
//                          完整 → pool_.execute: process + send + close
//                          不完整 → 保持注册等待更多数据（受空闲超时约束）
//
// 连接生命周期（报告 H5）：
//   - recv 三态：收到数据 / 对端 EOF / 错误或"当前无数据"（EAGAIN）
//     半关闭（收到 FIN）且请求不完整 → 立即关闭，不再等一个永远不来的可读事件
//   - 每连接空闲超时 idle_timeout_seconds：epoll_wait 的 100ms 节拍上扫描，
//     超时立即关闭，避免只有"有数据到达才刷新"的时间语义
//   - max_connections 上限：超出时对新连接直接回 503 并关闭
#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/config.h"
#include "server/connection_handler.h"
#include "server/router.h"
#include "common/thread_pool.h"

// 注意：pool_ 的声明顺序必须在 conn_handler_ 之后（析构顺序相反）
// pool_ 先析构 join 所有任务 此时 conn_handler_ 仍有效 线程池执行execute任务传入lambda时使用[this]捕获是安全的

namespace ai_gateway {

// 请求处理器签名：接收请求体 JSON，返回响应内容（状态码 + 响应体）
using RequestHandler = std::function<HttpReply(const std::string& request_body)>;

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
  void drain();

  // 实际监听端口（config.port 为 0 时由内核分配）
  int listen_port() const { return listen_port_; }

  // 当前登记中的连接数（用于观测与测试：确认半关闭/超时连接确实被回收）。
  // 该表由 reactor 线程独占，因此只在 reactor 停下后调用才有确定语义
  size_t active_connections() const { return conns_.size(); }

  // 线程池观测（/metrics 用）：两者都由 ThreadPool 的互斥量保护，可从任意线程调用。
  // 注意**不要**在 worker 里读 active_connections()——conns_ 由 reactor 独占，
  // 那是数据竞争
  size_t pending_tasks() const { return pool_.pending(); }
  size_t active_tasks() const { return pool_.active(); }
  size_t worker_threads() const { return pool_.size(); }

 private:
  // 创建非阻塞监听 socket
  int create_listen_socket();

  // 处理单个客户端连接
  void handle_client(int client_fd);

  // recv 的结果语义（报告 H5：旧实现用 bool，把"对端关闭"与"本轮无数据"混在一起）
  enum class ReadState {
    kData,    // 读到了数据（可能后续还有，调用方按 ET 循环已读到 EAGAIN）
    kEof,     // 对端关闭写端（收到 FIN）
    kAgain,   // 当前无数据可读（EAGAIN/EWOULDBLOCK）
    kError,   // 真实错误
  };
  ReadState read_into_buffer(int client_fd, std::string& buf);

  // 关闭并清理一个连接（从 epoll 摘除 + erase 状态 + close）
  void close_connection(int client_fd);

  // 用累积缓冲区判断请求是否完整并推进状态机；
  // 返回 true 表示该连接已被消费（提交线程池或已关闭），false 表示仍需等待更多数据
  bool handle_buffer(int client_fd, bool peer_closed);

  // 每轮 epoll 节拍上的连接维护：主动读取所有连接（ET 边沿可能丢掉"只发 FIN"
  // 的可读事件）+ 关闭空闲超时的连接
  void maintain_connections();

  // 设置 fd 为非阻塞
  static void set_nonblocking(int fd);

  ServerConfig config_;
  int listen_fd_ = -1;
  int listen_port_ = 0;
  int epoll_fd_ = -1;
  std::atomic<bool> running_{false};
  Router router_;
  ConnectionHandler conn_handler_{router_};
  ThreadPool pool_{4};  // 单 reactor + 线程池架构

  // 每个连接的累积接收缓冲与最近活动时间。
  // 主线程 epoll 循环独占访问（无需锁）；worker 只 close(fd)，不碰这张表，
  // 因此结构本身不需要同步。
  struct Connection {
    std::string buf;
    std::chrono::steady_clock::time_point last_activity;
  };
  // 用 list 而非 unordered_map：扫描时顺序遍历性能更好，且增删不影响其它元素
  std::list<std::pair<int, Connection>> conns_;
  std::unordered_map<int, std::list<std::pair<int, Connection>>::iterator>
      conn_index_;
};

}  // namespace ai_gateway
