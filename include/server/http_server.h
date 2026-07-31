// HTTP 服务器核心：epoll ET + 非阻塞 IO + 线程池
// 主线程 accept + epoll_wait 提交到线程池 worker 处理 + send + close
//     epoll_wait (主线程)
//       ├─ listen_fd EPOLLIN accept 后 EPOLL_CTL_ADD 新 client_fd
//       └─ client_fd EPOLLIN 然后 EPOLL_CTL_DEL 然后 handle_client p//ool_.execute
//                                                      ↓
//           handle_client:  recv (非阻塞) 然后 fcntl 切阻塞 然后 200ms × 3 重试 然后 装包
//                           pool_.execute(lambda):
//                             conn_handler_.process 然后 send 然后 close(fd)
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/config.h"
#include "server/connection_handler.h"
#include "server/router.h"
#include "common/thread_pool.h"

// 注意：pool_ 的声明顺序必须在 conn_handler_ 之后（析构顺序相反）
// pool_ 先析构 join 所有任务 此时 conn_handler_ 仍有效 线程池执行execute任务传入lambda时使用[this]捕获是安全的

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
  // external_shutdown: 可选的原子标志，信号处理器等外部代码可通过它触发优雅关闭
  void run(std::atomic<bool>* external_shutdown = nullptr);

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
  ThreadPool pool_{4};  // 单 reactor + 线程池架构
};

}  // namespace ai_gateway
