// HttpServer 连接生命周期回归（报告 H5 / M12）
//
// 覆盖点：
//   1. 半关闭（头部完整 + body 未发齐 + 客户端 FIN）→ 服务端立即回收 fd，
//      不再把该连接永久留在 epoll 里等一个永不到来的可读事件
//   2. 空闲超时：只连不发的连接在 idle_timeout_seconds 后被关闭
//   3. max_connections：达到上限后新连接直接收到 503
//   4. drain()：在途请求执行完毕后才返回（M12）
//
// 判别力说明：把 read_into_buffer 改回"EOF 也返回 true、body 未收齐时直接 return"
// 的旧逻辑后，第 1 段断言失败（fd 数不回落）；去掉 accept 处的 max_connections
// 检查后，第 3 段的 503 断言失败；把 drain() 改成空实现后，第 4 段
// "drain 返回时在途 handler 已完成"失败。
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "server/http_server.h"
#include "test_check.h"

using namespace ai_gateway;
using namespace std::chrono_literals;

namespace {

int connect_to(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

// 等某个谓词成立，最多等 timeout_ms
template <typename Pred>
bool wait_for(Pred pred, int timeout_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(50ms);
  }
  return false;
}

// 等到服务端开始监听（config.port=0 时端口由内核分配）
bool wait_listening(HttpServer& server, int timeout_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (server.listen_port() > 0) return true;
    std::this_thread::sleep_for(10ms);
  }
  return false;
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1. 半关闭回收（空闲超时设得足够长，回收只能来自 FIN 处理）─────────
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 30;  // 刻意远大于断言等待窗口
    sc.max_connections = 256;

    std::atomic<bool> stop_flag{false};
    std::atomic<int> served{0};
    HttpServer server(sc);
    server.set_handler([&served](const std::string& body) {
      ++served;
      return HttpReply{200, "application/json", "{\"echo\":\"" + body + "\"}"};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    // 头部完整 + Content-Length 声明 100 字节但只发 10 字节，然后 FIN
    constexpr int kHalf = 30;
    std::vector<int> half;
    const char* partial =
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
        "Content-Type: application/json\r\nContent-Length: 100\r\n\r\n0123456789";
    bool all_sent = true;
    for (int i = 0; i < kHalf; ++i) {
      int fd = connect_to(port);
      if (fd < 0) { all_sent = false; break; }
      if (send(fd, partial, std::strlen(partial), MSG_NOSIGNAL) <= 0)
        all_sent = false;
      shutdown(fd, SHUT_WR);  // 发 FIN：服务端应立刻回收
      half.push_back(fd);
    }
    CHECK(all_sent);
    ok++;
    // 修复前：这些连接永久驻留，active_connections 恒为 kHalf（30 > 3s 窗口）
    CHECK(wait_for([&] { return server.active_connections() == 0; }, 3000));
    ok++;
    // 半关闭连接的请求不完整，不应进入业务 handler
    CHECK(served.load() == 0);
    ok++;
    for (int fd : half) close(fd);
    stop_flag.store(true);
    t.join();
    server.drain();
  }

  // ── 2. 空闲超时：只连不发，服务端应主动关闭 ─────────────────────────
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 2;
    sc.max_connections = 256;

    std::atomic<bool> stop_flag{false};
    HttpServer server(sc);
    server.set_handler([](const std::string&) {
      return HttpReply{200, "application/json", "{}"};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    int fd = connect_to(port);
    CHECK(fd >= 0);
    ok++;
    if (fd >= 0) {
      bool closed_by_server = false;
      auto t0 = std::chrono::steady_clock::now();
      while (std::chrono::steady_clock::now() - t0 < 6s) {
        char c = 0;
        ssize_t n = recv(fd, &c, 1, MSG_DONTWAIT);
        if (n == 0) { closed_by_server = true; break; }
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) break;
        std::this_thread::sleep_for(100ms);
      }
      CHECK(closed_by_server);
      ok++;
      close(fd);
    }
    CHECK(server.active_connections() == 0);
    ok++;
    stop_flag.store(true);
    t.join();
    server.drain();
  }

  // ── 4. drain()：在途请求执行完毕后才返回（M12）──────────────────────
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 30;
    sc.max_connections = 64;

    std::atomic<bool> stop_flag{false};
    std::atomic<bool> slow_done{false};
    HttpServer server(sc);
    server.set_handler([](const std::string&) {
      return HttpReply{200, "application/json", "{}"};
    });
    server.add_route("/slow", [&slow_done](const std::string&) {
      std::this_thread::sleep_for(300ms);
      slow_done.store(true);
      return HttpReply{200, "application/json", "{\"ok\":true}"};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    int fd = connect_to(port);
    CHECK(fd >= 0);
    ok++;
    if (fd >= 0) {
      const char* req =
          "POST /slow HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}";
      send(fd, req, std::strlen(req), MSG_NOSIGNAL);
      std::this_thread::sleep_for(100ms);  // 让 worker 开始执行 handler
      CHECK(!slow_done.load());
      ok++;
      stop_flag.store(true);
      t.join();         // reactor 退出：不再接受新请求
      server.drain();   // 必须等到在途 handler 跑完
      CHECK(slow_done.load());
      ok++;
      close(fd);
    } else {
      stop_flag.store(true);
      t.join();
      server.drain();
    }
  }

  // ── 3. max_connections 上限 → 503 ────────────────────────────────────
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 30;  // 足够长，保证占用态稳定
    sc.max_connections = 3;

    std::atomic<bool> stop_flag{false};
    HttpServer server(sc);
    server.set_handler([](const std::string&) {
      return HttpReply{200, "application/json", "{}"};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    std::vector<int> held;
    for (int i = 0; i < 3; ++i) {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      held.push_back(fd);
    }
    std::this_thread::sleep_for(200ms);  // 让主线程 accept 并登记这 3 条

    int extra = connect_to(port);
    CHECK(extra >= 0);
    ok++;
    if (extra >= 0) {
      char buf[512] = {0};
      ssize_t n = recv(extra, buf, sizeof(buf) - 1, 0);
      CHECK(n > 0);
      ok++;
      CHECK(std::string(buf, n > 0 ? static_cast<size_t>(n) : 0)
                .find("503") != std::string::npos);
      ok++;
      close(extra);
    }
    for (int fd : held) close(fd);
    stop_flag.store(true);
    t.join();
    server.drain();
  }

  return test_check::finish("test_http_server", ok);
}
