// HTTP keep-alive 回归（本轮新增：此前恒 Connection: close）
//
// 覆盖点：
//   1. HTTP/1.1 同一连接连续 N 个请求都被正确解析与响应（请求边界靠内容消费推进）
//   2. 显式 Connection: close 必须被遵守（响应头回 close，且服务端随后关闭连接）
//   3. HTTP/1.0 默认不复用；显式 Connection: keep-alive 才复用
//   4. 空闲超时对复用的空闲连接同样生效（与 idle_timeout_seconds 一致）
//   5. GET /metrics 与 POST /v1/chat/completions 可以混在同一条连接上
//   6. 复用计数（reused_connection_count）证明"真复用"而非"客户端连了两次"
//
// 判别力说明：
//   - 把 rearm_keep_alive 去掉（回到"每次响应后 close"）：第 1/5 段的
//     "同一连接第二个请求得到响应"断言失败（recv 返回 0/EOF）
//   - 把 wants_keep_alive 改回"恒 false"：第 1 段失败；改成"恒 true"：
//     第 2 段的 close 断言失败
//   - 空闲超时断言在新连接建立前就成立（第一次请求后 30s 才关）——用 2s 的配置
//     让第 4 段在秒级内可判定
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
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

bool wait_listening(HttpServer& server, int timeout_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (server.listen_port() > 0) return true;
    std::this_thread::sleep_for(10ms);
  }
  return false;
}

template <typename Pred>
bool wait_for(Pred pred, int timeout_ms) {
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(20ms);
  }
  return false;
}

// 从 fd 上读一个完整响应：按 Content-Length 界定 body（keep-alive 下不能等 EOF）
// 返回 false 表示读失败/超时
bool read_one_response(int fd, std::string& out, int timeout_ms = 5000,
                       size_t* consumed = nullptr) {
  out.clear();
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  size_t body_start = std::string::npos;
  size_t content_length = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    if (body_start == std::string::npos) {
      auto hdr = out.find("\r\n\r\n");
      if (hdr != std::string::npos) {
        body_start = hdr + 4;
        auto cl = out.find("Content-Length: ");
        if (cl != std::string::npos && cl < hdr) {
          content_length = std::strtoul(out.c_str() + cl + 16, nullptr, 10);
        } else {
          content_length = 0;  // chunked 或无 body：本用例不依赖它
        }
      }
    }
    if (body_start != std::string::npos &&
        out.size() >= body_start + content_length) {
      if (consumed) *consumed = body_start + content_length;
      return true;
    }
    // 先用 poll 带超时等可读：直接对阻塞 socket 调 recv 的话，没有响应时
    // 会一直阻塞在里面，上面算出来的 deadline 就是摆设——"服务端没答"这种
    // 失败会表现成用例挂死而不是断言失败
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                          deadline - std::chrono::steady_clock::now())
                          .count();
    if (left <= 0) break;
    pollfd p{fd, POLLIN, 0};
    int prc = ::poll(&p, 1, static_cast<int>(left));
    if (prc == 0) break;  // 超时：这段时间内没有任何可读事件
    if (prc < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    char buf[4096];
    ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n > 0) {
      out.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) return false;  // 服务端关闭了：keep-alive 下说明复用没生效
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      std::this_thread::sleep_for(1ms);
      continue;
    }
    return false;
  }
  return false;
}

const char* kPostBody = R"({"model":"m","messages":[]})";

std::string post_request(bool keep_alive) {
  std::string r = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                  "Content-Type: application/json\r\nContent-Length: " +
                  std::to_string(std::strlen(kPostBody)) + "\r\n";
  r += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
  r += "\r\n";
  r += kPostBody;
  return r;
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1~3, 5, 6. keep-alive 复用与 close 遵守 ─────────────────────────
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 30;
    sc.max_connections = 64;

    std::atomic<bool> stop_flag{false};
    std::atomic<int> hits{0};
    HttpServer server(sc);
    server.set_handler([&hits](const std::string& , ResponseWriter&, HttpRequestInfo&) {
      ++hits;
      return HttpReply{200, "application/json", R"({"ok":true})"};
    });
    server.add_route("GET", "/metrics", [](const std::string& , ResponseWriter&, HttpRequestInfo&) {
      return HttpReply{200, "text/plain", "ai_gateway_up 1\n"};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    // 1. 同一连接 5 个 POST：全部得到响应，且确实发生了复用
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        const uint64_t reuse_before = server.reused_connection_count();
        const uint64_t accept_before = server.accepted_connections();
        int answered = 0;
        for (int i = 0; i < 5; ++i) {
          std::string req = post_request(true);
          send(fd, req.data(), req.size(), MSG_NOSIGNAL);
          std::string resp;
          if (read_one_response(fd, resp) &&
              resp.find("200 OK") != std::string::npos)
            ++answered;
        }
        CHECK(answered == 5);
        ok++;
        CHECK(hits.load() == 5);
        ok++;
        std::fprintf(stderr,
                     "[keep-alive] answered=%d hits=%d reuse %llu -> %llu, "
                     "累计 accept=%llu\n",
                     answered, hits.load(),
                     (unsigned long long)reuse_before,
                     (unsigned long long)server.reused_connection_count(),
                     (unsigned long long)server.accepted_connections());
        // 5 个请求只应发生 1 次 accept：这是 keep-alive 最硬的判据
        // （判别力：把 keep-alive 关掉后这个数是 +5）
        CHECK(server.accepted_connections() <= accept_before + 1);
        ok++;
        // 复用计数：同一连接上"重新登记"的次数。
        // 注意口径：连续 pipelining 的多个请求可能在**同一个**归还/重新登记
        // 周期里被消费（rearm_keep_alive 末尾会把缓冲区里已到的下一个请求直接
        // 推进状态机），因此计数 == 重新登记次数，而不是请求数减一。
        // 这里断言 >= 3 就足以证明"确实复用了"（恒 close 的实现是 0）
        // 计数在 worker 归还之后递增，客户端读到响应时可能还没到，等一拍
        CHECK(wait_for(
            [&] { return server.reused_connection_count() >= reuse_before + 3; },
            2000));
        ok++;
        close(fd);
      }
    }

    // 2. 显式 Connection: close：响应头回 close，且服务端主动关连接（读到 EOF）
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        std::string req = post_request(false);
        send(fd, req.data(), req.size(), MSG_NOSIGNAL);
        std::string resp;
        auto deadline = std::chrono::steady_clock::now() + 5s;
        bool eof = false;
        while (std::chrono::steady_clock::now() < deadline) {
          char buf[4096];
          ssize_t n = recv(fd, buf, sizeof(buf), 0);
          if (n > 0) {
            resp.append(buf, static_cast<size_t>(n));
            continue;
          }
          if (n == 0) {
            eof = true;
            break;
          }
          if (errno == EINTR) continue;
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            std::this_thread::sleep_for(1ms);
            continue;
          }
          break;
        }
        CHECK(resp.find("Connection: close") != std::string::npos);
        ok++;
        CHECK(eof);  // 客户端要求 close，服务端就必须关（不能挂住等下一个请求）
        ok++;
        close(fd);
      }
    }

    // 3. HTTP/1.0：默认不复用（除非显式 keep-alive）
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        std::string req =
            std::string("POST /v1/chat/completions HTTP/1.0\r\nHost: x\r\n"
                        "Content-Length: ") +
            std::to_string(std::strlen(kPostBody)) + "\r\n\r\n" + kPostBody;
        send(fd, req.data(), req.size(), MSG_NOSIGNAL);
        std::string resp;
        auto deadline = std::chrono::steady_clock::now() + 5s;
        bool eof = false;
        while (std::chrono::steady_clock::now() < deadline) {
          char buf[4096];
          ssize_t n = recv(fd, buf, sizeof(buf), 0);
          if (n > 0) {
            resp.append(buf, static_cast<size_t>(n));
            continue;
          }
          if (n == 0) {
            eof = true;
            break;
          }
          if (errno == EINTR) continue;
          std::this_thread::sleep_for(1ms);
        }
        CHECK(resp.find("HTTP/1.1 200 OK") != std::string::npos);
        ok++;
        CHECK(resp.find("Connection: close") != std::string::npos);
        ok++;
        CHECK(eof);
        ok++;
        close(fd);
      }
    }

    // 5. GET /metrics 与 POST 混在同一条连接上
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        const char* get = "GET /metrics HTTP/1.1\r\nHost: x\r\n\r\n";
        send(fd, get, std::strlen(get), MSG_NOSIGNAL);
        std::string r1;
        CHECK(read_one_response(fd, r1));
        ok++;
        CHECK(r1.find("200 OK") != std::string::npos);
        ok++;
        CHECK(r1.find("ai_gateway_up 1") != std::string::npos);
        ok++;

        std::string req = post_request(true);
        send(fd, req.data(), req.size(), MSG_NOSIGNAL);
        std::string r2;
        CHECK(read_one_response(fd, r2));
        ok++;
        CHECK(r2.find("200 OK") != std::string::npos);
        ok++;
        close(fd);
      }
    }

    // 请求边界：把两个请求**一次性写进同一个 TCP 段**（pipelining），
    // 服务端必须把两个都答出来（第二个请求的字节可能跟第一个请求同批到达）
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        std::string two = post_request(true) + post_request(true);
        send(fd, two.data(), two.size(), MSG_NOSIGNAL);
        // 两个响应必须都能读出来。注意必须用**同一个**累积缓冲：第一个响应的
        // 两次 recv 可能一次把两个响应都读进缓冲区，残留部分属于第二个响应
        std::string stream;
        int got = 0;
        for (int i = 0; i < 2; ++i) {
          size_t consumed = 0;
          // 注意：find(...) 返回的是下标，命中在开头时是 0（假值），
          // 不能直接当布尔用——这里显式与 npos 比较。
          // 读超时给宽一些：Debug 构建 + 机器被压满时，5s 会把"服务端稍慢"
          // 误判成"复用没生效"
          if (read_one_response(fd, stream, 15000, &consumed) &&
              stream.find("200 OK") != std::string::npos)
            ++got;
          stream.erase(0, consumed);
        }
        CHECK(got == 2);
        ok++;
        close(fd);
      }
    }

    stop_flag.store(true);
    t.join();
    server.drain();
  }

  // ── 4. 空闲超时对复用的空闲连接同样生效 ──────────────────────────────
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 2;  // 秒级可判定
    sc.max_connections = 64;

    std::atomic<bool> stop_flag{false};
    HttpServer server(sc);
    server.set_handler([](const std::string& , ResponseWriter&, HttpRequestInfo&) {
      return HttpReply{200, "application/json", R"({"ok":true})"};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    int fd = connect_to(port);
    CHECK(fd >= 0);
    ok++;
    if (fd >= 0) {
      std::string req = post_request(true);
      send(fd, req.data(), req.size(), MSG_NOSIGNAL);
      std::string resp;
      CHECK(read_one_response(fd, resp));
      ok++;
      CHECK(resp.find("Connection: keep-alive") != std::string::npos);
      ok++;
      // 之后不再发任何东西：连接必须在 ~2s 空闲超时后被服务端关闭
      auto t0 = std::chrono::steady_clock::now();
      bool closed = false;
      while (std::chrono::steady_clock::now() - t0 < 6s) {
        char c = 0;
        ssize_t n = recv(fd, &c, 1, MSG_DONTWAIT);
        if (n == 0) {
          closed = true;
          break;
        }
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) break;
        std::this_thread::sleep_for(100ms);
      }
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      std::fprintf(stderr, "[keep-alive] 空闲连接被服务端关闭耗时 %lldms\n",
                   static_cast<long long>(elapsed_ms));
      CHECK(closed);
      ok++;
      // 必须真的是"超时关"，不能是立刻关（那样等于没复用）
      CHECK(elapsed_ms >= 1500);
      ok++;
      close(fd);
    }
    CHECK(server.active_connections() == 0);
    ok++;

    stop_flag.store(true);
    t.join();
    server.drain();
  }

  // ── 7. 并发连接下的借出/归还协议：8 条连接同时处理时，每条连接都必须拿到
  //      属于自己的那个响应（本轮内存破坏崩溃的回归判据）──────────────────
  //   (a) 4 条线程每条请求都用新连接（close 语义），覆盖"借出后立即归还"
  //   (b) 4 条线程各自复用一条连接，覆盖"归还后重新登记再借出"
  //
  // 判别力（两条独立的缺陷都能被这一段抓住）：
  //   - 把令牌发号改回"每条连接各自从 1 开始计数"（修复前的写法）：8 条并发连接的
  //     第 1 次借出都登记成令牌 1，而 borrowed_tokens_ 是按值去重的集合，于是只有
  //     1 个 worker 认领成功，其余任务被丢弃且不归还所有权 —— 那些连接的
  //     worker_owned 永远为真，reactor 既不读也不关，客户端只能读超时，
  //     下面的 answered == 总数 断言失败（实测 200 请求并发 8 丢 15 个）
  //   - 把连接表改回 `deque<pair<int,Connection>>` + `unordered_map<int,
  //     deque::iterator>`：deque::erase 会搬移元素，索引里的迭代器跟着失效/错位，
  //     随后 close_connection 用陈旧迭代器 erase 会在 deque 有效区间之外读写
  //     （glibc: "corrupted double-linked list" / "double free detected"），
  //     这一段会崩在这里而不是给出响应
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 30;  // 足够长：连接只能因为响应结束而关闭
    sc.max_connections = 64;

    constexpr int kThreads = 8;
    constexpr int kPerThread = 15;  // 合计 120 个请求
    constexpr int kHandlerMs = 25;  // 让 worker 慢一点，制造借出重叠的窗口
    constexpr int kReadTimeoutMs = 8000;

    std::atomic<bool> stop_flag{false};
    std::atomic<int> hits{0};
    HttpServer server(sc);
    // 回显请求体：这样"响应属于哪条连接"可以被客户端自己核对
    // （连接状态错位时客户端会收到别人的请求体，而不是超时）
    server.set_handler([&hits, kHandlerMs](const std::string& body, ResponseWriter&,
                                          HttpRequestInfo&) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kHandlerMs));
      ++hits;
      return HttpReply{200, "application/json", body};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    auto make_request = [](const std::string& tag, bool keep_alive) {
      std::string body = R"({"model":"m","messages":[],"tag":")" + tag + R"("})";
      std::string r = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                      "Content-Type: application/json\r\nContent-Length: " +
                      std::to_string(body.size()) + "\r\n";
      r += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
      r += "\r\n";
      r += body;
      return r;
    };

    std::atomic<int> answered{0};
    std::atomic<int> wrong_body{0};
    std::atomic<int> unreadable{0};
    std::vector<std::thread> clients;
    for (int id = 0; id < kThreads; ++id) {
      const bool reuse = id >= kThreads / 2;  // 后半数线程复用同一条连接
      clients.emplace_back([&, id, reuse] {
        int fd = -1;
        for (int i = 0; i < kPerThread; ++i) {
          const std::string tag =
              "c" + std::to_string(id) + "-" + std::to_string(i);
          if (fd < 0) {
            fd = connect_to(port);
            if (fd < 0) {
              ++unreadable;
              return;
            }
          }
          std::string req = make_request(tag, reuse);
          if (send(fd, req.data(), req.size(), MSG_NOSIGNAL) < 0) {
            ++unreadable;
            return;
          }
          std::string resp;
          if (!read_one_response(fd, resp, kReadTimeoutMs) ||
              resp.find("200 OK") == std::string::npos) {
            // 服务端没答（或答了别的）——正是"任务被丢弃、所有权泄漏"的表现
            ++unreadable;
            close(fd);
            return;
          }
          if (resp.find(tag) == std::string::npos) ++wrong_body;
          ++answered;
          if (!reuse) {
            close(fd);
            fd = -1;
          }
        }
        if (fd >= 0) close(fd);
      });
    }
    for (auto& c : clients) c.join();

    std::fprintf(stderr,
                 "[keep-alive] 并发借出：answered=%d/%d hits=%d 无响应=%d "
                 "响应串线=%d accept=%llu\n",
                 answered.load(), kThreads * kPerThread, hits.load(),
                 unreadable.load(), wrong_body.load(),
                 (unsigned long long)server.accepted_connections());
    CHECK(answered.load() == kThreads * kPerThread);
    ok++;
    CHECK(unreadable.load() == 0);  // 没有请求被静默丢弃
    ok++;
    CHECK(wrong_body.load() == 0);  // 没有响应被写到别的连接上
    ok++;
    CHECK(hits.load() == kThreads * kPerThread);
    ok++;

    stop_flag.store(true);
    t.join();
    server.drain();
  }

  return test_check::finish("test_keep_alive", ok);
}
