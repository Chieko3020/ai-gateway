// SSE 流式透传回归（本轮新增：stream:true 由 400 拒绝改为真透传）
//
// 覆盖点：
//   1. MessageFilter::sse_feed —— 按 SSE 事件边界逐条过滤
//      （含"事件跨 chunk 边界"与"最后一段没有空行结尾"两种真实上游形态），
//      命中 URL 的那一条事件被丢弃并判定 reject，其余事件原样放行
//   2. ResponseWriter —— 流式响应头无 Content-Length、有 Transfer-Encoding: chunked，
//      Content-Type 按上游原样透传；每块正文被正确分帧；结束块是 0\r\n\r\n
//   3. HttpServer 端到端：handler 边收边发时，
//      - 客户端**逐块**收到（而不是等整段凑齐才一次性到达）
//      - Content-Type: text/event-stream
//      - 最后一个事件 data: [DONE] 完整
//      - 客户端中途断开后服务端连接被回收、没有 fd 泄漏
//      - 流式响应结束后连接可以 keep-alive 复用
//
// 判别力说明：
//   - 把 sse_feed 换回"对整段做 check_output + 正则替换"：第 1 段的
//     "只丢被拒事件、其余照常"断言失败（整段会被判 reject，前后事件一起丢）
//   - 把 write_stream_head 改成 write_head(kChunkedLength)：第 2/3 段会看到
//     Content-Length: 18446744073709551615，分帧断言也失败
//   - 把 on_chunk 改成先缓冲整段再写：第 3 段的"逐块到达"断言失败
#include <arpa/inet.h>
#include <fcntl.h>
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

#include "server/filter.h"
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

// 带超时的阻塞读：返回 >0 读到字节, 0 = 超时, -1 = 对端关闭
ssize_t recv_timeout(int fd, void* buf, size_t len, int timeout_ms) {
  pollfd p{fd, POLLIN, 0};
  int rc = ::poll(&p, 1, timeout_ms);
  if (rc <= 0) return rc == 0 ? 0 : -1;
  return ::recv(fd, buf, len, 0);
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

// 造一段标准 SSE：若干 data 事件 + 结束事件
std::string sse_event(const std::string& payload) {
  return "data: " + payload + "\n\n";
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1. 按事件边界过滤 ────────────────────────────────────────────────
  {
    FilterConfig fc;
    fc.block_urls = true;
    MessageFilter filter(fc);

    // 1.1 三条事件，其中中间一条含 URL：只丢它，前后照常
    {
      MessageFilter::SseFilterState st;
      std::string in = sse_event(R"({"a":1})") +
                       sse_event(R"({"b":"see https://evil.example/x"})") +
                       sse_event(R"({"c":3})");
      // 一次给全（真实上游也会一次给多块，这里先测整段一次喂）
      std::string out = filter.sse_feed(st, in, /*final_chunk=*/true);
      CHECK(st.rejected);                       // 命中 URL -> reject
      ok++;
      CHECK(out.find(R"({"a":1})") != std::string::npos);   // 前面的事件已放行
      ok++;
      CHECK(out.find(R"({"c":3})") == std::string::npos);   // 后面的不再放行
      ok++;
      CHECK(!st.rejected_event.empty());
      ok++;
      CHECK(st.rejected_event.find("evil.example") != std::string::npos);
      ok++;
      // 关键：不能整段替换成 {"error":...}——放行部分的字节必须原样保留
      CHECK(out.find("data: ") != std::string::npos);
      ok++;
    }

    // 1.2 事件跨 chunk 边界：URL 被拆成两块，仍必须被识别（不能因为分块而漏检）
    {
      MessageFilter::SseFilterState st;
      std::string e1 = sse_event(R"({"b":"https://ev";)");
      // 手工切在 "https://ev" 中间
      std::string first = e1.substr(0, e1.find("ev") + 1);
      std::string rest = e1.substr(first.size());
      std::string out1 = filter.sse_feed(st, first, false);
      CHECK(out1.empty());  // 事件还没结束，一个字节都不放行（避免放行半条被拒事件）
      ok++;
      std::string out2 = filter.sse_feed(st, rest, false);
      CHECK(st.rejected);
      ok++;
      CHECK(out2.empty());
      ok++;
    }

    // 1.3 正常流（无 URL）：全部放行，且放行内容与输入逐字节相同
    {
      MessageFilter::SseFilterState st;
      std::string in = sse_event(R"({"delta":"你好"})") + sse_event("[DONE]");
      // 分三块喂，模拟网络分包
      size_t a = in.size() / 3, b = 2 * in.size() / 3;
      std::string out;
      out += filter.sse_feed(st, in.substr(0, a), false);
      out += filter.sse_feed(st, in.substr(a, b - a), false);
      out += filter.sse_feed(st, in.substr(b), false);
      out += filter.sse_feed(st, {}, true);
      CHECK(!st.rejected);
      ok++;
      CHECK(out == in);  // 逐字节透传
      ok++;
    }

    // 1.4 最后一段没有空行结尾（上游断流/非标准实现）：final_chunk 时必须处理
    {
      MessageFilter::SseFilterState st;
      std::string out = filter.sse_feed(st, "data: {\"x\":1}", false);
      CHECK(out.empty());  // 还不是完整事件，先攒着
      ok++;
      out += filter.sse_feed(st, "", true);
      CHECK(out == "data: {\"x\":1}");
      ok++;
      CHECK(!st.rejected);
      ok++;
    }

    // 1.5 block_urls=false 时不拦任何事件（过滤开关必须真的生效）
    {
      FilterConfig off;
      off.block_urls = false;
      MessageFilter relaxed(off);
      MessageFilter::SseFilterState st;
      std::string in = sse_event(R"({"b":"https://ok.example"})");
      std::string out = relaxed.sse_feed(st, in, true);
      CHECK(!st.rejected);
      ok++;
      CHECK(out == in);
      ok++;
    }

    // 1.6 输出截断（max_output_chars > 0）：上限必须**粘性**，且截断要补发终止事件。
    //     旧实现的两处缺陷：① 触发上限后每个后续 chunk 仍会泄漏它的第一个事件
    //     （上限形同虚设）；② 同一 chunk 里跟在这条事件后面的字节被 pending.clear()
    //     吞掉，真实上游的 `data: [DONE]` 正是这样丢的 —— 客户端拿到一条永不结束
    //     的流（keep-alive 下连接不关，只能等读超时）。
    {
      auto count_of = [](const std::string& s, std::string_view sub) {
        size_t n = 0, pos = 0;
        while ((pos = s.find(sub, pos)) != std::string::npos) {
          ++n;
          pos += sub.size();
        }
        return n;
      };

      FilterConfig fc;
      fc.max_output_chars = 40;  // 每条事件 25 字节，第二条后越界
      MessageFilter trunc(fc);
      MessageFilter::SseFilterState st;
      std::string e1 = sse_event(R"({"delta":"aaaa"})");
      std::string e2 = sse_event(R"({"delta":"bbbb"})");
      std::string e3 = std::string(kSseDoneEvent);  // 上游自己的终止事件
      std::string out = trunc.sse_feed(st, e1 + e2 + e3, false);

      CHECK(st.truncated);
      ok++;
      CHECK(out.find("aaaa") != std::string::npos);  // 上限内的事件照常放行
      ok++;
      CHECK(out.find("bbbb") != std::string::npos);  // 触发上限的那条也放行
      ok++;
      CHECK(out.ends_with(std::string(kSseDoneEvent)));  // 必须补发流结束标志
      ok++;
      // 上游那个 [DONE] 与补发的那个不能都出现（客户端只应看到一次流结束）
      CHECK(count_of(out, kSseDoneEvent) == 1);
      ok++;

      // 粘性：已达上限后后续 chunk 一个字节都不放行（旧实现这里会漏出 e4）
      std::string out2 = trunc.sse_feed(st, sse_event(R"({"delta":"dddd"})"), false);
      CHECK(out2.empty());
      ok++;
      std::string out3 = trunc.sse_feed(st, {}, /*final_chunk=*/true);
      CHECK(out3.empty());
      ok++;
    }
  }

  // ── 2. ResponseWriter 的流式分帧 ─────────────────────────────────────
  {
    int sp[2] = {-1, -1};
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sp) == 0);
    ok++;
    if (sp[0] >= 0) {
      int fl = fcntl(sp[0], F_GETFL, 0);
      fcntl(sp[0], F_SETFL, fl | O_NONBLOCK);
      std::atomic<uint64_t> eagain{0};
      auto deadline = std::chrono::steady_clock::now() + 5s;
      ResponseWriter w(sp[0], 5000, deadline, &eagain);
      // 读端也设成非阻塞：只读到"当前可读的都读完"为止，不依赖 EOF
      // （sp[0] 还开着，socketpair 上不会有 EOF，阻塞读会永久挂住）
      int rfl = fcntl(sp[1], F_GETFL, 0);
      fcntl(sp[1], F_SETFL, rfl | O_NONBLOCK);

      CHECK(w.write_stream_head(200, "text/event-stream", true));
      ok++;
      CHECK(w.committed());
      ok++;
      CHECK(w.chunked());
      ok++;
      CHECK(w.write_body(sse_event(R"({"a":1})")));
      ok++;
      CHECK(w.write_body(sse_event("[DONE]")));
      ok++;
      CHECK(w.finish_stream());
      ok++;

      std::string got;
      char buf[4096];
      while (true) {
        ssize_t n = recv(sp[1], buf, sizeof(buf), 0);
        if (n > 0) {
          got.append(buf, static_cast<size_t>(n));
          continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        if (n <= 0) break;
      }
      CHECK(got.find("HTTP/1.1 200 OK\r\n") == 0);
      ok++;
      CHECK(got.find("Content-Type: text/event-stream\r\n") != std::string::npos);
      ok++;
      CHECK(got.find("Transfer-Encoding: chunked\r\n") != std::string::npos);
      ok++;
      CHECK(got.find("Content-Length") == std::string::npos);
      ok++;
      // 分帧结构：头之后第一个 chunk 的长度字段 = 第一条事件的十六进制长度
      auto hdr_end = got.find("\r\n\r\n");
      CHECK(hdr_end != std::string::npos);
      ok++;
      std::string first_event = sse_event(R"({"a":1})");
      char lenbuf[32];
      std::snprintf(lenbuf, sizeof(lenbuf), "%zx", first_event.size());
      std::string expect_frame = std::string(lenbuf) + "\r\n" + first_event + "\r\n";
      CHECK(got.compare(hdr_end + 4, expect_frame.size(), expect_frame) == 0);
      ok++;
      // 结束块
      CHECK(got.size() >= 5 && got.compare(got.size() - 5, 5, "0\r\n\r\n") == 0);
      ok++;
      // [DONE] 完整存在于流里
      CHECK(got.find("data: [DONE]\n\n") != std::string::npos);
      ok++;

      close(sp[0]);
      close(sp[1]);
    }
  }

  // ── 3. 端到端：逐块到达 + Content-Type + [DONE] + 断开回收 + keep-alive ──
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 30;
    sc.max_connections = 64;
    sc.write_timeout_seconds = 10;

    std::atomic<bool> stop_flag{false};
    std::atomic<int> streams_started{0};
    std::atomic<int> streams_finished{0};
    // 假后端：分 5 批发 SSE，每批间隔 120ms（模拟真实上游的逐 token 到达）
    auto fake_sse = [&](ResponseWriter& writer, bool keep_alive) {
      ++streams_started;
      if (!writer.write_stream_head(200, "text/event-stream", keep_alive)) return;
      for (int i = 0; i < 4; ++i) {
        std::string ev = sse_event(std::string("{\"delta\":\"chunk") +
                                   std::to_string(i) + "\"}");
        if (!writer.write_body(ev)) return;
        std::this_thread::sleep_for(120ms);
      }
      if (!writer.write_body(sse_event("[DONE]"))) return;
      if (!writer.finish_stream()) return;
      ++streams_finished;
    };

    HttpServer server(sc);
    server.set_handler([](const std::string& , ResponseWriter&, HttpRequestInfo&) {
      return HttpReply{200, "application/json", "{}"};
    });
    server.add_route("/sse", [&fake_sse](const std::string&, ResponseWriter& w,
                                        HttpRequestInfo& info) {
      // 与生产代码同一套规则：响应头的 Connection 如实反映客户端意愿
      fake_sse(w, info.keep_alive);
      return HttpReply{200, "text/event-stream", {}};  // 已自行写完，body 留空
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    // 3.1 逐块到达：记录每次 recv 的时刻与累计字节，相邻两块的到达时间差应显著
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        const char* req =
            "POST /sse HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}";
        send(fd, req, std::strlen(req), MSG_NOSIGNAL);

        std::vector<std::pair<double, size_t>> arrivals;  // (ms, 累计字节)
        auto t0 = std::chrono::steady_clock::now();
        std::string resp;
        char buf[4096];
        // keep-alive 下客户端不会看到 EOF：以 chunked 结束块（0\r\n\r\n）作为读完标记，
        // 并用 2s 静默期兜底，避免用例挂死在"等一个不会来的 EOF"上
        while (true) {
          ssize_t n = recv_timeout(fd, buf, sizeof(buf), 2000);
          if (n > 0) {
            resp.append(buf, static_cast<size_t>(n));
            arrivals.emplace_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0)
                    .count(),
                resp.size());
            if (resp.find("0\r\n\r\n") != std::string::npos) break;
            continue;
          }
          break;  // 超时或对端关闭
        }

        std::fprintf(stderr,
                     "[sse] 收到 %zu 字节，%zu 次 recv，首个 body 片段到达时刻=%.1fms，"
                     "最后到达=%.1fms\n",
                     resp.size(), arrivals.size(),
                     arrivals.empty() ? 0.0 : arrivals.front().first,
                     arrivals.empty() ? 0.0 : arrivals.back().first);

        CHECK(resp.find("HTTP/1.1 200 OK") != std::string::npos);
        ok++;
        CHECK(resp.find("Content-Type: text/event-stream") != std::string::npos);
        ok++;
        CHECK(resp.find("Transfer-Encoding: chunked") != std::string::npos);
        ok++;
        CHECK(resp.find("Content-Length:") == std::string::npos);
        ok++;
        // [DONE] 完整
        CHECK(resp.find("data: [DONE]\n\n") != std::string::npos);
        ok++;
        CHECK(streams_finished.load() == 1);
        ok++;
        // 逐块到达的判别力：假后端每 120ms 一批共 5 批 => 整段至少跨 400ms；
        // 且必须有多次 recv 到达（一次性缓冲的实现在这里只会有 1~2 次且总时长 ~400ms
        // 之后才到）。这里断言"第一块明显早于最后一块"
        CHECK(arrivals.size() >= 3);
        ok++;
        double span = arrivals.back().first - arrivals.front().first;
        std::fprintf(stderr, "[sse] 首末到达跨度=%.1fms（应 >= 300ms）\n", span);
        CHECK(span >= 300.0);
        ok++;
        // 首块必须是响应头（前端要尽快拿到 200 + text/event-stream 才能开始渲染）
        CHECK(arrivals.front().first < 100.0);
        ok++;
        close(fd);
      }
    }

    // 3.2 流式响应可以 keep-alive 复用（同步点见下）：同一条连接上连续两次流式请求。
    // 用一个同步点消除时序竞态：先等服务端把这条连接**重新登记**好
    // （keep_alive_rearms 计数增加），再发第二个请求。否则客户端可能在服务端
    // 把 fd 挂回 epoll 之前就把字节发出去，那一瞬间连接上的字节没有任何 epoll
    // 事件来源，用例会随机失败——这不是产品缺陷，是测试没同步
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        const char* req =
            "POST /sse HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}";
        const uint64_t rearm_before = server.keep_alive_rearms();
        const uint64_t reuse_before = server.reused_connection_count();
        char buf[4096];

        // 请求 1
        send(fd, req, std::strlen(req), MSG_NOSIGNAL);
        std::string resp1;
        while (resp1.find("0\r\n\r\n") == std::string::npos) {
          ssize_t n = recv_timeout(fd, buf, sizeof(buf), 3000);
          if (n > 0) {
            resp1.append(buf, static_cast<size_t>(n));
            continue;
          }
          break;
        }
        CHECK(resp1.find("data: [DONE]") != std::string::npos);
        ok++;
        CHECK(resp1.find("Connection: keep-alive") != std::string::npos);
        ok++;

        // 同步点：等服务端确实把连接重新登记回来
        CHECK(wait_for([&] { return server.keep_alive_rearms() > rearm_before; },
                       3000));
        ok++;

        // 请求 2：同一条连接
        send(fd, req, std::strlen(req), MSG_NOSIGNAL);
        std::string resp2;
        while (resp2.find("0\r\n\r\n") == std::string::npos) {
          ssize_t n = recv_timeout(fd, buf, sizeof(buf), 3000);
          if (n > 0) {
            resp2.append(buf, static_cast<size_t>(n));
            continue;
          }
          break;
        }
        std::fprintf(stderr,
                     "[sse-keepalive] resp1=%zu resp2=%zu done2=%d reuse %llu -> %llu\n",
                     resp1.size(), resp2.size(),
                     resp2.find("data: [DONE]") != std::string::npos ? 1 : 0,
                     (unsigned long long)reuse_before,
                     (unsigned long long)server.reused_connection_count());
        // 第二个请求在**同一条连接**上得到了完整响应
        CHECK(resp2.find("data: [DONE]") != std::string::npos);
        ok++;
        // 复用计数证明这是同一条连接上的第二个请求，而不是新连接的第一次。
        // 计数在 worker 归还之后才递增（客户端可能已经读完响应），因此这里必须
        // 等一拍——直接断言会随机失败（这是测试的时序问题，不是产品缺陷）
        CHECK(wait_for(
            [&] { return server.reused_connection_count() > reuse_before; }, 2000));
        ok++;
        close(fd);
      }
    }

    // 3.3 客户端中途断开：服务端不泄漏连接
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        const char* req =
            "POST /sse HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}";
        send(fd, req, std::strlen(req), MSG_NOSIGNAL);
        std::this_thread::sleep_for(150ms);  // 只读一点点就断开
        char buf[64];
        recv(fd, buf, sizeof(buf), 0);
        close(fd);  // RST/FIN：服务端下一次写必然失败
        int before = streams_started.load();
        CHECK(before >= 2);
        ok++;
        // 连接必须在数秒内被回收（判别力：worker 若在断开后仍占着 fd 不放，
        // active_connections 不会回落）
        CHECK(wait_for([&] { return server.active_connections() <= 1; }, 5000));
        ok++;
      }
    }

    // 3.4 流式响应必须遵守客户端的 Connection: close（不能只顾着复用）
    {
      int fd = connect_to(port);
      CHECK(fd >= 0);
      ok++;
      if (fd >= 0) {
        const char* req =
            "POST /sse HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n"
            "Connection: close\r\n\r\n{}";
        send(fd, req, std::strlen(req), MSG_NOSIGNAL);
        std::string resp;
        char buf[4096];
        auto deadline = std::chrono::steady_clock::now() + 5s;
        bool eof = false;
        while (std::chrono::steady_clock::now() < deadline) {
          ssize_t n = recv_timeout(fd, buf, sizeof(buf), 2000);
          if (n > 0) {
            resp.append(buf, static_cast<size_t>(n));
            continue;
          }
          if (n < 0) break;
          // n == 0：静默期；只有拿到终止块之后再判定
          if (resp.find("0\r\n\r\n") != std::string::npos) {
            eof = true;  // 读到终止块即认为流已结束（随后服务端关闭连接）
            break;
          }
          break;
        }
        CHECK(resp.find("Connection: close") != std::string::npos);
        ok++;
        CHECK(resp.find("Transfer-Encoding: chunked") != std::string::npos);
        ok++;
        CHECK(resp.find("data: [DONE]") != std::string::npos);
        ok++;
        CHECK(eof);
        ok++;
        close(fd);
      }
    }

    CHECK(streams_started.load() >= 4);
    ok++;

    stop_flag.store(true);
    t.join();
    server.drain();
  }

  return test_check::finish("test_sse_stream", ok);
}
