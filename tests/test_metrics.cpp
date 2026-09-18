// /metrics 端点回归（报告 8.7 第 4 条）
//
// 覆盖点：
//   1. 渲染出的文本符合 Prometheus 暴露格式：每个指标有唯一 HELP/TYPE，值可解析
//   2. Stats 的字段都能在文本里找到，且口径与 report() 一致
//      （requests 不含旁路/合并、merged 单列、命中率 0~1）
//   3. 新增 GET /metrics 不破坏既有 POST /v1/chat/completions
//   4. 方法语义不变：没有对应 method 路由的非 POST -> 405，POST 到未知路径 -> 404
//
// 判别力说明：
//   - 少了 status_text(200)/response 构造时的 405/404 分支，第 3/4 段断言失败；
//   - 把 connection_handler 的"非 POST 一律 405"改回去（本项修复前就是那样），
//     GET /metrics 会拿到 405 而不是 200 + 指标文本，第 3 段失败；
//   - 把 render_metrics 的 quantile 标签去掉后，"quantile=\"0.5\"" 断言失败。
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "server/http_server.h"
#include "server/metrics.h"
#include "stats/stats.h"
#include "test_check.h"

using namespace ai_gateway;
using namespace std::chrono_literals;

namespace {

size_t count_of(const std::string& hay, const std::string& needle) {
  size_t n = 0;
  for (size_t pos = hay.find(needle); pos != std::string::npos;
       pos = hay.find(needle, pos + needle.size()))
    ++n;
  return n;
}

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

// 发一条请求并读完整响应（服务端恒 Connection: close）
std::string roundtrip(int port, const std::string& request) {
  int fd = connect_to(port);
  if (fd < 0) return "";
  if (send(fd, request.data(), request.size(), MSG_NOSIGNAL) <= 0) {
    close(fd);
    return "";
  }
  std::string resp;
  char buf[4096];
  while (true) {
    ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n > 0) {
      resp.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) break;
    if (errno == EINTR) continue;
    break;
  }
  close(fd);
  return resp;
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

std::string status_line(const std::string& resp) {
  auto nl = resp.find("\r\n");
  return nl == std::string::npos ? resp : resp.substr(0, nl);
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1/2. 文本格式与字段 ────────────────────────────────────────────
  {
    Stats st;
    st.set_pricing(TokenPricing{0.001, 0.004});
    st.record_api_call(20, 1000, 2000);  // 未命中
    st.record_cache_hit(5);              // 命中
    st.record_cache_hit(4);              // 命中
    st.record_bypass(300, 100, 50);      // 旁路（不进命中率）
    st.record_merge(6);                  // 合并（不进 hits/total）

    const std::string m = render_metrics(st, /*uptime*/ 42, /*pending*/ 1,
                                         /*active*/ 2, /*threads*/ 4);
    CHECK(!m.empty());
    ok++;

    // 关键字段存在且取值与统计口径一致
    CHECK(m.find("ai_gateway_requests_total 3") != std::string::npos);
    ok++;  // hits(2) + misses(1)，不含旁路与合并
    CHECK(m.find("ai_gateway_cache_hits_total 2") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_cache_misses_total 1") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_cache_bypassed_total 1") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_cache_merged_total 1") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_tokens_prompt_total 1100") != std::string::npos);
    ok++;  // 1000 + 旁路 100
    CHECK(m.find("ai_gateway_tokens_completion_total 2050") != std::string::npos);
    ok++;  // 2000 + 旁路 50
    CHECK(m.find("ai_gateway_price_input_per_1k_yuan 0.001") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_price_output_per_1k_yuan 0.004") != std::string::npos);
    ok++;
    // 命中率 = 2/3 = 0.666667（0~1，不是百分数）
    CHECK(m.find("ai_gateway_cache_hit_ratio 0.666667") != std::string::npos);
    ok++;

    // 分位数用 quantile 标签
    CHECK(m.find("ai_gateway_latency_milliseconds{quantile=\"0.5\"}") !=
          std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_latency_milliseconds{quantile=\"0.95\"}") !=
          std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_latency_milliseconds{quantile=\"0.99\"}") !=
          std::string::npos);
    ok++;
    // 旁路延迟池独立，p50 是 300ms 而不是主池的 5ms 量级
    CHECK(m.find("ai_gateway_bypass_latency_milliseconds{quantile=\"0.5\"} 300") !=
          std::string::npos);
    ok++;

    // 线程池与 uptime
    CHECK(m.find("ai_gateway_thread_pool_pending_tasks 1") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_thread_pool_active_tasks 2") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_thread_pool_size 4") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_uptime_seconds 42") != std::string::npos);
    ok++;
    CHECK(m.find("ai_gateway_up 1") != std::string::npos);
    ok++;

    // 每个指标名只能有一组 HELP/TYPE（重复定义会被抓取端判为格式错误）；
    // 用 "latency_milliseconds"（3 个分位数样本共用一组）验证去重生效
    CHECK(count_of(m, "# TYPE ai_gateway_latency_milliseconds gauge") == 1);
    ok++;
    CHECK(count_of(m, "ai_gateway_latency_milliseconds{") == 3);
    ok++;

    // 不可用字段不上报（避免假数据）
    const std::string m2 = render_metrics(st);
    CHECK(m2.find("ai_gateway_thread_pool_pending_tasks") == std::string::npos);
    ok++;
    CHECK(m2.find("ai_gateway_uptime_seconds") == std::string::npos);
    ok++;
    // 每行都必须是 "# HELP/# TYPE/样本" 之一，且无 \r（Prometheus 用 \n）
    size_t bad = 0;
    size_t lines = 0;
    for (size_t pos = 0; pos < m.size();) {
      auto nl = m.find('\n', pos);
      std::string line = m.substr(pos, nl - pos);
      ++lines;
      if (m.find('\r') != std::string::npos) ++bad;
      if (!line.empty() && line[0] != '#' &&
          line.find(' ') == std::string::npos)
        ++bad;
      if (nl == std::string::npos) break;
      pos = nl + 1;
    }
    CHECK(bad == 0 && lines > 20);
    ok++;
  }

  // ── 3/4. 端到端：GET /metrics 可用，POST 行为不变 ───────────────────
  {
    ServerConfig sc;
    sc.port = 0;
    sc.idle_timeout_seconds = 30;
    sc.max_connections = 32;

    std::atomic<bool> stop_flag{false};
    std::atomic<int> post_hits{0};
    Stats st;
    st.record_api_call(11, 10, 20);

    HttpServer server(sc);
    server.set_handler([&post_hits](const std::string&) {
      ++post_hits;
      return HttpReply{200, "application/json", R"({"ok":true})"};
    });
    server.add_route("GET", "/metrics", [&st](const std::string&) {
      return HttpReply{200, kMetricsContentType, render_metrics(st)};
    });

    std::thread t([&] { server.run(&stop_flag); });
    CHECK(wait_listening(server, 5000));
    ok++;
    const int port = server.listen_port();

    auto get_metrics = roundtrip(
        port, "GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    CHECK(status_line(get_metrics).find("200 OK") != std::string::npos);
    ok++;
    // 判别力：修复前"非 POST 一律 405"，这里会是 405 Method Not Allowed
    CHECK(status_line(get_metrics).find("405") == std::string::npos);
    ok++;
    CHECK(get_metrics.find("text/plain; version=0.0.4") != std::string::npos);
    ok++;
    CHECK(get_metrics.find("ai_gateway_requests_total 1") != std::string::npos);
    ok++;

    // 既有 POST 路由不受影响
    auto post = roundtrip(port,
                          "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n"
                          "Content-Type: application/json\r\nContent-Length: 2\r\n"
                          "Connection: close\r\n\r\n{}");
    CHECK(status_line(post).find("200 OK") != std::string::npos);
    ok++;
    CHECK(post_hits.load() == 1);
    ok++;

    // 未注册的 GET 路径：保持旧语义 405（而不是 404）
    auto get_other = roundtrip(port,
                               "GET /nope HTTP/1.1\r\nHost: x\r\n"
                               "Connection: close\r\n\r\n");
    CHECK(status_line(get_other).find("405") != std::string::npos);
    ok++;
    // POST 到未注册路径：保持旧语义 404
    auto post_other = roundtrip(port,
                                "POST /nope HTTP/1.1\r\nHost: x\r\n"
                                "Content-Length: 2\r\nConnection: close\r\n\r\n{}");
    CHECK(status_line(post_other).find("404") != std::string::npos);
    ok++;

    stop_flag.store(true);
    t.join();
    server.drain();
  }

  return test_check::finish("test_metrics", ok);
}
