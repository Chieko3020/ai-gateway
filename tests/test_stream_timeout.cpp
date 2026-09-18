// 流式响应的写死线语义回归（目标 5：长流不被切断）
//
// 缺陷：`write_timeout_seconds`（默认 10s）此前是**整条响应**的总死线。对流式响应
// 来说这等于"回答超过 10s 就被拦腰切断"——生成 4K token 的回答轻易超过 60s。
//
// 修复后的语义：
//   缓冲式（write_head）    ：整条响应受 total_deadline 约束（防慢客户端占住 worker）
//   流式（write_stream_head）：**不设总死线**，改为"两次成功写入之间的空闲超时"，
//                              每次写出数据后续期
//
// 本用例全部走 socketpair，不依赖 TCP 时序（与 test_http_server 第 6 段同一手法）：
//   1. 流式 + 总死线已过期 + 每次写入间隔 < 空闲死线 → **必须全部写成功**
//      （这是"长流不被切断"的最小复现：旧实现在第一次写入后就放弃了）
//   2. 流式 + 客户端停顿 > 空闲死线 → 中停，abort_reason == kIdle
//   3. 缓冲式 + 超大响应 + 不读的客户端 → 仍受总死线约束，abort_reason == kDeadline
//      （证明"为流式放开总死线"没有把缓冲式的自我保护一起放开）
//   4. 空闲死线会在每次成功写入后续期（否则第 1 段的长流总时长 > 空闲死线 也会挂）
//
// 判别力说明（回退修复即失败）：
//   - 把 current_total_deadline() 改回"流式也用 total_deadline_"：第 1 段挂
//     （旧行为：总死线 300ms 到点，第 2 次写入就 abort）
//   - 把 send_raw/flush 里的续期去掉：第 1 段在累计时长超过空闲死线后挂
//   - 把缓冲式的 total_deadline_ 忽略（一律 max）：第 3 段挂
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "server/http_server.h"
#include "test_check.h"

using namespace ai_gateway;
using namespace std::chrono_literals;

namespace {

int make_socketpair(int sp[2], int sndbuf) {
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sp) != 0) return -1;
  if (sndbuf > 0) setsockopt(sp[0], SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
  int flags = fcntl(sp[0], F_GETFL, 0);
  if (flags < 0) return -1;
  if (fcntl(sp[0], F_SETFL, flags | O_NONBLOCK) < 0) return -1;
  return 0;
}

// 在对端持续 recv（别让它把发送缓冲堵满），返回收到的字节数。
// stop 置位后线程退出（不关闭 fd，避免写端看到 EPIPE 而非死线）
std::thread start_reader(int fd, std::atomic<bool>* stop, std::atomic<size_t>* got,
                         std::chrono::milliseconds lag = 0ms) {
  return std::thread([fd, stop, got, lag] {
    std::vector<char> buf(65536);
    std::string keep;  // 只用于累计计数，不保留内容
    while (!stop->load()) {
      ssize_t n = recv(fd, buf.data(), buf.size(), 0);
      if (n > 0) {
        got->fetch_add(static_cast<size_t>(n));
        continue;
      }
      std::this_thread::sleep_for(lag.count() > 0 ? lag : 1ms);
    }
  });
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1. 长流：总死线已过期，但每次写入都在空闲死线内 → 不得被切断 ────
  {
    int sp[2] = {-1, -1};
    CHECK(make_socketpair(sp, 0) == 0); ok++;
    if (sp[0] >= 0) {
      std::atomic<bool> stop{false};
      std::atomic<size_t> got{0};
      auto reader = start_reader(sp[1], &stop, &got);

      std::atomic<uint64_t> eagain{0};
      // 总死线：故意给 300ms（旧实现下这就是"这条流只能活 300ms"）
      const auto total_deadline =
          std::chrono::steady_clock::now() + 300ms;
      // 空闲死线 200ms，每次写入间隔 400ms、共 6 块：总时长约 2.4s
      //   - 远超总死线 300ms（旧实现下第 2 块就失败）
      //   - 每次间隔 400ms 都超过空闲死线 200ms，但**每次写入本身不阻塞**
      //     （对端一直在 recv，64KB 远小于发送缓冲），因此不触发空闲死线。
      //     这正是"空闲死线 ≠ 总死线"的语义：有进展就不算超时
      ResponseWriter w(sp[0], /*write_deadline_ms=*/200, total_deadline, &eagain,
                       /*stream_idle_ms=*/200);
      CHECK(w.write_stream_head(200, "text/event-stream", /*keep_alive=*/false));
      ok++;

      constexpr int kChunks = 6;
      const std::string chunk(64 * 1024, 's');
      bool all_ok = true;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < kChunks; ++i) {
        std::this_thread::sleep_for(400ms);
        if (!w.write_body(chunk)) {
          all_ok = false;
          break;
        }
      }
      const bool finish_ok = w.finish_stream();
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      stop.store(true);
      reader.join();

      std::fprintf(stderr,
                   "[stream-idle] 长流(%d 块/400ms 间隔, 总死线 300ms, 空闲 200ms): "
                   "all_ok=%d finish=%d failed=%d abort=%d 用时=%lldms\n",
                   kChunks, all_ok ? 1 : 0, finish_ok ? 1 : 0,
                   w.failed() ? 1 : 0, static_cast<int>(w.abort_reason()),
                   static_cast<long long>(elapsed_ms));

      CHECK(all_ok);      // 每一块都写出去了（旧实现：第 2 块就失败）
      ok++;
      CHECK(finish_ok);   // chunked 终止块也发出去了
      ok++;
      CHECK(!w.failed()); ok++;
      CHECK(w.abort_reason() == AbortReason::kNone); ok++;
      // 总时长确实超过了总死线 300ms 与空闲死线 200ms
      // （证明这段测试真的落在"旧实现会切断"的区间里）
      CHECK(elapsed_ms > 1000); ok++;
      close(sp[0]);
      close(sp[1]);
    }
  }

  // ── 2. 空闲死线到点：客户端停顿超过空闲值 → 中停，原因 kIdle ─────────
  // 注意"空闲"只在**写不进去**时才被检查：客户端一直在正常接收时，写调用立刻
  // 返回成功，不存在"等待"这回事（这正确：有进展就不该超时）。因此本段用
  // 极小发送缓冲 + 不读的对端，先把发送缓冲填满，再停顿超过空闲值，
  // 让下一次写入真的阻塞在 poll 上
  {
    int sp[2] = {-1, -1};
    CHECK(make_socketpair(sp, 4096) == 0); ok++;
    if (sp[0] >= 0) {
      std::atomic<uint64_t> eagain{0};
      ResponseWriter w(sp[0], /*write_deadline_ms=*/200,
                       std::chrono::steady_clock::time_point::max(), &eagain,
                       /*stream_idle_ms=*/200);
      CHECK(w.write_stream_head(200, "text/event-stream", false)); ok++;
      // 先在空闲值内写入，证明"前一次成功"
      CHECK(w.write_body("data: hello\n\n")); ok++;
      std::this_thread::sleep_for(600ms);  // 停顿 600ms（> 空闲 200ms）
      // 再写一大块：会撞 EAGAIN -> poll 等到空闲死线 -> 失败并标记 kIdle
      const bool after_pause = w.write_body(std::string(256 * 1024, 'p'));
      std::fprintf(stderr,
                   "[stream-idle] 停顿 600ms 后写入大块: ok=%d abort=%d(3=kIdle) "
                   "EAGAIN=%llu\n",
                   after_pause ? 1 : 0, static_cast<int>(w.abort_reason()),
                   static_cast<unsigned long long>(eagain.load()));
      CHECK(!after_pause); ok++;
      CHECK(w.failed()); ok++;
      CHECK(w.abort_reason() == AbortReason::kIdle); ok++;
      CHECK(eagain.load() > 0); ok++;
      close(sp[0]);
      close(sp[1]);
    }
  }

  // ── 3. 缓冲式：总死线仍然生效（放开流式不等于放开缓冲式）───────────
  {
    int sp[2] = {-1, -1};
    // 极小发送缓冲 + 不读的对端：写 4MB 必然 EAGAIN 并撞上总死线
    CHECK(make_socketpair(sp, 4096) == 0); ok++;
    if (sp[0] >= 0) {
      std::atomic<uint64_t> eagain{0};
      const auto total_deadline = std::chrono::steady_clock::now() + 300ms;
      ResponseWriter w(sp[0], /*write_deadline_ms=*/5000, total_deadline, &eagain,
                       /*stream_idle_ms=*/5000);
      const std::string big(4 * 1024 * 1024, 'b');
      CHECK(w.write_head(200, "application/json", big.size(), false)); ok++;
      auto t0 = std::chrono::steady_clock::now();
      const bool body_ok = w.write_body(big);  // 头部先发得出去，正文会卡住
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      std::fprintf(stderr,
                   "[stream-idle] 缓冲式大响应(不读对端): body_ok=%d abort=%d(2=kDeadline) "
                   "%lldms EAGAIN=%llu\n",
                   body_ok ? 1 : 0, static_cast<int>(w.abort_reason()),
                   static_cast<long long>(elapsed_ms),
                   static_cast<unsigned long long>(eagain.load()));
      CHECK(!body_ok); ok++;
      CHECK(w.abort_reason() == AbortReason::kDeadline); ok++;
      // 死线 300ms，且必须真的等过（不是立刻放弃）
      CHECK(elapsed_ms >= 250 && elapsed_ms < 3000); ok++;
      CHECK(eagain.load() > 0); ok++;
      close(sp[0]);
      close(sp[1]);
    }
  }

  // ── 4. 续期语义：单次写入都很快、但总时长远超空闲死线 ────────────────
  // 这一段的写端永远可写（对端持续读），因此不会撞 EAGAIN；
  // 真正被验证的是"总时长 > 空闲死线也不失败"这一条（即空闲死线不是总死线）
  {
    int sp[2] = {-1, -1};
    CHECK(make_socketpair(sp, 0) == 0); ok++;
    if (sp[0] >= 0) {
      std::atomic<bool> stop{false};
      std::atomic<size_t> got{0};
      auto reader = start_reader(sp[1], &stop, &got);
      std::atomic<uint64_t> eagain{0};
      ResponseWriter w(sp[0], 600, std::chrono::steady_clock::time_point::max(),
                       &eagain, /*stream_idle_ms=*/600);
      CHECK(w.write_stream_head(200, "text/event-stream", false)); ok++;
      bool all_ok = true;
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 6; ++i) {  // 6 × 150ms = 900ms > 空闲死线 600ms
        std::this_thread::sleep_for(150ms);
        if (!w.write_body("data: tick\n\n")) {
          all_ok = false;
          break;
        }
      }
      auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      stop.store(true);
      reader.join();
      std::fprintf(stderr,
                   "[stream-idle] 续期: 总时长 %lldms > 空闲死线 600ms, all_ok=%d\n",
                   static_cast<long long>(elapsed_ms), all_ok ? 1 : 0);
      CHECK(all_ok); ok++;
      CHECK(elapsed_ms > 600); ok++;
      CHECK(!w.failed()); ok++;
      close(sp[0]);
      close(sp[1]);
    }
  }

  return test_check::finish("test_stream_timeout", ok);
}
