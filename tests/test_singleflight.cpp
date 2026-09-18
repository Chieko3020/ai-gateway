// Singleflight 单元测试（报告 C2 回归）
//
// 覆盖点：
//   1. 精确键 / 语义（cosine >= 0.95）合并
//   2. 取消：等待者 wait_for 立即 ready，get() 抛 SingleflightCancelled，
//      且**不得**退化成 broken_promise 的 std::future_error
//   3. leader 在途时等待者按超时返回（不伪造失败）
//   4. 被顶替的旧 leader 迟到的 complete 不得污染新槽位
//   5. 等待者在 worker 线程里捕获后回源（不 terminate）
//
// 所有读取都经过 read()（带超时上限），保证失败时是断言的失败而不是整条测试挂死
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/singleflight.h"
#include "test_check.h"

using namespace ai_gateway;
using namespace std::chrono_literals;

namespace {

enum class Outcome { kValue, kCancelled, kFutureError, kTimeout };

// 在 deadline 内读取 shared_future：超时返回 kTimeout，异常按类型区分
Outcome read(const std::shared_future<std::string>& fut, std::string* value) {
  if (fut.wait_for(500ms) != std::future_status::ready) return Outcome::kTimeout;
  try {
    *value = fut.get();
    return Outcome::kValue;
  } catch (const SingleflightCancelled&) {
    return Outcome::kCancelled;
  } catch (const std::future_error&) {
    return Outcome::kFutureError;  // broken_promise：C2 里会让进程 abort 的那一种
  } catch (...) {
    return Outcome::kFutureError;
  }
}

}  // namespace

int main() {
  int ok = 0;
  std::vector<float> emb(8, 0.5f);
  std::vector<float> other(8, 0.0f);
  other[0] = 1.0f;  // 与 emb 的 cosine ≈ 0.35，远低于合并阈值

  // 1. 精确键合并 + complete 传递结果
  {
    Singleflight sf;
    auto owner = sf.insert("k", emb);
    auto fut = sf.try_merge("k", emb);
    CHECK(fut.has_value()); ok++;
    sf.complete("k", "answer", owner);
    std::string v;
    Outcome out = Outcome::kTimeout;
    if (fut.has_value()) out = read(*fut, &v);
    CHECK(out == Outcome::kValue); ok++;
    CHECK(v == "answer"); ok++;
    CHECK(!sf.try_merge("k", emb).has_value()); ok++;  // 完成后槽位已释放
  }

  // 2. 语义合并：同向量合并，不同向量不合并
  {
    Singleflight sf;
    (void)sf.insert("k1", emb);
    CHECK(sf.try_merge("k2", emb).has_value()); ok++;
    CHECK(!sf.try_merge("k3", other).has_value()); ok++;
  }

  // 3. 取消：显式失败信号，而非 broken_promise
  {
    Singleflight sf;
    auto owner = sf.insert("k", emb);
    auto fut = sf.try_merge("k", emb);
    CHECK(fut.has_value()); ok++;
    sf.cancel("k", owner);
    owner.reset();  // 旧语义下 promise 在此析构并写出 broken_promise
    std::string v;
    Outcome out = Outcome::kTimeout;
    if (fut.has_value()) out = read(*fut, &v);
    CHECK(out == Outcome::kCancelled); ok++;
    CHECK(out != Outcome::kFutureError); ok++;  // C2：不得再是 future_error(broken_promise)
    CHECK(out != Outcome::kTimeout); ok++;
  }

  // 4. leader 在途：等待者按超时返回
  {
    Singleflight sf;
    (void)sf.insert("k", emb);
    auto fut = sf.try_merge("k", emb);
    CHECK(fut.has_value()); ok++;
    CHECK(fut->wait_for(20ms) == std::future_status::timeout); ok++;
  }

  // 5. 被顶替的旧 leader 迟到的 complete 不得写入新槽位
  {
    Singleflight sf;
    auto old_owner = sf.insert("k", emb);  // leader A
    auto fut_a = sf.try_merge("k", emb);   // 等待者合并到 A
    auto new_owner = sf.insert("k", emb);  // A 超时被顶替，B 成为 leader
    sf.complete("k", "A的结果（旧世代，应被丢弃）", old_owner);

    auto fut_b = sf.try_merge("k", emb);   // 槽位应仍属于 B
    CHECK(fut_b.has_value()); ok++;
    if (fut_b.has_value()) {
      sf.complete("k", "B的结果", new_owner);
      std::string vb;
      CHECK(read(*fut_b, &vb) == Outcome::kValue); ok++;
      CHECK(vb == "B的结果"); ok++;
    }
    std::string va;
    Outcome out_a = Outcome::kTimeout;
    if (fut_a.has_value()) out_a = read(*fut_a, &va);
    CHECK(out_a == Outcome::kCancelled); ok++;  // 旧世代被显式取消
    CHECK(out_a != Outcome::kTimeout); ok++;
  }

  // 6. 未知键上的 cancel/complete 不得抛异常（网关线程里抛异常会终止进程）
  {
    Singleflight sf;
    auto ghost = std::make_shared<std::promise<std::string>>();
    sf.cancel("nope", ghost);
    sf.complete("nope", "x", ghost);
    auto owner = sf.insert("k", emb);
    sf.complete("k", "v", owner);
    sf.complete("k", "v", owner);  // 重复 complete：找不到条目，直接返回
    CHECK(true); ok++;
  }

  // 7. 等待者在 worker 线程内 try/catch 后回源（等价 main.cpp 的等待方逻辑）
  {
    Singleflight sf;
    auto owner = sf.insert("k", emb);
    auto fut = sf.try_merge("k", emb);
    CHECK(fut.has_value()); ok++;
    sf.cancel("k", owner);
    owner.reset();  // 旧语义下 promise 在此析构；新语义下已显式 set_exception
    std::string result;
    if (fut.has_value()) {
      std::thread worker([&] {
        try {
          result = "shared:" + fut->get();
        } catch (const std::exception& e) {
          result = std::string("fallback:") + e.what();
        }
      });
      worker.join();
    }
    CHECK(result.starts_with("fallback:")); ok++;
  }

  return test_check::finish("test_singleflight", ok);
}
