// Singleflight 单元测试（报告 C2 回归）
//
// 覆盖点：
//   1. 精确键 / 语义（cosine >= 0.95）合并
//   2. 取消：等待者 wait_for 立即 ready，get() 抛 SingleflightCancelled，
//      且**不得**退化成 broken_promise 的 std::future_error
//   3. leader 在途时等待者按超时返回（不伪造失败）
//   4. 被顶替的旧 leader 迟到的 complete 不得污染新槽位
//   5. 等待者在 worker 线程里捕获后回源（不 terminate）
//   6. 实体一致性否决（第六轮新增）：向量完全相同（cos=1.0）但数字/大写缩略语
//      不一致的在途请求不得合并；实体一致时仍必须合并；一条候选被否决不等于
//      整体放弃（在途表里另一条实体一致的候选仍可合并）
//
// 所有读取都经过 read()（带超时上限），保证失败时是断言的失败而不是整条测试挂死
#include <chrono>
#include <future>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cache/entity_tokens.h"
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
    auto owner = sf.insert("k", emb, EntityTokens{});
    auto fut = sf.try_merge("k", emb, "",
                           EntityTokens{}, false);
    CHECK(fut.has_value()); ok++;
    sf.complete("k", "answer", owner);
    std::string v;
    Outcome out = Outcome::kTimeout;
    if (fut.has_value()) out = read(*fut, &v);
    CHECK(out == Outcome::kValue); ok++;
    CHECK(v == "answer"); ok++;
    CHECK(!sf.try_merge("k", emb, "",
                           EntityTokens{}, false).has_value()); ok++;  // 完成后槽位已释放
  }

  // 2. 语义合并：同向量合并，不同向量不合并
  {
    Singleflight sf;
    (void)sf.insert("k1", emb, EntityTokens{});
    CHECK(sf.try_merge("k2", emb, "",
                           EntityTokens{}, false).has_value()); ok++;
    CHECK(!sf.try_merge("k3", other, "",
                           EntityTokens{}, false).has_value()); ok++;
  }

  // 3. 取消：显式失败信号，而非 broken_promise
  {
    Singleflight sf;
    auto owner = sf.insert("k", emb, EntityTokens{});
    auto fut = sf.try_merge("k", emb, "",
                           EntityTokens{}, false);
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
    (void)sf.insert("k", emb, EntityTokens{});
    auto fut = sf.try_merge("k", emb, "",
                           EntityTokens{}, false);
    CHECK(fut.has_value()); ok++;
    CHECK(fut->wait_for(20ms) == std::future_status::timeout); ok++;
  }

  // 5. 被顶替的旧 leader 迟到的 complete 不得写入新槽位
  {
    Singleflight sf;
    auto old_owner = sf.insert("k", emb, EntityTokens{});  // leader A
    auto fut_a = sf.try_merge("k", emb, "",
                           EntityTokens{}, false);   // 等待者合并到 A
    auto new_owner = sf.insert("k", emb, EntityTokens{});  // A 超时被顶替，B 成为 leader
    sf.complete("k", "A的结果（旧世代，应被丢弃）", old_owner);

    auto fut_b = sf.try_merge("k", emb, "",
                           EntityTokens{}, false);   // 槽位应仍属于 B
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
    auto owner = sf.insert("k", emb, EntityTokens{});
    sf.complete("k", "v", owner);
    sf.complete("k", "v", owner);  // 重复 complete：找不到条目，直接返回
    CHECK(true); ok++;
  }

  // 7. 等待者在 worker 线程内 try/catch 后回源（等价 main.cpp 的等待方逻辑）
  {
    Singleflight sf;
    auto owner = sf.insert("k", emb, EntityTokens{});
    auto fut = sf.try_merge("k", emb, "",
                           EntityTokens{}, false);
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

  // 8. 实体不一致不得合并（本轮修复的回归，判别力见每条 CHECK 的注释）
  //
  //    构造方式刻意把"语义"这一维拉满：所有请求用**同一个向量**（cos = 1.0，
  //    远超 0.95 合并阈值），唯一差别落在实体标记上。因此这一组断言只有两种
  //    可能的结果——实体规则生效（不合并）或规则缺失（合并）。
  //    修复前（try_merge 只比余弦）这里 3 条 CHECK 全部失败；把实体规则误删或
  //    改成"从不合并"也会失败（见第 9 组）。
  {
    Singleflight sf;
    const EntityTokens leader = extract_entity_tokens("压力测试唯一问题编号1234");
    (void)sf.insert("qa", emb, leader);

    // 每个候选：数字/缩略语与 leader 存在不对称差集
    const std::string mismatch_cases[] = {
        "压力测试唯一问题编号5678",  // 数字不同（1 万条灌入实测中 5185 次误合并的来源）
        "继续12题",                  // 查询侧有数字、leader 侧一个都没有
        "什么是DMA",                 // 大写缩略语不同
    };
    for (const auto& text : mismatch_cases) {
      CHECK(!sf.try_merge("qb", emb, "",
                           extract_entity_tokens(text), true)
                 .has_value());
      ok++;
    }
    CHECK(sf.merge_veto_count() == std::size(mismatch_cases)); ok++;

    // 同一批候选在 veto 关闭时**必须**合并：证明上面 3 次失败只来自实体规则，
    // 而不是键/向量本来就没对上（否则那 3 条断言是假阳性）
    CHECK(sf
              .try_merge("qb", emb, "",
                           extract_entity_tokens(mismatch_cases[0]),
                         false)
              .has_value());
    ok++;
  }

  // 9. 实体一致时仍必须合并（防止"把合并整条路径关掉"被当成修好了）
  {
    Singleflight sf;
    (void)sf.insert("k-http", emb, extract_entity_tokens("什么是HTTP协议"));
    CHECK(sf.try_merge("k-http2", emb,
                       "",
                           extract_entity_tokens("请问什么是HTTP协议"), true)
              .has_value());
    ok++;
    // 两边实体集合都为空（普通同义句，`继续下一题` ↔ `请继续下一题`）：放行
    (void)sf.insert("k-next", emb, extract_entity_tokens("继续下一题"));
    CHECK(sf.try_merge("k-next2", emb,
                       "",
                           extract_entity_tokens("请继续下一题"), true)
              .has_value());
    ok++;
    // 实体完全相同而问题措辞不同：放行
    (void)sf.insert("k-num", emb, extract_entity_tokens("问题编号1234 是什么"));
    CHECK(sf.try_merge("k-num2", emb,
                       "",
                           extract_entity_tokens("编号1234 到底是什么意思"), true)
              .has_value());
    ok++;
  }

  // 10. 一条候选被否决 ≠ 整体放弃：在途表里另一条实体一致的候选仍应合并
  //     （unordered_map 的遍历顺序不确定，因此这组断言与顺序无关：
  //      两条候选里恰有一条实体一致 ⇒ 必须返回 has_value）
  {
    Singleflight sf;
    (void)sf.insert("k-1234", emb, extract_entity_tokens("问题编号1234"));
    (void)sf.insert("k-5678", emb, extract_entity_tokens("问题编号5678"));
    CHECK(sf.try_merge("k-9999", emb, "",
                           extract_entity_tokens("问题编号5678"),
                       true)
              .has_value());
    ok++;
    // 注意这里不检查 sf.merge_veto_count()：遍历到 k-5678 就返回了，是否恰好
    // 先路过 k-1234 取决于 unordered_map 的桶顺序，断它必然是 flaky 的

    // 只有不一致的候选时：不得合并，且确实是被实体规则否决（而不是没匹配上）
    Singleflight sf2;
    (void)sf2.insert("k-1234", emb, extract_entity_tokens("问题编号1234"));
    CHECK(!sf2.try_merge("k-9999", emb, "",
                           extract_entity_tokens("问题编号5678"),
                         true)
               .has_value());
    ok++;
    CHECK(sf2.merge_veto_count() == 1); ok++;
    // 同一对请求在 veto 关闭时必须合并：这是"没匹配上"与"被否决"的对照
    CHECK(sf2.try_merge("k-9999", emb, "",
                           extract_entity_tokens("问题编号5678"),
                        false)
              .has_value());
    ok++;
  }

  return test_check::finish("test_singleflight", ok);
}
