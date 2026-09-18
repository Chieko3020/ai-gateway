// CacheEngine 单元测试
//
// 覆盖点：
//   1. C3 回归：purge 造成键空间空洞后 rebuild，再写入不得复用存活键
//      （修复前 next_id_ 退回"存活条目数+1"，新键 msg:N 覆盖旧键，
//       于是旧问题的向量检索会取回另一个问题的答案）
//   2. M8：索引向量数通过 engine.index_size() 观察（不得依赖已失效的索引对象）
//   3. C1：检索与 rebuild 并发时不丢结果、不死锁
//   4. 幽灵率统计 + try_rebuild_if_ghosty() 自动重建（不得同线程递归加锁）
#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cache/cache_engine.h"
#include "test_check.h"

using namespace ai_gateway;
using namespace std::chrono_literals;

namespace {

constexpr int kDim = 512;

// 文本中的数字作为维度下标，构造单位向量：同一文本 -> 同一向量，不同文本互不碰撞
std::vector<float> vec_of_text(const std::string& text) {
  int idx = 0;
  for (char c : text) {
    if (c >= '0' && c <= '9') idx = idx * 10 + (c - '0');
  }
  std::vector<float> v(kDim, 0.0f);
  v[static_cast<size_t>(idx) % kDim] = 1.0f;
  return v;
}

std::vector<float> embed_fn(const std::string& text, int) {
  return vec_of_text(text);
}

// 收集所有"带向量的存活键"（= 索引可达的键集合）
std::vector<std::string> embedding_keys(const std::shared_ptr<LruStore>& lru) {
  std::vector<std::string> keys;
  lru->for_each_embedding(
      [&](const std::string& k, const std::vector<float>&) { keys.push_back(k); });
  std::sort(keys.begin(), keys.end());
  return keys;
}

// 阻塞直到 ttl 到期后的条目被清理干净（避免固定 sleep 在负载高时不够），
// 并累加返回清理条目总数
bool purge_until_empty(const std::shared_ptr<LruStore>& lru, size_t* total_purged) {
  auto deadline = std::chrono::steady_clock::now() + 5s;
  *total_purged = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    *total_purged += lru->purge_expired();
    if (lru->size() == 0) return true;
    std::this_thread::sleep_for(50ms);
  }
  return false;
}

}  // namespace

int main() {
  int ok = 0;
  EmbeddingConfig ec;
  CacheConfig cc;
  cc.similarity_threshold = 0.85f;

  // ── 1. C3 回归：purge -> rebuild -> 写入不得与存活键冲突 ────────────────
  {
    const std::string a1 = "A1-问题1的答案", a2 = "A2-问题2的答案";
    const std::string a3 = "A3-问题3的答案", a5 = "A5-问题5的答案";

    auto lru = std::make_shared<LruStore>(100, 1);  // TTL = 1s
    CacheEngine engine(ec, cc, lru,
                       std::make_shared<HnswIndex>(HnswConfig{kDim, 16, 100, 50}),
                       embed_fn);

    engine.cache_reply("q1", a1, vec_of_text("q1"), "");  // 键 msg:1
    engine.cache_reply("q2", a2, vec_of_text("q2"), "");  // 键 msg:2

    // 等 q1/q2 过期并清理：键空间留下空洞（msg:1/msg:2 消失，编号但已前进）
    size_t purged = 0;
    CHECK(purge_until_empty(lru, &purged)); ok++;
    // M4：每条回复只落一个条目（旧实现额外写一份 ns_key 副本，这里是 4）
    CHECK(purged == 2); ok++;

    engine.cache_reply("q3", a3, vec_of_text("q3"), "");  // 键 msg:3
    engine.cache_reply("q4", "A4-问题4的答案", vec_of_text("q4"), "");  // 键 msg:4

    auto before = embedding_keys(lru);
    CHECK(before.size() == 2); ok++;

    engine.rebuild_index();
    CHECK(engine.index_size() == before.size()); ok++;  // M8：观察值走引擎

    // 重建后写入第 5 条：修复前 next_id_ 退回 3，键 msg:3 与存活键冲突
    engine.cache_reply("q5", a5, vec_of_text("q5"), "");

    auto after = embedding_keys(lru);
    CHECK(after.size() == before.size() + 1); ok++;  // 新键必须是全新键
    bool all_alive = true;
    for (const auto& k : before) {
      if (std::find(after.begin(), after.end(), k) == after.end()) all_alive = false;
    }
    CHECK(all_alive); ok++;  // 旧键全部仍存活（未被新写入覆盖）

    // 旧问题的向量必须取回旧答案（修复前这里返回 A5 —— 缓存串答案）
    auto hit3 = engine.try_hit("q3", "");
    CHECK(hit3.hit); ok++;
    CHECK(hit3.reply == a3); ok++;
    auto hit5 = engine.try_hit("q5", "");
    CHECK(hit5.hit); ok++;
    CHECK(hit5.reply == a5); ok++;
    CHECK(engine.index_size() == 3); ok++;
  }

  // ── 2. 幽灵条目 -> try_rebuild_if_ghosty() 自动重建 ────────────────────
  {
    auto lru = std::make_shared<LruStore>(100, 1);
    CacheEngine engine(ec, cc, lru,
                       std::make_shared<HnswIndex>(HnswConfig{kDim, 16, 100, 50}),
                       embed_fn);
    for (int i = 1; i <= 12; ++i) {
      engine.cache_reply("g" + std::to_string(i), "A" + std::to_string(i),
                         vec_of_text("g" + std::to_string(i)), "");
    }
    CHECK(engine.index_size() == 12); ok++;

    // 全部过期并清理：索引仍指向已消失的键 -> 检索"搜到但取不到"
    size_t purged = 0;
    CHECK(purge_until_empty(lru, &purged)); ok++;

    int hits = 0;
    for (int i = 1; i <= 12; ++i) {
      if (engine.try_hit("g" + std::to_string(i), "").hit) ++hits;
    }
    CHECK(hits == 0); ok++;
    auto [rate, total] = engine.ghost_stats();
    CHECK(total >= 12); ok++;
    CHECK(rate == 100); ok++;

    // 幽灵率 100% > 5% 且样本 >= 10 -> 自动重建
    // （rebuild_index() 自带锁，这里若持锁调用会死锁，本用例即该回归）
    engine.try_rebuild_if_ghosty();
    CHECK(engine.index_size() == 0); ok++;  // 重建后索引与已清空的 store 一致
    CHECK(engine.ghost_stats().second == 0); ok++;  // 计数已清零
  }

  // ── 3. 检索与 rebuild 并发：不丢结果、不崩溃 ──────────────────────────
  {
    auto lru = std::make_shared<LruStore>(1000, 0);  // 永不过期
    CacheEngine engine(ec, cc, lru,
                       std::make_shared<HnswIndex>(HnswConfig{kDim, 16, 100, 50}),
                       embed_fn);
    constexpr int kEntries = 20;
    for (int i = 0; i < kEntries; ++i) {
      engine.cache_reply("c" + std::to_string(i), "A" + std::to_string(i),
                         vec_of_text("c" + std::to_string(i)), "");
    }

    std::atomic<int> hits{0}, misses{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
      workers.emplace_back([&, t] {
        for (int i = 0; i < 100; ++i) {
          auto r = engine.try_hit("c" + std::to_string((i + t) % kEntries), "");
          if (r.hit) ++hits; else ++misses;
        }
      });
    }
    for (int i = 0; i < 5; ++i) engine.rebuild_index();
    for (auto& w : workers) w.join();

    CHECK(hits + misses == 400); ok++;
    CHECK(misses == 0); ok++;  // 重建期间旧索引副本仍由引用计数保活
    CHECK(engine.index_size() == kEntries); ok++;
  }

  // ── 4. M4：一条回复只占一个条目 + 降级精确匹配 ────────────────────────
  {
    auto lru = std::make_shared<LruStore>(2, 0);  // 只装 2 条，永不过期
    // embed_fn 可切换：先正常向量化，再模拟 embedding 不可用
    bool embed_ok = true;
    auto engine = std::make_shared<CacheEngine>(
        ec, cc, lru,
        std::make_shared<HnswIndex>(HnswConfig{kDim, 16, 100, 50}),
        [&embed_ok](const std::string& text, int) {
          if (!embed_ok) return std::vector<float>{};
          return vec_of_text(text);
        });

    engine->cache_reply("m4-q1", "A1", vec_of_text("m4-q1"), "");
    engine->cache_reply("m4-q2", "A2", vec_of_text("m4-q2"), "");
    // 旧实现每条回复写 2 个条目 -> max_entries=2 时实际只剩 1 条回复
    CHECK(lru->size() == 2); ok++;
    CHECK(engine->index_size() == 2); ok++;

    auto probe1 = engine->try_hit("m4-q1", "");
    CHECK(probe1.hit); ok++;
    CHECK(probe1.reply == "A1"); ok++;

    // embedding 不可用时的降级路径：必须能精确命中（靠条目内保存的 source 键）
    embed_ok = false;
    auto exact1 = engine->try_hit("m4-q1", "");
    CHECK(exact1.hit); ok++;
    CHECK(exact1.reply == "A1"); ok++;
    auto exact_miss = engine->try_hit("m4-unknown", "");
    CHECK(!exact_miss.hit); ok++;

    // 带 namespace：source 键含 ns 前缀，隔离必须成立
    embed_ok = true;
    engine->cache_reply("m4-q3", "A3", vec_of_text("m4-q3"), "uuidA");
    embed_ok = false;
    auto ns_hit = engine->try_hit("m4-q3", "uuidA");
    CHECK(ns_hit.hit); ok++;
    CHECK(ns_hit.reply == "A3"); ok++;
    auto ns_other = engine->try_hit("m4-q3", "uuidB");
    CHECK(!ns_other.hit); ok++;  // 另一个 namespace 不应命中原条目的 source
  }

  return test_check::finish("test_cache_engine", ok);
}
