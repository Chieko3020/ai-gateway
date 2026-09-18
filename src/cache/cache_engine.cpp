// 缓存协调器实现
#include "cache/cache_engine.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string_view>

#include "common/logger.h"

namespace ai_gateway {

namespace {

// 从缓存键 "[ns:]msg:N" 中解析出编号 N。
// 解析失败返回 false（调用方按"没有编号"处理，不会因此回退 next_id_）
bool parse_msg_id(const std::string& key, int64_t& out) {
  constexpr std::string_view kTag = "msg:";
  auto pos = key.rfind(kTag);
  if (pos == std::string::npos) return false;
  // 前缀必须是空串或形如 "<ns>:"，避免把用户消息自带的 "msg:" 误当成缓存键编号
  if (pos != 0 && key[pos - 1] != ':') return false;
  auto digits = key.substr(pos + kTag.size());
  if (digits.empty()) return false;
  int64_t value = 0;
  for (char c : digits) {
    if (c < '0' || c > '9') return false;
    if (value > (int64_t{1} << 40)) return false;  // 异常长的数字：放弃解析
    value = value * 10 + (c - '0');
  }
  out = value;
  return true;
}

}  // namespace

CacheEngine::CacheEngine(const EmbeddingConfig& emb_cfg,
                         const CacheConfig& cache_cfg,
                         std::shared_ptr<LruStore> store,
                         std::shared_ptr<HnswIndex> index,
                         EmbedFn embed_fn)
    : emb_cfg_(emb_cfg),
      threshold_(cache_cfg.similarity_threshold),
      hnsw_cfg_(index ? index->config() : HnswConfig{}),
      store_(std::move(store)),
      index_(std::move(index)),
      embed_fn_(std::move(embed_fn)) {}

CacheEngine::HitResult CacheEngine::try_hit(
    const std::string& user_message,
    const std::string& ns) {
  // 缓存 key: namespace:msg
  auto ns_key = ns.empty() ? user_message : ns + ":" + user_message;

  // 1. 向量化用户消息（不持锁 embed_fn_ 可能很慢）
  auto vec = embed_fn_(user_message, 5);
  if (vec.empty()) {
    // 嵌入失败 降级为精确匹配。
    // 匹配的是"条目的原始查表键"（由 set_source 写入），因此不需要为每条回复
    // 再存一份完整副本（旧实现为此把 max_entries 的实际容量砍半，报告 M4）
    LOG_WARN("cache: embedding failed, fallback to exact match");
    auto exact = store_->get_exact(ns_key);
    if (exact.has_value()) {
      LOG_INFO_SAMPLED("cache: HIT (exact) ns={}", ns.empty() ? "default" : ns);
      return HitResult{true, std::move(exact.value()), 1.0f};
    }
    return HitResult{};  // 未命中
  }

  // 2. 向量检索 Top-K
  //    在临界区内取出索引副本，之后在临界区外检索该副本：
  //    rebuild_index() 会替换 index_，旧索引对象的生命周期由这里的 shared_ptr
  //    引用计数兜住，检索期间不会被析构（原实现直接读裸指针 → 可能 use-after-free）
  std::shared_ptr<HnswIndex> idx;
  {
    std::lock_guard lock(mutex_);
    idx = index_;
  }
  std::vector<HnswResult> results = idx->search(vec, top_k_);

  // 3. 遍历结果，检查是否命中（相似度 ≥ 阈值，且命名空间匹配）
  std::string ns_prefix = ns.empty() ? "" : ns + ":";
  total_search_.fetch_add(1, std::memory_order_relaxed);
  for (auto& r : results) {
    if (r.similarity >= threshold_) {
      if (!ns.empty() && !r.key.starts_with(ns_prefix)) continue;
      auto cached = store_->get(r.key);
      if (cached.has_value()) {
        LOG_INFO_SAMPLED("cache: HIT key={} sim={:.3f} ns={}", r.key,
                         r.similarity, ns.empty() ? "default" : ns);
        return HitResult{true, std::move(cached.value()), r.similarity};
      }
      ghost_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // 4. 未命中 带回 embedding 避免 cache_reply 重复计算
  LOG_INFO_SAMPLED("cache: MISS top_sim={:.3f} threshold={:.3f}",
                   results.empty() ? 0.0f : results[0].similarity, threshold_);
  return HitResult{false, "", 0.0f, std::move(vec)};
}

void CacheEngine::cache_reply(const std::string& user_message,
                               const std::string& reply,
                               const std::vector<float>& cached_embedding,
                               const std::string& ns) {
  std::lock_guard lock(mutex_);

  auto key = std::format("{}msg:{}", ns.empty() ? "" : ns + ":", next_id_);
  ++next_id_;

  // 原始查表键（namespace:user_message）：只作为条目的 source 字段保存，
  // 不再额外存一份回复——旧实现每条回复写两个条目，max_entries=10000 实际只装 5000 条
  auto ns_key = ns.empty() ? user_message : ns + ":" + user_message;

  if (cached_embedding.empty()) {
    store_->put(key, reply);
    store_->set_source(key, std::move(ns_key));
    LOG_DEBUG("cache: stored key={} (no embedding) len={}", key, reply.size());
    return;
  }

  store_->put_with_embedding(key, reply, cached_embedding);
  store_->set_source(key, ns_key);
  index_->add(next_id_ - 1, key, cached_embedding);

  // 不打印源消息原文（ns_key 含完整用户消息），只落长度（报告 M5）
  LOG_DEBUG("cache: stored key={} content_len={} src_len={} dims={}", key,
            reply.size(), ns_key.size(), cached_embedding.size());
}

void CacheEngine::rebuild_index() {
  LOG_INFO("cache: rebuilding vector index...");

  // 1. 锁外构建新索引：建图要对每个条目跑一次 O(ef_construction) 搜索，
  //    整个过程持 mutex_ 会让检索与写入全部阻塞，因此先构建、再交换。
  auto new_index_ptr = std::make_shared<HnswIndex>(hnsw_cfg_);
  auto& new_idx = *new_index_ptr;

  // next_id_ 不能由"存活条目数 + 1"派生：TTL 是逐条过期的，存活键的空间里存在空洞，
  // 退回的编号会让新条目复用仍然存活的键 msg:N（旧索引节点仍指向该键，于是用旧问题
  // 的向量检索会命中旧节点、却取回新问题的答案）。这里取"存活键里 max(N) + 1"，
  // 保证新编号只前进、不与任何存活键冲突。
  int64_t max_id = 0;
  store_->for_each_embedding(
      [&](const std::string& key, const std::vector<float>& emb) {
        int64_t id = 0;
        if (parse_msg_id(key, id) && id > max_id) max_id = id;
        new_idx.add(static_cast<int>(id), key, emb);
      });

  // 2. 短临界区交换：与 try_hit 的取副本、cache_reply 的 add 互斥
  //    （日志用的两个值都在临界区内取出，避免锁外读 next_id_ 的竞争）
  size_t vectors = 0;
  int64_t next_id = 0;
  {
    std::lock_guard lock(mutex_);
    index_ = std::move(new_index_ptr);
    next_id_ = std::max(next_id_, max_id + 1);  // 单调不回退
    vectors = index_->size();
    next_id = next_id_;
  }

  LOG_INFO("cache: index rebuilt, {} vectors, next_id={}", vectors, next_id);
}

size_t CacheEngine::index_size() const {
  std::lock_guard lock(mutex_);
  return index_ ? index_->size() : 0;
}

std::pair<int, size_t> CacheEngine::ghost_stats() const {
  std::lock_guard lock(mutex_);
  const size_t total = total_search_.load(std::memory_order_relaxed);
  if (total == 0) return {0, 0};
  const size_t ghosts = ghost_count_.load(std::memory_order_relaxed);
  int rate = static_cast<int>(ghosts * 100 / total);
  return {rate, total};
}

void CacheEngine::try_rebuild_if_ghosty() {
  auto [rate, total] = ghost_stats();
  if (total < 10) return;
  if (rate > 5) {
    LOG_INFO("cache: ghost rate {}% ({}/{}), auto-rebuilding index", rate,
             ghost_count_.load(std::memory_order_relaxed), total);
    // rebuild_index() 自带 mutex_：这里不能先取锁再调用（同线程递归加锁会死锁）
    rebuild_index();
    ghost_count_.store(0, std::memory_order_relaxed);
    total_search_.store(0, std::memory_order_relaxed);
  }
}

}  // namespace ai_gateway
