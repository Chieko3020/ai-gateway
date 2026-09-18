// 缓存协调器实现
#include "cache/cache_engine.h"

#include <cmath>
#include <format>

#include "common/logger.h"

namespace ai_gateway {

CacheEngine::CacheEngine(const EmbeddingConfig& emb_cfg,
                         const CacheConfig& cache_cfg,
                         std::shared_ptr<LruStore> store,
                         std::shared_ptr<HnswIndex> index,
                         EmbedFn embed_fn)
    : emb_cfg_(emb_cfg),
      threshold_(cache_cfg.similarity_threshold),
      store_(std::move(store)),
      index_(std::move(index)),
      embed_fn_(std::move(embed_fn)) {}

CacheEngine::HitResult CacheEngine::try_hit(
    const std::string& user_message,
    const std::string& ns) {
  // 缓存 key: namespace:msg
  auto ns_key = ns.empty() ? user_message : ns + ":" + user_message;

  // 1. 向量化用户消息（不持锁 embed_fn_ 可能很慢）
  auto vec = embed_fn_(emb_cfg_.url, emb_cfg_.api_key,
                           emb_cfg_.model, user_message, 5);
  if (vec.empty()) {
    // 嵌入失败 降级为精确匹配
    LOG_WARN("cache: embedding failed, fallback to exact match");
    auto exact = store_->get(ns_key);
    if (exact.has_value()) {
      LOG_INFO("cache: HIT (exact) ns={}", ns.empty() ? "default" : ns);
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
  ++total_search_;
  for (auto& r : results) {
    if (r.similarity >= threshold_) {
      if (!ns.empty() && !r.key.starts_with(ns_prefix)) continue;
      auto cached = store_->get(r.key);
      if (cached.has_value()) {
        LOG_INFO("cache: HIT key={} sim={:.3f} ns={}", r.key, r.similarity,
                 ns.empty() ? "default" : ns);
        return HitResult{true, std::move(cached.value()), r.similarity};
      }
      ++ghost_count_;
    }
  }

  // 4. 未命中 带回 embedding 避免 cache_reply 重复计算
  LOG_INFO("cache: MISS top_sim={:.3f} threshold={:.3f}",
           results.empty() ? 0.0f : results[0].similarity,
           threshold_);
  return HitResult{false, "", 0.0f, std::move(vec)};
}

void CacheEngine::cache_reply(const std::string& user_message,
                               const std::string& reply,
                               const std::vector<float>& cached_embedding,
                               const std::string& ns) {
  std::lock_guard lock(mutex_);

  auto key = std::format("{}msg:{}", ns.empty() ? "" : ns + ":", next_id_);
  ++next_id_;

  auto ns_key = ns.empty() ? user_message : ns + ":" + user_message;

  if (cached_embedding.empty()) {
    store_->put(std::move(key), reply);
    store_->put(std::move(ns_key), reply);
    return;
  }

  store_->put_with_embedding(key, reply, cached_embedding);
  index_->add(next_id_ - 1, key, cached_embedding);
  store_->put(std::move(ns_key), reply);

  LOG_DEBUG("cache: stored key={} ns_key={} dims={}", key,
            ns.empty() ? user_message : ns + ":" + user_message,
            cached_embedding.size());
}

void CacheEngine::rebuild_index() {
  LOG_INFO("cache: rebuilding vector index...");

  // 1. 锁外构建新索引：建图要对每个条目跑一次 O(ef_construction) 搜索，
  //    整个过程持 mutex_ 会让检索与写入全部阻塞，因此先构建、再交换。
  auto new_index_ptr = std::make_shared<HnswIndex>();
  auto& new_idx = *new_index_ptr;
  int64_t new_id = 1;

  store_->for_each_embedding(
      [&](const std::string& key, const std::vector<float>& emb) {
        new_idx.add(new_id, key, emb);
        ++new_id;
      });

  // 2. 短临界区交换：与 try_hit 的取副本、cache_reply 的 add 互斥
  size_t vectors = 0;
  {
    std::lock_guard lock(mutex_);
    index_ = std::move(new_index_ptr);
    next_id_ = new_id;
    vectors = index_->size();
  }

  LOG_INFO("cache: index rebuilt, {} vectors", vectors);
}

std::pair<int, size_t> CacheEngine::ghost_stats() const {
  std::lock_guard lock(mutex_);
  if (total_search_ == 0) return {0, 0};
  int rate = static_cast<int>(ghost_count_ * 100 / total_search_);
  return {rate, total_search_};
}

void CacheEngine::try_rebuild_if_ghosty() {
  auto [rate, total] = ghost_stats();
  if (total < 10) return;
  if (rate > 5) {
    LOG_INFO("cache: ghost rate {}% ({}/{}), auto-rebuilding index", rate,
             ghost_count_, total);
    // rebuild_index() 自带 mutex_：这里不能先取锁再调用（同线程递归加锁会死锁）
    rebuild_index();
    ghost_count_ = 0;
    total_search_ = 0;
  }
}

}  // namespace ai_gateway
