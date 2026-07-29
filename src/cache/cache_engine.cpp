// 缓存协调器实现
#include "cache_engine.h"

#include <cmath>
#include <format>

#include "common/logger.h"
#include "embedding.h"

namespace ai_gateway {

CacheEngine::CacheEngine(const EmbeddingConfig& emb_cfg,
                         const CacheConfig& cache_cfg,
                         std::shared_ptr<LruStore> store,
                         std::shared_ptr<VectorIndex> index,
                         EmbedFn embed_fn)
    : emb_cfg_(emb_cfg),
      threshold_(cache_cfg.similarity_threshold),
      store_(std::move(store)),
      index_(std::move(index)),
      embed_fn_(std::move(embed_fn)) {}

std::optional<CacheEngine::HitResult> CacheEngine::try_hit(
    const std::string& user_message) {
  // 1. 向量化用户消息
  auto vec = embed_fn_(emb_cfg_.url, emb_cfg_.api_key,
                           emb_cfg_.model, user_message, 5);
  if (vec.empty()) {
    // 嵌入失败 → 降级为精确匹配（MD5级）
    LOG_WARN("cache: embedding failed, fallback to exact match");
    auto exact = store_->get(user_message);
    if (exact.has_value()) {
      LOG_INFO("cache: HIT (exact)");
      return HitResult{std::move(exact.value()), 1.0f};
    }
    return std::nullopt;
  }

  // 2. 向量检索 Top-K
  auto results = index_->search(vec, top_k_);

  // 3. 遍历结果，检查是否命中（相似度 ≥ 阈值）
  for (auto& r : results) {
    if (r.similarity >= threshold_) {
      auto cached = store_->get(r.key);
      if (cached.has_value()) {
        LOG_INFO("cache: HIT key={} sim={:.3f}", r.key, r.similarity);
        return HitResult{std::move(cached.value()), r.similarity};
      }
    }
  }

  // 4. 未命中
  LOG_INFO("cache: MISS top_sim={:.3f} threshold={:.3f}",
           results.empty() ? 0.0f : results[0].similarity,
           threshold_);
  HitResult miss;
  miss.embedding = std::move(vec);  // 带回 embedding，避免 cache_reply 重复计算
  return miss;
}

void CacheEngine::cache_reply(const std::string& user_message,
                               const std::string& reply,
                               const std::vector<float>& cached_embedding) {
  if (cached_embedding.empty()) {
    // 无 embedding 时仍缓存回复（精确缓存可用）
    store_->put(user_message, reply);
    return;
  }

  auto key = std::format("msg:{}", next_id_);
  ++next_id_;

  store_->put_with_embedding(key, reply, cached_embedding);
  index_->add(next_id_ - 1, key, cached_embedding);

  LOG_DEBUG("cache: stored key={} dims={}", key, cached_embedding.size());
}

void CacheEngine::rebuild_index() {
  // 从 lru_store 重建 vector_index：
  // 遍历所有非过期条目 → 调用 for_each_embedding 添加到 index
  // 当前实现：先清空 index，再全量重建
  // TODO: 改为增量同步（但 <10k 条目全量重建 < 1ms）
  LOG_INFO("cache: rebuilding vector index...");

  // 先构建新索引，完成后再替换旧索引
  auto new_index_ptr = std::make_shared<VectorIndex>();
  auto& new_idx = *new_index_ptr;
  int64_t new_id = 1;

  store_->for_each_embedding(
      [&](const std::string& key, const std::vector<float>& emb) {
        new_idx.add(new_id, key, emb);
        ++new_id;
      });

  index_ = std::move(new_index_ptr);
  next_id_ = new_id;

  LOG_INFO("cache: index rebuilt, {} vectors", index_->size());
}

}  // namespace ai_gateway
