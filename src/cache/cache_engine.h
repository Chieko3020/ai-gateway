// 缓存协调器：语义缓存的核心编排引擎
//
// 判断流程：
//   1. 接收 user message → 调用 embedding API → 得到向量
//   2. vector_index.search → Top-K 相似条目
//   3. max(similarity) ≥ threshold？→ 命中：返回 lru_store 中的缓存回复
//   4. 未命中 → 返回 nullopt，由上层转发 LLM 后将结果 + 向量存入缓存
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <functional>

#include "common/config.h"
#include "embedding.h"
#include "lru_store.h"
#include "vector_index.h"

namespace ai_gateway {

class CacheEngine {
 public:
  // 传入配置和依赖组件
  using EmbedFn = std::function<std::vector<float>(
      std::string url, std::string key,
      std::string model, std::string text,
      int timeout)>;

  CacheEngine(const EmbeddingConfig& emb_cfg,
              const CacheConfig& cache_cfg,
              std::shared_ptr<LruStore> store,
              std::shared_ptr<VectorIndex> index,
              EmbedFn embed_fn = get_embedding);

  // 尝试从缓存中获取回复
  // user_message: 用户最新一条消息文本
  // 返回 (缓存的 LLM 回复, 相似度)；未命中返回 nullopt
  struct HitResult {
    std::string reply;
    float similarity = 0.0f;
    std::vector<float> embedding;  // 未命中时带回 embedding，避免 cache_reply 重复计算
  };
  std::optional<HitResult> try_hit(const std::string& user_message);

  // 将 LLM 回复存入缓存，传入已计算的 embedding 向量（避免重复 API 调用）
  void cache_reply(const std::string& user_message,
                   const std::string& reply,
                   const std::vector<float>& cached_embedding);

  // 从 LruStore 重建向量索引（缓存恢复后调用）
  void rebuild_index();

  // 配置访问
  float threshold() const { return threshold_; }
  int top_k() const { return top_k_; }

 private:
  EmbeddingConfig emb_cfg_;
  std::shared_ptr<LruStore> store_;
  std::shared_ptr<VectorIndex> index_;

  EmbedFn embed_fn_;
  float threshold_ = 0.0f;  // 由 CacheConfig 注入
  int top_k_ = 3;
  int64_t next_id_ = 1;
};

}  // namespace ai_gateway
