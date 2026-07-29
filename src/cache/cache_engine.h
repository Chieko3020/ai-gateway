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

#include "common/config.h"
#include "lru_store.h"
#include "vector_index.h"

namespace ai_gateway {

class CacheEngine {
 public:
  // 传入配置和依赖组件
  CacheEngine(const EmbeddingConfig& emb_cfg,
              const CacheConfig& cache_cfg,
              std::shared_ptr<LruStore> store,
              std::shared_ptr<VectorIndex> index);

  // 尝试从缓存中获取回复
  // user_message: 用户最新一条消息文本
  // 返回 (缓存的 LLM 回复, 相似度)；未命中返回 nullopt
  struct HitResult {
    std::string reply;
    float similarity = 0.0f;
  };
  std::optional<HitResult> try_hit(const std::string& user_message);

  // 将 LLM 回复存入缓存（供未命中后使用）
  void cache_reply(const std::string& user_message,
                   const std::string& reply);

  // 从 lru_store 重建 vector_index（启动恢复 / 缓存逐出后同步）
  void rebuild_index();

  // 统计
  size_t hit_count() const { return hit_count_; }
  size_t miss_count() const { return miss_count_; }
  double hit_rate() const;

  // 配置访问
  float threshold() const { return threshold_; }
  int top_k() const { return top_k_; }

 private:
  EmbeddingConfig emb_cfg_;
  std::shared_ptr<LruStore> store_;
  std::shared_ptr<VectorIndex> index_;

  float threshold_ = 0.0f;  // 由 CacheConfig 注入
  int top_k_ = 3;

  int64_t next_id_ = 1;  // 自增 ID，用于 vector_index 映射

  size_t hit_count_ = 0;
  size_t miss_count_ = 0;
};

}  // namespace ai_gateway
