// 缓存协调器：语义缓存的核心编排引擎
//
// 判断流程：
//   1. 接收 user message 调用 embedding API 得到向量
//   2. hnsw_index.search 得到 Top-K 相似条目
//   3. max(similarity) ≥ threshold？
//      命中返回 lru_store 中的缓存回复
//      未命中返回 nullopt，由上层转发 LLM 后将结果 + 向量存入缓存
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <functional>

#include "common/config.h"
#include "cache/lru_store.h"
#include "cache/hnsw_index.h"

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
              std::shared_ptr<HnswIndex> index,
              EmbedFn embed_fn);

  // 尝试从缓存中获取回复
  // user_message: 用户最新一条消息文本
  // 返回命中结果：hit=true 表示找到缓存，hit=false 表示未命中
  // 未命中时 embedding 字段带回已计算的向量（避免 cache_reply 重复调用 API）
  struct HitResult {
    bool hit = false;
    std::string reply;
    float similarity = 0.0f;
    std::vector<float> embedding;
  };
  HitResult try_hit(const std::string& user_message,
                     const std::string& ns = "");

  // 将 LLM 回复存入缓存，传入已计算的 embedding 向量（避免重复 API 调用）
  void cache_reply(const std::string& user_msg, const std::string& reply,
                   const std::vector<float>& embedding,
                   const std::string& ns = "");

  // 从 LruStore 重建向量索引（缓存恢复后调用）
  void rebuild_index();

  // 配置访问
  float threshold() const { return threshold_; }
  int top_k() const { return top_k_; }

 private:
  EmbeddingConfig emb_cfg_;
  std::shared_ptr<LruStore> store_;
  std::shared_ptr<HnswIndex> index_;

  EmbedFn embed_fn_;
  float threshold_ = 0.0f;  // 由 CacheConfig 注入
  int top_k_ = 3;
  int64_t next_id_ = 1;

  // 保护 next_id_ + HNSW add/search 并发（LruStore 自有锁）
  mutable std::mutex mutex_;
};

}  // namespace ai_gateway
