// 缓存协调器：语义缓存的核心编排引擎
//
// 判断流程：
//   1. 接收 user message 调用 embedding API 得到向量
//   2. hnsw_index.search 得到 Top-K 相似条目
//   3. max(similarity) ≥ threshold？
//      命中返回 lru_store 中的缓存回复
//      未命中返回 nullopt，由上层转发 LLM 后将结果 + 向量存入缓存
#pragma once

#include <atomic>
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
  // 线程安全：函数内部自行加锁，调用方无需（也不得）持 mutex_——
  // 持锁调用会在同一线程递归加锁，直接死锁。
  void rebuild_index();

  // 配置访问
  float threshold() const { return threshold_; }
  int top_k() const { return top_k_; }

  // 幽灵向量观测：返回"搜到但取不到"的比例，用于判断是否需要 rebuild_index
  std::pair<int, size_t> ghost_stats() const;

  // 如果幽灵率超过阈值，自动重建索引（在 60s 定时器中调用）
  void try_rebuild_if_ghosty();

 private:
  EmbeddingConfig emb_cfg_;
  std::shared_ptr<LruStore> store_;

  // 索引由本引擎独占持有（调用方不应继续持有同一个 shared_ptr 用于观察）。
  // 读写约定：所有对 index_ 的读（含取出副本）与写（rebuild_index 交换）都在
  // mutex_ 临界区内完成；检索方在临界区内取一份 shared_ptr 副本后在临界区外使用，
  // 由引用计数保证检索期间索引对象不被析构。
  std::shared_ptr<HnswIndex> index_;

  EmbedFn embed_fn_;
  float threshold_ = 0.0f;  // 由 CacheConfig 注入
  int top_k_ = 3;
  int64_t next_id_ = 1;

  // 保护 next_id_ + HNSW add/search 并发（LruStore 自有锁）
  mutable std::mutex mutex_;

  // 幽灵向量观测计数器：try_hit() 在 mutex_ 之外自增（原实现是裸 size_t，
  // 与 ghost_stats()/try_rebuild_if_ghosty() 的读取构成数据竞争），故用原子量
  std::atomic<size_t> ghost_count_{0};
  std::atomic<size_t> total_search_{0};
};

}  // namespace ai_gateway
