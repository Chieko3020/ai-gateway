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
#include <thread>
#include <unordered_set>
#include <vector>

#include <functional>

#include "common/config.h"
#include "cache/lru_store.h"
#include "cache/hnsw_index.h"

namespace ai_gateway {

class CacheEngine {
 public:
  // 向量化回调：只传文本与超时。
  // 旧签名还传 url/api_key/model 三个参数，它们对应的配置段本轮已删除
  // （模型路径与维度改由 EmbeddingConfig 驱动进程内 ONNX，报告 M15）
  using EmbedFn =
      std::function<std::vector<float>(const std::string& text, int timeout)>;

  CacheEngine(const EmbeddingConfig& emb_cfg,
              const CacheConfig& cache_cfg,
              std::shared_ptr<LruStore> store,
              std::shared_ptr<HnswIndex> index,
              EmbedFn embed_fn);

  // 析构要回收后台建图线程：若建图仍在进行会**等它跑完**再销毁成员
  // （detach 会让线程访问已析构的成员，是 UAF）
  ~CacheEngine();

  // 尝试从缓存中获取回复
  // user_message: 用户最新一条消息文本
  // 返回命中结果：hit=true 表示找到缓存，hit=false 表示未命中
  // 未命中时 embedding 字段带回已计算的向量（避免 cache_reply 重复调用 API）
  struct HitResult {
    bool hit = false;
    std::string reply;
    float similarity = 0.0f;
    std::vector<float> embedding;
    // 命中条目的键。流式命中要靠它把当初记下的原始 SSE 字节取回来回放；
    // 降级路径（embedding 失败后走精确匹配）拿不到条目键，这里留空 —— 流式
    // 命中随即退回回源，而不是把非流式 JSON 当流发出去
    std::string key;
  };
  HitResult try_hit(const std::string& user_message,
                     const std::string& ns = "");

  // 将 LLM 回复存入缓存，传入已计算的 embedding 向量（避免重复 API 调用）。
  // sse_bytes 非空时一并存入：那是**上游原始 SSE 字节**，供流式命中回放
  // （见 LruStore::put_full）。非流式路径传空
  void cache_reply(const std::string& user_msg, const std::string& reply,
                   const std::vector<float>& embedding,
                   const std::string& ns = "",
                   const std::string& sse_bytes = {});

  // 读取条目关联的原始 SSE 字节（流式命中回放用）。key 取自 HitResult::key。
  // 只读、不计命中——命中判定已经由 try_hit 记过账
  std::optional<std::string> sse_of(const std::string& key) const {
    return store_->sse_of(key);
  }

  // 从 LruStore 重建向量索引（缓存恢复后调用）——**同步**版本，调用线程会被
  // 占住整段建图时间。产品路径请用 rebuild_index_async()，本函数保留给
  // 单测与"确实需要阻塞等待"的场景。
  // 线程安全：函数内部自行加锁，调用方无需（也不得）持 mutex_——
  // 持锁调用会在同一线程递归加锁，直接死锁。
  void rebuild_index();

  // 异步重建：立刻返回，建图在后台线程进行，构建完成后才原子交换。
  //
  // 为什么需要它：建图的耗时随规模超线性增长（512 维**随机向量** 1 万条实测 117s；
  // 真实 embedding 1 万条约 7s——评估建图性能必须用带簇结构的数据，见 README 实测性能章）。
  // 同步版本会让"启动"与"运行中重建"都占住调用线程；启动场景下这段时间进程
  // 还没 listen，客户端连接会被直接拒绝——这就是 L33 的"建索引阻塞窗口"。
  //
  // 建图期间 is_index_ready() 为 false，try_hit 退化为**线性扫描**
  // （LruStore::scan_topk + judge_candidates，见 cache_engine.cpp:61-89）；
  // 只有 embedding 不可用时才退化为精确匹配。
  // 已有建图在跑时重复调用是**空操作**（不会并发建图）。
  void rebuild_index_async();

  // 索引是否已就绪。false = 正在后台建图，语义检索暂时不可用
  bool index_ready() const {
    return index_ready_.load(std::memory_order_acquire);
  }
  // 是否有后台建图正在进行
  bool index_building() const {
    return index_building_.load(std::memory_order_acquire);
  }

  // 活动索引中的向量数。
  // 调用方不要再持有构造时传入的索引指针用于观察：rebuild_index() 会把 index_
  // 换成新对象，旧对象随即失效且不再被引擎使用（原实现据此打印的向量数恒为旧值）。
  size_t index_size() const;

  // 配置访问
  float threshold() const { return threshold_; }
  int top_k() const { return top_k_; }
  bool entity_veto_enabled() const { return entity_veto_; }

  // 实体一致性否决的观测计数：命中候选的相似度过了阈值、但因为实体不一致
  // （数字/大写缩略语/混合标识符对不上）被否决的次数。
  // 单列计数器而不是只落日志：这个数字直接回答"否决规则有没有在工作、
  // 会不会把正常流量也拦掉"（阈值调优时它是主要输入）
  size_t entity_veto_count() const {
    return entity_veto_count_.load(std::memory_order_relaxed);
  }

  // 索引不可用时走了**暴力扫描**降级的查询次数。单列计数：
  // 建图窗口内命中率不再断崖，靠的就是这条路径，它生效与否必须看得见
  size_t degraded_searches() const {
    return degraded_searches_.load(std::memory_order_relaxed);
  }

  // 幽灵向量观测：返回"搜到但取不到"的比例，用于判断是否需要 rebuild_index
  std::pair<int, size_t> ghost_stats() const;

  // 如果幽灵率超过阈值，自动重建索引（在 60s 定时器中调用）
  void try_rebuild_if_ghosty();

 private:
  EmbeddingConfig emb_cfg_;
  // 构造时传入的索引构建参数：rebuild_index() 用它重建，
  // 避免重建出来的索引悄悄退回 HnswConfig 的默认值（与首个索引不一致）
  HnswConfig hnsw_cfg_;

  // ---- 异步建图状态 ---------------------------------------------------------
  // 索引是否已就绪（false = 后台建图中）。用 acquire/release 而不是 relaxed：
  // 读取方（try_hit / cache_reply）要靠它决定"能不能用 index_"，必须与
  // rebuild 线程的交换动作建立 happens-before
  std::atomic<bool> index_ready_{true};
  std::atomic<bool> index_building_{false};
  std::thread rebuild_thread_;  // 析构时 join（不能 detach：会访问已销毁成员）

  // ⚠️ 死成员（注释与实现不符，待清理）：建图补插实际用的是 rebuild_index_impl()
  // 的**局部** built_keys；本成员只写不读（写点 cache_engine.cpp:200/314，全仓无读取点）。
  // 已知副作用：第二次 for_each_embedding 快照（:296）到取锁（:311）之间写入的条目仍会漏索引。
  std::unordered_set<std::string> indexed_keys_;

  // 真正的重建实现（构建 → 补插 → 交换）。由同步/异步两个入口共用
  void rebuild_index_impl();

  // 判定用的候选：索引路径与降级扫描路径统一成这一种形态
  struct Candidate {
    std::string key;
    float similarity = 0.0f;
  };

  // 从候选集判定命中：阈值 → 命名空间 → 实体一致性否决 → 取回复。
  // **两条路径共用**（HNSW 检索 / 索引不可用时的暴力扫描）：判定必须逐字一致，
  // 否则会出现"索引正常时命中、降级时命中不了"这类最难查的漂移
  // count_ghosts：只有索引路径计"搜到但取不到"（暴力扫描没有幽灵问题）
  HitResult judge_candidates(const std::vector<Candidate>& candidates,
                             const std::string& ns,
                             const std::string& user_message,
                             bool count_ghosts) const;

  // 走了降级扫描的查询次数（见 degraded_searches()）
  std::atomic<size_t> degraded_searches_{0};
  std::shared_ptr<LruStore> store_;

  // 索引由本引擎独占持有（调用方不应继续持有同一个 shared_ptr 用于观察）。
  // 读写约定：所有对 index_ 的读（含取出副本）与写（rebuild_index 交换）都在
  // mutex_ 临界区内完成；检索方在临界区内取一份 shared_ptr 副本后在临界区外使用，
  // 由引用计数保证检索期间索引对象不被析构。
  std::shared_ptr<HnswIndex> index_;

  EmbedFn embed_fn_;
  float threshold_ = 0.0f;  // 由 CacheConfig 注入
  // 实体一致性否决开关（CacheConfig::entity_veto）。关闭时退回"纯阈值"判定，
  // 便于把"否决规则带来的收益/损失"做成 A/B 对照（评测脚本按这个口径跑）
  bool entity_veto_ = true;
  int top_k_ = 3;
  int64_t next_id_ = 1;

  // 保护 next_id_ + HNSW add/search 并发（LruStore 自有锁）
  mutable std::mutex mutex_;

  // 幽灵向量观测计数器：try_hit()/judge_candidates() 在 mutex_ 之外自增
  // （mutable：判定函数是 const，但观测计数本就该能改）（原实现是裸 size_t，
  // 与 ghost_stats()/try_rebuild_if_ghosty() 的读取构成数据竞争），故用原子量
  mutable std::atomic<size_t> ghost_count_{0};
  mutable std::atomic<size_t> total_search_{0};
  // 实体否决计数：同样是锁外自增的热路径计数器，用原子量
  mutable std::atomic<size_t> entity_veto_count_{0};
};

}  // namespace ai_gateway
