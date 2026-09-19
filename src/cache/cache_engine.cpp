// 缓存协调器实现
#include "cache/cache_engine.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <string_view>

#include "cache/entity_tokens.h"
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
      entity_veto_(cache_cfg.entity_veto),
      hnsw_cfg_(index ? index->config() : HnswConfig{}),
      store_(std::move(store)),
      index_(std::move(index)),
      embed_fn_(std::move(embed_fn)) {}

CacheEngine::HitResult CacheEngine::try_hit(
    const std::string& user_message,
    const std::string& ns) {
  // 缓存 key: namespace:msg
  auto ns_key = ns.empty() ? user_message : ns + ":" + user_message;

  // 0. 索引尚未就绪（后台建图中）：语义检索退化为**暴力扫描**，而不是直接放弃。
  //    直接放弃的代价是这段时间**所有**请求都转成回源——一次几百毫秒、还真的花
  //    token；而线性扫描万条实测约 8ms（LruStore::scan_topk 持锁但只做点积）。
  //    判定走同一个 judge_candidates，所以降级期间的命中行为与索引正常时一致
  if (!index_ready_.load(std::memory_order_acquire)) {
    auto vec = embed_fn_(user_message, 5);
    if (vec.empty()) {
      // 连向量都算不出来：只剩"字符串完全相同"这一条路
      auto exact = store_->get_exact(ns_key);
      if (exact.has_value()) {
        LOG_INFO_SAMPLED("cache: HIT (exact, no embedding) ns={}",
                         ns.empty() ? "default" : ns);
        return HitResult{true, std::move(exact.value()), 1.0f};
      }
      return HitResult{};
    }
    degraded_searches_.fetch_add(1, std::memory_order_relaxed);
    auto scan = store_->scan_topk(vec, top_k_);
    std::vector<Candidate> degraded_candidates;
    degraded_candidates.reserve(scan.size());
    for (const auto& h : scan)
      degraded_candidates.push_back(Candidate{h.key, h.similarity});
    auto degraded_hit = judge_candidates(degraded_candidates, ns, user_message,
                                         /*count_ghosts=*/false);
    if (degraded_hit.hit) {
      LOG_INFO_SAMPLED("cache: HIT (degraded scan) key={} sim={:.3f}",
                       degraded_hit.key, degraded_hit.similarity);
      return degraded_hit;
    }
    LOG_INFO_SAMPLED("cache: MISS (degraded scan) top_sim={:.3f}",
                     scan.empty() ? 0.0f : scan[0].similarity);
    return HitResult{false, "", 0.0f, std::move(vec)};
  }

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

  // 3. 判定候选（阈值 → 命名空间 → 实体一致性否决 → 取回复）。
  //    判定逻辑与降级扫描路径共用同一个 judge_candidates，避免两条路径漂移
  std::vector<Candidate> candidates;
  candidates.reserve(results.size());
  for (const auto& r : results) candidates.push_back(Candidate{r.key, r.similarity});
  auto hit = judge_candidates(candidates, ns, user_message, /*count_ghosts=*/true);
  if (hit.hit) return hit;
  // 4. 未命中 带回 embedding 避免 cache_reply 重复计算
  LOG_INFO_SAMPLED("cache: MISS top_sim={:.3f} threshold={:.3f}",
                   results.empty() ? 0.0f : results[0].similarity, threshold_);
  return HitResult{false, "", 0.0f, std::move(vec)};
}

// 候选判定：索引路径与降级扫描路径共用（改动这里等于同时改两条路径）
CacheEngine::HitResult CacheEngine::judge_candidates(
    const std::vector<Candidate>& candidates, const std::string& ns,
    const std::string& user_message, bool count_ghosts) const {
  const std::string ns_prefix = ns.empty() ? "" : ns + ":";
  if (count_ghosts) total_search_.fetch_add(1, std::memory_order_relaxed);
  // 查询侧的实体标记只提取一次（候选侧每条条目各提一次）
  const EntityTokens query_entities =
      entity_veto_ ? extract_entity_tokens(user_message) : EntityTokens{};
  for (const auto& c : candidates) {
    if (c.similarity >= threshold_) {
      if (!ns.empty() && !c.key.starts_with(ns_prefix)) continue;
      if (entity_veto_) {
        auto source = store_->source_of(c.key);
        if (source.has_value()) {
          std::string_view text = *source;
          if (!ns.empty()) {
            const std::string prefix = ns + ":";
            if (text.starts_with(prefix)) text.remove_prefix(prefix.size());
          }
          if (entity_mismatch(query_entities, extract_entity_tokens(text))) {
            entity_veto_count_.fetch_add(1, std::memory_order_relaxed);
            LOG_INFO_SAMPLED(
                "cache: VETO by entity mismatch key={} sim={:.3f} ns={}",
                c.key, c.similarity, ns.empty() ? "default" : ns);
            continue;
          }
        }
      }
      auto cached = store_->get(c.key);
      if (cached.has_value()) {
        LOG_INFO_SAMPLED("cache: HIT key={} sim={:.3f} ns={}", c.key, c.similarity,
                         ns.empty() ? "default" : ns);
        return HitResult{true, std::move(cached.value()), c.similarity, {}, c.key};
      }
      if (count_ghosts) ghost_count_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return HitResult{};
}

void CacheEngine::cache_reply(const std::string& user_message,
                               const std::string& reply,
                               const std::vector<float>& cached_embedding,
                               const std::string& ns,
                               const std::string& sse_bytes) {
  std::lock_guard lock(mutex_);

  auto key = std::format("{}msg:{}", ns.empty() ? "" : ns + ":", next_id_);
  ++next_id_;

  // 原始查表键（namespace:user_message）：只作为条目的 source 字段保存，
  // 不再额外存一份回复——旧实现每条回复写两个条目，max_entries=10000 实际只装 5000 条
  auto ns_key = ns.empty() ? user_message : ns + ":" + user_message;

  if (cached_embedding.empty()) {
    store_->put_full(key, reply, sse_bytes, {});
    store_->set_source(key, std::move(ns_key));
    LOG_DEBUG("cache: stored key={} (no embedding) len={} sse={}", key,
              reply.size(), sse_bytes.size());
    return;
  }

  store_->put_full(key, reply, sse_bytes, cached_embedding);
  store_->set_source(key, ns_key);
  // 建图期间不写当前索引：那个索引马上会被后台构建结果交换掉，写进去等于丢。
  // 这一条与 rebuild_index_impl() 的补插是一对——少了补插就会漏条目，
  // 少了这个判断就会产生"索引里有、交换后消失"的假象
  if (index_ready_.load(std::memory_order_acquire)) {
    index_->add(next_id_ - 1, key, cached_embedding);
    indexed_keys_.insert(key);
  }

  // 不打印源消息原文（ns_key 含完整用户消息），只落长度（报告 M5）
  LOG_DEBUG("cache: stored key={} content_len={} src_len={} dims={} sse={}", key,
            reply.size(), ns_key.size(), cached_embedding.size(), sse_bytes.size());
}

CacheEngine::~CacheEngine() {
  // 等后台建图跑完再销毁成员：detach 会让线程访问已析构的 index_/store_
  if (rebuild_thread_.joinable()) rebuild_thread_.join();
}

void CacheEngine::rebuild_index() {
  rebuild_index_impl();
  index_ready_.store(true, std::memory_order_release);
}

void CacheEngine::rebuild_index_async() {
  // 已有建图在跑 → 空操作。用 compare_exchange 而不是 load+store：
  // 启动路径与 60s 定时器可能同时到达，`load` 后判空再置位会让两个线程都进来
  bool expected = false;
  if (!index_building_.compare_exchange_strong(expected, true,
                                               std::memory_order_acq_rel)) {
    LOG_DEBUG("cache: rebuild already in progress, skip");
    return;
  }
  if (rebuild_thread_.joinable()) rebuild_thread_.join();  // 复用前先回收上一轮
  // 先置"未就绪"再起线程：否则线程可能已经建完并置位，这里又把旧值覆盖回去
  index_ready_.store(false, std::memory_order_release);
  rebuild_thread_ = std::thread([this] {
    rebuild_index_impl();
    index_ready_.store(true, std::memory_order_release);
    index_building_.store(false, std::memory_order_release);
  });
}

void CacheEngine::rebuild_index_impl() {
  LOG_INFO("cache: rebuilding vector index...");

  // 0. 补算缺失的向量（cache.store_vectors=false 落盘的缓存只有文本）。
  //    必须在建图之前做——建图要靠向量。编码是重活（1 万条实测约 20s），
  //    而本函数正是后台建图线程的入口，放这里不会占住请求路径
  size_t encoded = 0;
  store_->for_each_missing_embedding(
      [&](const std::string& key, const std::string& source) {
        // source 形如 "ns<hash>:<原始消息>"：编码要用**消息部分**，
        // 与写入时保持一致（带上前缀会让同一个问题算出不同向量）。
        // 前缀形状是 16 位十六进制 + ':'，按形状识别而不是找第一个冒号——
        // 无 ns 时 source 就是原文，原文里完全可能有冒号（比如 URL）
        std::string_view text = source;
        if (source.size() > 17 && source[16] == ':') {
          bool hex = true;
          for (int i = 0; i < 16; ++i) {
            if (!std::isxdigit(static_cast<unsigned char>(source[i]))) {
              hex = false;
              break;
            }
          }
          if (hex) text = std::string_view(source).substr(17);
        }
        auto vec = embed_fn_(std::string(text), 5);
        if (!vec.empty() && store_->set_embedding(key, std::move(vec))) ++encoded;
      });
  if (encoded > 0)
    LOG_INFO("cache: re-encoded {} vectors from source (store_vectors=false)",
             encoded);

  // 1. 锁外构建新索引：建图要对每个条目跑一次 O(ef_construction) 搜索，
  //    整个过程持 mutex_ 会让检索与写入全部阻塞，因此先构建、再交换。
  auto new_index_ptr = std::make_shared<HnswIndex>(hnsw_cfg_);
  auto& new_idx = *new_index_ptr;

  // next_id_ 不能由"存活条目数 + 1"派生：TTL 是逐条过期的，存活键的空间里存在空洞，
  // 退回的编号会让新条目复用仍然存活的键 msg:N（旧索引节点仍指向该键，于是用旧问题
  // 的向量检索会命中旧节点、却取回新问题的答案）。这里取"存活键里 max(N) + 1"，
  // 保证新编号只前进、不与任何存活键冲突。
  int64_t max_id = 0;
  std::unordered_set<std::string> built_keys;  // 本轮已进索引的 key
  store_->for_each_embedding(
      [&](const std::string& key, const std::vector<float>& emb) {
        int64_t id = 0;
        if (parse_msg_id(key, id) && id > max_id) max_id = id;
        new_idx.add(static_cast<int>(id), key, emb);
        built_keys.insert(key);
      });
  const size_t built = built_keys.size();

  // 2. 补插构建期间新写入的条目。
  //    建图要跑几十秒到上百秒，这段时间里 cache_reply 因为 index_ready_=false
  //    而**不写索引**（它写的那个 index_ 马上会被交换掉，写进去等于丢）；
  //    如果不在这里补，这批条目就是"存储里有、索引里没有"的幽灵向量，
  //    要等下一轮重建才可能被检索到。启动场景没有写入所以看不出来，
  //    运行中的重建（幽灵率/过期清理触发）必然会遇到
  size_t backfilled = 0;
  store_->for_each_embedding(
      [&](const std::string& key, const std::vector<float>& emb) {
        if (built_keys.count(key)) return;
        int64_t id = 0;
        if (parse_msg_id(key, id) && id > max_id) max_id = id;
        new_idx.add(static_cast<int>(id), key, emb);
        built_keys.insert(key);
        ++backfilled;
      });

  // 3. 短临界区交换：与 try_hit 的取副本、cache_reply 的 add 互斥
  //    （日志用的两个值都在临界区内取出，避免锁外读 next_id_ 的竞争）
  size_t vectors = 0;
  int64_t next_id = 0;
  {
    std::lock_guard lock(mutex_);
    index_ = std::move(new_index_ptr);
    next_id_ = std::max(next_id_, max_id + 1);  // 单调不回退
    indexed_keys_ = std::move(built_keys);      // 交接给索引态，供去重
    vectors = index_->size();
    next_id = next_id_;
  }

  LOG_INFO("cache: index rebuilt, {} vectors ({} built + {} backfilled), next_id={}",
           vectors, built, backfilled, next_id);
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
    // 走异步：重建要跑几十秒（万条量级），同步会把 60s 定时器线程占满，
    // 连带拖住同线程的 purge/save
    // （rebuild_index() 自带 mutex_：这里不能先取锁再调用，同线程递归加锁会死锁）
    rebuild_index_async();
    ghost_count_.store(0, std::memory_order_relaxed);
    total_search_.store(0, std::memory_order_relaxed);
  }
}

}  // namespace ai_gateway
