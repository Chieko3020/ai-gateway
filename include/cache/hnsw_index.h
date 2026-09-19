// HNSW (Hierarchical Navigable Small World) 近似最近邻搜索
// 简化实现，直接取最近 M 个 0.5 概率 x16 层
// 参考论文: Malkov // 参考: Malkov & Yashunin (2018) Yashunin (2018)
// 构建参数: M=16, ef_construction=100, ef_search=50
// 复杂度: 搜索 O(log N), 构建 O(N log N)
// 精度: 10K 512d ≈ 99% recall vs 暴力搜索

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <shared_mutex>
#include <string>
#include <vector>

#include "common/logger.h"

namespace ai_gateway {

    // add(vec):   随机层级 逐层找到最近邻 建双向边
    // search(q):  顶层贪心下钻 0 层束搜索 top-K 余弦相似
    
    //   入口点（最高层节点）
    //   Layer 2:  节点少，边长距，粗导航
    //   Layer 1:  较密
    //   Layer 0:  全节点，最密，精细搜索

    // add() 插入节点，层级分配(几何分布)，多层级建边
    // search() 顶层下钻 0 层 ef_search 束搜索 cosine 排序返回
    // search_layer() 一层内的束搜索，visited 向量剪枝
    // l2() 欧氏距离（搜索用）
    // cosine() 余弦相似度（结果排序用）


struct HnswConfig {
  int dim = 512;
  int M = 16;
  int ef_construction = 100;
  int ef_search = 50;
};

struct HnswResult {
  int id = -1;
  float similarity = 0.0f;
  std::string key;
};

struct HnswDistNode {
  int id;
  float dist;
  bool operator<(const HnswDistNode& o) const { return dist < o.dist; }
  bool operator>(const HnswDistNode& o) const { return dist > o.dist; }
};

class HnswIndex {
 public:
  explicit HnswIndex(const HnswConfig& cfg = {}) : cfg_(cfg), gen_(42), gen_dis_(0.0, 1.0) {}

  void add(int id, const std::string& key, const std::vector<float>& embedding) {
    std::lock_guard lock(mutex_);
    if (static_cast<int>(embedding.size()) != cfg_.dim) {
      // 原来静默 return：维度不符时索引会悄悄变空、缓存退化为永不命中且无任何线索
      LOG_WARN("hnsw: skip vector dim={} (index dim={}), key={}",
               embedding.size(), cfg_.dim, key);
      return;
    }

    // 层数按论文的几何分布：level = floor(-ln(U) * mL)，mL = 1/ln(M)
    // 使 P(level >= 1) ≈ 1/M（M=16 时约 6.25%），保持高层稀疏、专司粗导航。
    // 原实现"每层 50% 概率"会让近一半节点进入高层，导航层失去稀疏性。
    int level = 0;
    {
      const double u = std::max(1e-12, static_cast<double>(gen_dis_(gen_)));
      const double ml = 1.0 / std::log(static_cast<double>(cfg_.M));
      level = static_cast<int>(-std::log(u) * ml);
      if (level > 16) level = 16;
    }

    Node node;
    node.level = level;
    node.vec = embedding;
    node.key = key;
    node.id = id;
    node.neighbors.resize(level + 1);
    int cur = static_cast<int>(nodes_.size());
    nodes_.push_back(std::move(node));

    if (entry_point_ == -1) {
      entry_point_ = cur;
      max_level_ = level;
      return;
    }

    int ep = entry_point_;
    for (int lc = max_level_; lc > level; --lc)
      ep = search_layer(embedding, ep, 1, lc).top().id;

    // 对齐论文的连接上限：高层 M_max = M，0 层 M_max0 = 2M。
    // 原实现放大到 2M / 4M 是"以密度换召回"的补偿；补上启发式剪枝与满时收缩后，
    // 连接质量由剪枝保证，可以回到论文取值（图更稀疏、构建与检索更省）。
    const int M_max = cfg_.M;
    const int M_max0 = 2 * cfg_.M;
    for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
      auto candidates = search_layer(embedding, ep, cfg_.ef_construction, lc);
      int max_conn = (lc == 0) ? M_max0 : M_max;

      std::vector<HnswDistNode> cands;
      while (!candidates.empty()) {
        cands.push_back(candidates.top());
        candidates.pop();
      }

      // 用启发式选择本节点的邻居（多样性筛选），而不是直接取最近 max_conn 个
      std::vector<int> selected = selectNeighborsHeuristic(cands, max_conn);
      nodes_[cur].neighbors[lc] = selected;

      // 建立双向连接；邻居已满时收缩重选，而不是直接放弃反向边
      for (int nei : selected) {
        if (nei == cur) continue;
        connectWithShrink(cur, nei, lc, max_conn);
      }

      if (!cands.empty()) ep = cands.front().id;
    }
    if (level > max_level_) {
      max_level_ = level;
      entry_point_ = cur;
    }
  }

  std::vector<HnswResult> search(const std::vector<float>& query, int k = 10) {
    std::shared_lock lock(mutex_);
    if (entry_point_ == -1 || nodes_.empty()) return {};
    if (static_cast<int>(query.size()) != cfg_.dim) return {};

    int ep = entry_point_;
    for (int lc = max_level_; lc > 0; --lc)
      ep = search_layer(query, ep, 1, lc).top().id;

    auto candidates = search_layer(query, ep, cfg_.ef_search, 0);

    // 转换为按余弦相似度排序的结果
    std::vector<HnswResult> results;
    while (!candidates.empty() && static_cast<int>(results.size()) < k) {
      auto& top = candidates.top();
      results.push_back({top.id, cosine(query, nodes_[top.id].vec), nodes_[top.id].key});
      candidates.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const HnswResult& a, const HnswResult& b) { return a.similarity > b.similarity; });
    return results;
  }

  size_t size() const {
    std::shared_lock lock(mutex_);
    return nodes_.size();
  }

  // 构建参数：供持有者（CacheEngine）在重建索引时沿用同一套配置
  const HnswConfig& config() const { return cfg_; }

  // 持久化：索引由 LruStore 管理（for_each_embedding 重建），不需要独立 save/load
  void save(const std::string&) const {}
  void load(const std::string&) {}

 private:
  struct Node {
    int level = 0;
    std::vector<float> vec;
    std::string key;
    int id = 0;
    std::vector<std::vector<int>> neighbors;
  };

  using MinHeap = std::priority_queue<HnswDistNode, std::vector<HnswDistNode>, std::greater<HnswDistNode>>;
  using MaxHeap = std::priority_queue<HnswDistNode>;

  // 搜索用的 visited 标记：线程本地复用 + epoch 版本号
  // （每次搜索只递增 epoch，无需清零整个数组；thread_local 保证并发读搜索互不干扰）
  struct VisitScratch {
    std::vector<uint32_t> tag;
    uint32_t epoch = 0;
  };
  static VisitScratch& visit_scratch() {
    thread_local VisitScratch s;
    return s;
  }

  // 启发式邻居选择（论文 Algorithm 4 SELECT-NEIGHBORS-HEURISTIC）：
  // 候选按到查询点的距离升序尝试加入；若某候选到"已选邻居"的距离比它到查询点更近，
  // 说明它与已选邻居方向重复（冗余），丢弃它——以此让邻居方向分散、维持图连通。
  // 副作用：就地按距离升序排序传入的候选列表。
  std::vector<int> selectNeighborsHeuristic(std::vector<HnswDistNode>& candidates,
                                            int M) {
    std::sort(candidates.begin(), candidates.end(),
              [](const HnswDistNode& a, const HnswDistNode& b) { return a.dist < b.dist; });
    std::vector<int> selected;
    selected.reserve(static_cast<size_t>(M));
    for (const auto& cand : candidates) {
      if (static_cast<int>(selected.size()) >= M) break;
      bool keep = true;
      for (int s : selected) {
        if (l2(nodes_[cand.id].vec, nodes_[s].vec) < cand.dist) {
          keep = false;
          break;
        }
      }
      if (keep) selected.push_back(cand.id);
    }
    return selected;
  }

  // 邻居已满时的收缩（论文 Algorithm 1 第 12-13 行）：
  // 把"新节点 + 现有邻居"合并后按启发式重选 max_conn 个，被淘汰的边断开。
  // 原实现是"满了就不加反向边"，会让后插入的节点没有入边、在检索中不可达
  // （实测自检索 top-1 仅 30.9% 的直接原因）。
  void connectWithShrink(int cur, int nei, int lc, int max_conn) {
    auto& nbrs = nodes_[nei].neighbors[lc];
    for (int x : nbrs) {
      if (x == cur) return;  // 已连接，无需重复
    }
    if (static_cast<int>(nbrs.size()) < max_conn) {
      nbrs.push_back(cur);
      return;
    }
    std::vector<HnswDistNode> cands;
    cands.reserve(nbrs.size() + 1);
    for (int x : nbrs) cands.push_back({x, l2(nodes_[nei].vec, nodes_[x].vec)});
    cands.push_back({cur, l2(nodes_[nei].vec, nodes_[cur].vec)});
    nbrs = selectNeighborsHeuristic(cands, max_conn);
  }

  MinHeap search_layer(const std::vector<float>& query, int ep, int ef, int lc) {
    // visited 标记复用：线程本地缓冲 + epoch 版本号，避免每次调用分配并清零 O(N)
    auto& sc = visit_scratch();
    if (sc.tag.size() != nodes_.size()) {
      sc.tag.assign(nodes_.size(), 0);
      sc.epoch = 0;
    }
    if (++sc.epoch == 0) {  // epoch 回绕：清零后重新开始，保证标记有效
      std::fill(sc.tag.begin(), sc.tag.end(), 0);
      sc.epoch = 1;
    }
    const uint32_t cur_epoch = sc.epoch;
    sc.tag[ep] = cur_epoch;
    MaxHeap result;
    MinHeap candidates;

    float d = l2(query, nodes_[ep].vec);
    candidates.push({ep, d});
    result.push({ep, d});

    while (!candidates.empty()) {
      auto cur = candidates.top(); candidates.pop();
      if (static_cast<int>(result.size()) >= ef && cur.dist > result.top().dist) break;

      for (int nei_id : nodes_[cur.id].neighbors[lc]) {
        if (sc.tag[nei_id] == cur_epoch) continue;
        sc.tag[nei_id] = cur_epoch;

        float nd = l2(query, nodes_[nei_id].vec);
        if (static_cast<int>(result.size()) < ef || nd < result.top().dist) {
          candidates.push({nei_id, nd});
          result.push({nei_id, nd});
          if (static_cast<int>(result.size()) > ef) result.pop();
        }
      }
    }

    MinHeap out;
    while (!result.empty()) {
      out.push(result.top()); result.pop();
    }
    return out;
  }

  static float l2(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
      float d = a[i] - b[i];
      sum += d * d;
    }
    return std::sqrt(sum);
  }

 public:
  // 余弦相似度：HNSW 检索与 LruStore 的降级扫描共用同一套定义
  static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    float sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8f);
    return std::clamp(sim, -1.0f, 1.0f);
  }

  HnswConfig cfg_;
  std::vector<Node> nodes_;
  int entry_point_ = -1;
  int max_level_ = 0;
  std::mt19937 gen_;
  std::uniform_real_distribution<float> gen_dis_;
  mutable std::shared_mutex mutex_;
};

}  // namespace ai_gateway
