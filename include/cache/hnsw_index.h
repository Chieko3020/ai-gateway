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
#include <string>
#include <vector>

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
    if (static_cast<int>(embedding.size()) != cfg_.dim) return;

    int level = 0;
    while (gen_dis_(gen_) < 0.5f && level < 16) ++level;

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

    int M_max = 2 * cfg_.M;
    int M_max0 = 2 * M_max;
    for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
      auto candidates = search_layer(embedding, ep, cfg_.ef_construction, lc);
      int max_conn = (lc == 0) ? M_max0 : M_max;
      // 取最近 max_conn 个邻居并添加双向链接
      std::vector<HnswDistNode> sorted;
      while (!candidates.empty()) { sorted.push_back(candidates.top()); candidates.pop(); }
      int taken = std::min(max_conn, static_cast<int>(sorted.size()));
      nodes_[cur].neighbors[lc].reserve(taken);
      for (int i = 0; i < taken; ++i) {
        int nei = sorted[i].id;
        if (nei == cur) continue;
        nodes_[cur].neighbors[lc].push_back(nei);
        // 若未超过限制则添加反向边
        int rev_max = (lc == 0) ? M_max0 : M_max;
        if (static_cast<int>(nodes_[nei].neighbors[lc].size()) < rev_max)
          nodes_[nei].neighbors[lc].push_back(cur);
      }
      ep = sorted[0].id;
    }
    if (level > max_level_) {
      max_level_ = level;
      entry_point_ = cur;
    }
  }

  std::vector<HnswResult> search(const std::vector<float>& query, int k = 10) {
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

  size_t size() const { return nodes_.size(); }

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

  MinHeap search_layer(const std::vector<float>& query, int ep, int ef, int lc) {
    std::vector<bool> visited(nodes_.size(), false);
    visited[ep] = true;
    MaxHeap result;
    MinHeap candidates;

    float d = l2(query, nodes_[ep].vec);
    candidates.push({ep, d});
    result.push({ep, d});

    while (!candidates.empty()) {
      auto cur = candidates.top(); candidates.pop();
      if (static_cast<int>(result.size()) >= ef && cur.dist > result.top().dist) break;

      for (int nei_id : nodes_[cur.id].neighbors[lc]) {
        if (visited[nei_id]) continue;
        visited[nei_id] = true;

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
};

}  // namespace ai_gateway
