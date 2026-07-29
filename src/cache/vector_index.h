// 向量索引：存储 embedding 向量 + 支持 Top-K 余弦相似度搜索
//
// 内部实现：暴力搜索（线性扫描 + AVX2 SIMD 加速）——零外部依赖
// 余弦相似度 = dot(A,B) / (|A| * |B|)
// 由于存储时向量已归一化，搜索时直接计算内积即可（等价于余弦相似度）
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ai_gateway {

struct SearchResult {
  std::string key;       // 对应的缓存 key
  float similarity = 0;  // 余弦相似度 [0, 1]
};

class VectorIndex {
 public:
  // 添加向量：id 和 key 对应，dim 必须和已有向量一致
  void add(int64_t id, const std::string& key,
           const std::vector<float>& vec);

  // 搜索 Top-K 最相似向量，返回 (key, similarity) 列表（按相似度降序）
  std::vector<SearchResult> search(const std::vector<float>& query,
                                   int k) const;

  // 删除指定 id 的向量；不存在时无操作
  void remove(int64_t id);

  // 获取向量数量
  size_t size() const { return ids_.size(); };

  // 遍历所有向量用于索引重建
  void for_each(const std::function<void(int64_t, const std::string&,
                                          const std::vector<float>&)>& fn) const;

  // 持久化：保存到文件（二进制格式）
  bool save(const std::string& path) const;

  // 持久化：从文件加载；失败返回 false
  bool load(const std::string& path);

 private:
  // 计算两个向量的内积（向量已归一化时等于余弦相似度）
  static float dot_product(const std::vector<float>& a,
                           const std::vector<float>& b);

  std::vector<int64_t> ids_;
  std::vector<std::string> keys_;
  std::vector<std::vector<float>> vecs_;
};

}  // namespace ai_gateway
