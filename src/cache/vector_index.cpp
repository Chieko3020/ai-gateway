// 向量索引实现：线性扫描 + SIMD 加速内积计算
#include "vector_index.h"

#include <algorithm>
#include <cmath>
#include <queue>

#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace ai_gateway {

float VectorIndex::dot_product(const std::vector<float>& a,
                                const std::vector<float>& b) {
  const float* pa = a.data();
  const float* pb = b.data();
  size_t n = a.size();
  float sum = 0.0f;
  size_t i = 0;

#ifdef __AVX2__
  // 一次处理8个floats，使用FMA指令加速内积计算
  // 8 floats per iteration
  __m256 vsum = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(pa + i);
    __m256 vb = _mm256_loadu_ps(pb + i);
    vsum = _mm256_fmadd_ps(va, vb, vsum);  // vsum += va * vb
  }
  // Horizontal sum of 8 floats
  __m128 hi = _mm256_extractf128_ps(vsum, 1);
  __m128 lo = _mm256_castps256_ps128(vsum);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum += _mm_cvtss_f32(sum128);

// cpu兼容
#elif defined(__SSE4_1__)
  // 4 floats per iteration
  __m128 vsum = _mm_setzero_ps();
  for (; i + 4 <= n; i += 4) {
    __m128 va = _mm_loadu_ps(pa + i);
    __m128 vb = _mm_loadu_ps(pb + i);
    vsum = _mm_add_ps(vsum, _mm_mul_ps(va, vb));
  }
  // Horizontal sum of 4 floats
  __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
  __m128 sums = _mm_add_ps(vsum, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  sum += _mm_cvtss_f32(sums);
#endif

  // 余数scalar标量计算
  for (; i < n; ++i) {
    sum += pa[i] * pb[i];
  }
  return sum;
}

void VectorIndex::add(int64_t id, const std::string& key,
                       const std::vector<float>& vec) {
  auto it = id_to_idx_.find(id);
  if (it != id_to_idx_.end()) {
    keys_[it->second] = key;
    vecs_[it->second] = vec;
    return;
  }
  id_to_idx_[id] = ids_.size();
  ids_.push_back(id);
  keys_.push_back(key);
  vecs_.push_back(vec);
}

std::vector<SearchResult> VectorIndex::search(
    const std::vector<float>& query, int k) const {
  if (vecs_.empty()) return {};

  // 用最小堆维护 Top-K 堆顶是第 K 大的最小值
  using Item = std::pair<float, size_t>;  // (similarity, index)
  auto cmp = [](const Item& a, const Item& b) { return a.first > b.first; };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp)> heap(cmp);

  for (size_t i = 0; i < vecs_.size(); ++i) {
    float sim = dot_product(query, vecs_[i]);
    if (heap.size() < static_cast<size_t>(k)) {
      heap.emplace(sim, i);
    } else if (sim > heap.top().first) {
      heap.pop();
      heap.emplace(sim, i);
    }
  }

  // 从堆中取出结果（降序排列）
  std::vector<SearchResult> results(heap.size());
  for (int i = static_cast<int>(heap.size()) - 1; i >= 0; --i) {
    auto& top = heap.top();
    results[i].key = keys_[top.second];
    results[i].similarity = top.first;
    heap.pop();
  }

  return results;
}

void VectorIndex::remove(int64_t id) {
  auto it = id_to_idx_.find(id);
  if (it == id_to_idx_.end()) return;
  size_t idx = it->second;
  // 与末尾交换后 pop（O(1) 删除），更新被交换元素的映射
  size_t last = ids_.size() - 1;
  if (idx != last) {
    ids_[idx] = ids_[last];
    keys_[idx] = std::move(keys_[last]);
    vecs_[idx] = std::move(vecs_[last]);
    id_to_idx_[ids_[idx]] = idx;
  }
  ids_.pop_back();
  keys_.pop_back();
  vecs_.pop_back();
  id_to_idx_.erase(it);
}

void VectorIndex::for_each(
    const std::function<void(int64_t, const std::string&,
                              const std::vector<float>&)>& fn) const {
  for (size_t i = 0; i < ids_.size(); ++i) {
    fn(ids_[i], keys_[i], vecs_[i]);
  }
}

// 二进制格式：
//   [int64_t count][for each: int64_t id, int32_t key_len, char[key_len], int32_t dim, float[dim]]
bool VectorIndex::save(const std::string& path) const {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;

  int64_t count = static_cast<int64_t>(ids_.size());
  if (fwrite(&count, sizeof(count), 1, f) != 1) { fclose(f); return false; }

  for (size_t i = 0; i < ids_.size(); ++i) {
    int64_t id = ids_[i];
    if (fwrite(&id, sizeof(id), 1, f) != 1) { fclose(f); return false; }

    int32_t key_len = static_cast<int32_t>(keys_[i].size());
    if (fwrite(&key_len, sizeof(key_len), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(keys_[i].data(), 1, key_len, f) != static_cast<size_t>(key_len)) {
      fclose(f); return false;
    }

    int32_t dim = static_cast<int32_t>(vecs_[i].size());
    if (fwrite(&dim, sizeof(dim), 1, f) != 1) { fclose(f); return false; }
    if (fwrite(vecs_[i].data(), sizeof(float), dim, f) != static_cast<size_t>(dim)) {
      fclose(f); return false;
    }
  }

  fclose(f);
  return true;
}

bool VectorIndex::load(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;

  int64_t count = 0;
  if (fread(&count, sizeof(count), 1, f) != 1) { fclose(f); return false; }

  ids_.clear(); keys_.clear(); vecs_.clear(); id_to_idx_.clear();
  ids_.reserve(count); keys_.reserve(count); vecs_.reserve(count);

  for (int64_t i = 0; i < count; ++i) {
    int64_t id;
    if (fread(&id, sizeof(id), 1, f) != 1) break;
    ids_.push_back(id);

    int32_t key_len;
    if (fread(&key_len, sizeof(key_len), 1, f) != 1) break;
    if (key_len < 0 || key_len > 65536) break;  // 合理性校验
    std::string key(key_len, '\0');
    if (fread(key.data(), 1, key_len, f) != static_cast<size_t>(key_len)) break;
    keys_.push_back(std::move(key));
    id_to_idx_[id] = ids_.size() - 1;

    int32_t dim;
    if (fread(&dim, sizeof(dim), 1, f) != 1) break;
    if (dim <= 0 || dim > 10000) break;          // 恶意/损坏 dim 防护
    std::vector<float> vec(dim);
    if (fread(vec.data(), sizeof(float), dim, f) != static_cast<size_t>(dim)) break;
    vecs_.push_back(std::move(vec));
  }

  fclose(f);
  return true;
}

}  // namespace ai_gateway
