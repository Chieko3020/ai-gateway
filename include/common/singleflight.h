// 请求合并（singleflight）：相同 key 的并发请求只放一个去后端，其余等待共享结果
//
// 两层匹配：
//   1. 精确字符串匹配 O(1)：key = namespace:user_message
//   2. 向量语义匹配 O(N)：cosine(embedding, in_flight.embedding) >= 0.95
//
// 失败不共享：LLM 返回错误时从表移除，不 set promise，等待者超时后各自重试
#pragma once

#include <algorithm>
#include <cmath>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_gateway {

class Singleflight {
 public:
  // 尝试合并到已有进行中请求。返回 shared_future 供等待，nullopt 表示无匹配
  std::optional<std::shared_future<std::string>> try_merge(
      const std::string& key, const std::vector<float>& embedding);

  // 插入新的进行中请求（try_merge 返回 nullopt 后调用）
  std::shared_ptr<std::promise<std::string>> insert(
      const std::string& key, const std::vector<float>& embedding);

  // 完成：set_value 并移除
  void complete(const std::string& key, const std::string& result);

  // 取消：不移除，不 set_value（让等待者超时后各自重试）
  void cancel(const std::string& key);

 private:
  static float cosine(const std::vector<float>& a, const std::vector<float>& b);

  struct Entry {
    std::vector<float> embedding;
    std::shared_ptr<std::promise<std::string>> promise;
    std::shared_future<std::string> future;
  };

  std::unordered_map<std::string, Entry> inflight_;
  std::mutex mutex_;
};

// ── 实现 ──────────────────────────────────────────────────────

inline float Singleflight::cosine(const std::vector<float>& a,
                                   const std::vector<float>& b) {
  float dot = 0.0f, na = 0.0f, nb = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  float sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8f);
  return std::clamp(sim, -1.0f, 1.0f);
}

inline std::optional<std::shared_future<std::string>> Singleflight::try_merge(
    const std::string& key, const std::vector<float>& embedding) {
  std::lock_guard lock(mutex_);

  // 第一层：精确字符串匹配
  auto it = inflight_.find(key);
  if (it != inflight_.end()) return it->second.future;

  // 第二层：向量语义匹配（阈值 0.95，比缓存命中更严格）
  for (auto& [k, entry] : inflight_) {
    if (cosine(embedding, entry.embedding) >= 0.95f)
      return entry.future;
  }

  return std::nullopt;
}

inline std::shared_ptr<std::promise<std::string>> Singleflight::insert(
    const std::string& key, const std::vector<float>& embedding) {
  std::lock_guard lock(mutex_);

  auto promise = std::make_shared<std::promise<std::string>>();
  Entry entry;
  entry.embedding = embedding;
  entry.promise = promise;
  entry.future = promise->get_future().share();
  inflight_[key] = std::move(entry);
  return promise;
}

inline void Singleflight::complete(const std::string& key,
                                     const std::string& result) {
  std::lock_guard lock(mutex_);
  auto it = inflight_.find(key);
  if (it == inflight_.end()) return;
  it->second.promise->set_value(result);
  inflight_.erase(it);
}

inline void Singleflight::cancel(const std::string& key) {
  std::lock_guard lock(mutex_);
  inflight_.erase(key);
  // promise 销毁时不 set_value，等待者 wait_for 超时
}

}  // namespace ai_gateway