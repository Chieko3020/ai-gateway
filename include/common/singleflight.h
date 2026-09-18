// 请求合并（singleflight）：相同 key 的并发请求只放一个去后端，其余等待共享结果
//
// 两层匹配：
//   1. 精确字符串匹配 O(1)：key = namespace:user_message
//   2. 向量语义匹配 O(N)：cosine(embedding, in_flight.embedding) >= 0.95
//
// 失败不共享：LLM 返回错误时以 SingleflightCancelled 异常结束该 promise，
// 等待者 get() 抛出该异常后各自回源重试（不能靠"销毁 promise 让等待者超时"：
// promise 析构会把 future_error(broken_promise) 写进共享状态，等待者 wait_for
// 立即返回 ready，随后 get() 抛出 std::future_error —— 语义是程序错误而非上游失败）
#pragma once

#include <algorithm>
#include <cmath>
#include <exception>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_gateway {

// 被合并请求的上游失败（含超时）时，写入等待者的显式失败结果
class SingleflightCancelled : public std::runtime_error {
 public:
  SingleflightCancelled()
      : std::runtime_error("singleflight: leader request failed") {}
};

class Singleflight {
 public:
  // 尝试合并到已有进行中请求。返回 shared_future 供等待，nullopt 表示无匹配
  std::optional<std::shared_future<std::string>> try_merge(
      const std::string& key, const std::vector<float>& embedding);

  // 插入新的进行中请求（try_merge 返回 nullopt 后调用）
  std::shared_ptr<std::promise<std::string>> insert(
      const std::string& key, const std::vector<float>& embedding);

  // 完成：set_value 并移除
  // owner 为 insert() 返回的 promise：只有仍持有该槽位的 leader 才能写入，
  // 避免"等待超时后已被顶替"的旧 leader 把结果写进新 leader 的 promise
  void complete(const std::string& key, const std::string& result,
                const std::shared_ptr<std::promise<std::string>>& owner);

  // 取消：以 SingleflightCancelled 结束该 promise（等待者立即就绪并抛出该异常），
  // 然后移除条目——等待者据此自行回源，不会共享失败结果；
  // owner 语义同 complete()
  void cancel(const std::string& key,
              const std::shared_ptr<std::promise<std::string>>& owner);

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

  auto it = inflight_.find(key);
  if (it == inflight_.end()) {
    inflight_.emplace(key, std::move(entry));
    return promise;
  }

  // 该 key 上已有 leader（通常是等待超时后原 leader 仍在途）：显式以"被取消"
  // 结束旧 promise，而不是让它随 Entry 赋值被析构（那会写出隐式 broken_promise，
  // 旧等待者拿到的是 future_error 而非可处理的失败信号）
  auto old_promise = it->second.promise;
  it->second = std::move(entry);
  if (old_promise) {
    try {
      old_promise->set_exception(
          std::make_exception_ptr(SingleflightCancelled{}));
    } catch (...) {
      // 旧 promise 已被 set_value/set_exception 满足，忽略
    }
  }
  return promise;
}

inline void Singleflight::complete(
    const std::string& key, const std::string& result,
    const std::shared_ptr<std::promise<std::string>>& owner) {
  std::lock_guard lock(mutex_);
  auto it = inflight_.find(key);
  if (it == inflight_.end()) return;
  if (owner && it->second.promise != owner) return;  // 槽位已被新 leader 顶替
  auto promise = it->second.promise;
  inflight_.erase(it);
  if (!promise) return;
  try {
    promise->set_value(result);
  } catch (const std::future_error&) {
    // 已被取消（insert 覆盖 / cancel）后再 complete：丢弃结果即可
  }
}

inline void Singleflight::cancel(
    const std::string& key,
    const std::shared_ptr<std::promise<std::string>>& owner) {
  std::lock_guard lock(mutex_);
  auto it = inflight_.find(key);
  if (it == inflight_.end()) return;
  if (owner && it->second.promise != owner) return;  // 槽位已被新 leader 顶替
  // 先取出 promise 副本：Entry 被 erase 时引用计数不为 0，
  // 避免析构产生 implicit broken_promise
  auto promise = it->second.promise;
  inflight_.erase(it);
  if (!promise) return;
  try {
    promise->set_exception(std::make_exception_ptr(SingleflightCancelled{}));
  } catch (const std::future_error&) {
    // 已满足：忽略
  }
}

}  // namespace ai_gateway