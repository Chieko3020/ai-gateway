// 请求合并（singleflight）：相同 key 的并发请求只放一个去后端，其余等待共享结果
//
// 两层匹配：
//   1. 精确字符串匹配 O(1)：key = namespace:user_message
//   2. 向量语义匹配 O(N)：cosine(embedding, in_flight.embedding) >= 0.95
//      **并且**（entity_veto 打开时）与在途 leader 的实体标记一致
//      **并且** 与在途 leader 处于同一个 namespace（无 system prompt 时同样是
//      "无 namespace"这一档，不与非空 namespace 的候选合并）
//
// 为什么合并也要做实体一致性校验（第六轮修复）：
//   余弦只反映"整体语义接近"，对"同一模板、只差一个编号"的句子没有判别力。
//   实测把 1 万个语义完全不同的问题（同一模板 + 不同编号）并发灌入，合并路径
//   单独吃掉了约 5200 条（上一轮 5185，第六轮复跑 5163）：跟随者拿到的是**别人
//   问题的上游答案**。这比缓存误命中更
//   严重——它绕过了缓存，直接把别人的上游结果当成自己的结果返回给客户端。
//   合并路径此前只比余弦，`cache.entity_veto` 只挂在缓存命中路径上，是个缺口。
//   复用同一套规则（cache/entity_tokens.h）后两条路径口径一致：数字/大写缩略语/
//   混合标识符存在不对称差集就不许合并。
//
// 为什么合并还要做 namespace 隔离（本轮修复，报告 H2）：
//   第六轮只补了"实体"这一个维度，namespace 维度仍缺：第二层语义匹配拿到了
//   候选 key（`for (auto& [k, entry] : inflight_)`）却**不看它**，于是缓存命中
//   路径的隔离（system prompt 的 FNV-1a 哈希 + 前缀二次过滤，见
//   cache_engine.cpp 的 r.key.starts_with(ns_prefix)）在合并路径被整体绕过。
//   实测：两个 system prompt 不同的对话、user message 相同、间隔 0.25s 并发 →
//   B 拿到 `_cache:"merged"` 且内容是 **A 对话的答案**。user message 相同时
//   余弦恒为 1.0，因此不需要任何"近似"条件就必然触发。
//
//   无 system prompt（namespace 为空）的请求必须**对称地**只与同样无 namespace
//   的条目合并，而不是"空 ns 表示不做限制"：后者会把无 system 的请求与任意
//   带 system 的对话合并到同一份答案上，等于把上面那个洞从另一侧敞开。
//
// 失败不共享：LLM 返回错误时以 SingleflightCancelled 异常结束该 promise，
// 等待者 get() 抛出该异常后各自回源重试（不能靠"销毁 promise 让等待者超时"：
// promise 析构会把 future_error(broken_promise) 写进共享状态，等待者 wait_for
// 立即返回 ready，随后 get() 抛出 std::future_error —— 语义是程序错误而非上游失败）
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cache/entity_tokens.h"

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
  //
  // key / key_prefix：key 是本次请求的完整合并键（ns_key = "ns<hash>:" + user_msg
  // 或裸 user_msg）；key_prefix 是**它自己的 namespace 前缀**（"ns<hash>:" 或空串）。
  // 第二层语义匹配时用它过滤候选：只有同 namespace 的在途条目才允许合并
  // （前缀为空 = 只与同样无 namespace 的条目合并，见文件头说明）。
  // 精确匹配层用完整 key 查表，天然带 namespace，无需再校验。
  //
  // query_entities / entity_veto：与缓存命中路径同一套实体一致性硬约束
  // （CacheConfig::entity_veto）。veto 打开时，候选 leader 的数字/大写缩略语/
  // 混合标识符与查询存在不对称差集就跳过它（**跳过而不是整体放弃**：在途表里可能
  // 还有另一条实体一致的请求可合并）。
  //
  // 为什么用显式 bool 而不是"传一个空 EntityTokens 表示关闭"：空集合与"带编号的
  // 候选"恰好是必须被否决的组合（`继续下一题` ↔ `继续12题`），用空集表达"关闭"
  // 会把语义相反的两种情况混成一种。
  std::optional<std::shared_future<std::string>> try_merge(
      const std::string& key, const std::vector<float>& embedding,
      const std::string& key_prefix, const EntityTokens& query_entities,
      bool entity_veto);

  // 插入新的进行中请求（try_merge 返回 nullopt 后调用）
  // entities 为该 leader 自己那段用户消息的实体标记，供后续 try_merge 比对方
  std::shared_ptr<std::promise<std::string>> insert(
      const std::string& key, const std::vector<float>& embedding,
      const EntityTokens& entities);

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

  // 因实体不一致而被否决的合并候选次数（一次 try_merge 可能否决多条候选）。
  // 只做观测用：它 >> 0 说明"语义接近但实体不同"的并发流量确实存在，
  // 也说明修复前这些请求会共享上游答案
  size_t merge_veto_count() const {
    return merge_veto_count_.load(std::memory_order_relaxed);
  }

 private:
  static float cosine(const std::vector<float>& a, const std::vector<float>& b);

  // 候选 key 是否与本次请求处于同一个 namespace。
  // prefix 为空（请求自身没有 system prompt）时：只接受同样**不带** ns 前缀的
  // 候选 key。ns 前缀的形状由本文件的使用方约定：main/pipeline 里是
  // "ns" + 16 位十六进制（FNV-1a 64 位）+ ":"，见 extract_namespace()
  static bool same_namespace(const std::string& candidate, const std::string& prefix);

  struct Entry {
    std::vector<float> embedding;
    // leader 自己那段用户消息的实体标记（在 insert 时提取一次；try_merge 每次比
    // 一对候选都要用，放在这里避免每条在途请求被反复重新提取）
    EntityTokens entities;
    std::shared_ptr<std::promise<std::string>> promise;
    std::shared_future<std::string> future;
  };

  std::unordered_map<std::string, Entry> inflight_;
  std::mutex mutex_;
  std::atomic<size_t> merge_veto_count_{0};
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

// ns 前缀的形状："ns" + 16 位十六进制 + ":"（extract_namespace() 的产物）。
// 判定"无 ns"时按这个形状识别，而不是简单地"看有没有冒号"——用户消息本身完全
// 可能含冒号（"http://..."、"key:value"），那些都不该被当成命名空间
inline bool Singleflight::same_namespace(const std::string& candidate,
                                         const std::string& prefix) {
  if (!prefix.empty()) return candidate.starts_with(prefix);
  // 本次请求没有 system prompt：只有当候选也不带 ns 前缀时才允许合并
  constexpr size_t kNSPrefixLen = 19;  // "ns" + 16 hex + ":"
  if (candidate.size() < kNSPrefixLen) return true;
  if (candidate.compare(0, 2, "ns") != 0) return true;
  if (candidate[18] != ':') return true;
  for (size_t i = 2; i < 18; ++i) {
    const char c = candidate[i];
    const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    if (!hex) return true;  // 不是 ns 前缀形状
  }
  return false;  // 候选带 ns 前缀，而本次请求没有 → 不许合并
}

inline std::optional<std::shared_future<std::string>> Singleflight::try_merge(
    const std::string& key, const std::vector<float>& embedding,
    const std::string& key_prefix, const EntityTokens& query_entities,
    bool entity_veto) {
  std::lock_guard lock(mutex_);

  // 第一层：精确字符串匹配。key 相同 ⇒ namespace 与（截断后的）用户消息原文都
  // 相同 ⇒ 实体标记必然一致，这里不需要再做实体校验（更不是漏校验）
  auto it = inflight_.find(key);
  if (it != inflight_.end()) return it->second.future;

  // 第二层：向量语义匹配（阈值 0.95，比缓存命中更严格）
  //   + namespace 一致（报告 H2：跨对话串答案，缓存命中有这道闸、合并路径此前没有）
  //   + 实体一致性
  for (auto& [k, entry] : inflight_) {
    if (!same_namespace(k, key_prefix)) continue;
    if (cosine(embedding, entry.embedding) < 0.95f) continue;
    // 语义接近但实体不同：跳过这条候选（不能把"编号1234"的答案发给"编号5678"）；
    // 与 cache_engine 的命中路径同款判定，见 cache/entity_tokens.h 的规则说明
    if (entity_veto && entity_mismatch(query_entities, entry.entities)) {
      merge_veto_count_.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    return entry.future;
  }

  return std::nullopt;
}

inline std::shared_ptr<std::promise<std::string>> Singleflight::insert(
    const std::string& key, const std::vector<float>& embedding,
    const EntityTokens& entities) {
  std::lock_guard lock(mutex_);

  auto promise = std::make_shared<std::promise<std::string>>();
  Entry entry;
  entry.embedding = embedding;
  entry.entities = entities;
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