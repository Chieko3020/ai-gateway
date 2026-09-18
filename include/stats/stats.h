// 请求统计：缓存命中率、token 消耗、费用估算
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace ai_gateway {

// DeepSeek v4-flash 估算：输入 ¥1/M tok, 输出 ¥4/M tok
// 简化：统一按 ¥0.001/1K tokens 估算
constexpr double kCostPer1KTokens = 0.001;

class Stats {
 public:
  // 记录一次 LLM 调用（非缓存命中）
  // prompt_tokens + completion_tokens 来自 DeepSeek usage 字段
  void record_api_call(int64_t latency_ms,
                       int prompt_tokens, int completion_tokens);

  // 记录一次缓存命中（零 token 消耗）
  void record_cache_hit(int64_t latency_ms);

  // 记录一次旁路请求（带工具调用 / 流式等不可缓存流量）
  // 计入 token 与费用，但不计入命中率分母——命中率只反映可缓存流量
  void record_bypass(int64_t latency_ms,
                     int prompt_tokens, int completion_tokens);

  // 定期输出统计摘要到日志
  void report() const;

  // 基础计数
  size_t total_requests() const { std::shared_lock lock(mutex_); return total_; }
  size_t cache_hits() const { std::shared_lock lock(mutex_); return hits_; }
  size_t cache_misses() const { std::shared_lock lock(mutex_); return misses_; }
  size_t bypassed() const { std::shared_lock lock(mutex_); return bypassed_; }
  double hit_rate() const;  // ODR-used, defined in .cpp

  // Token 统计
  int64_t total_prompt_tokens() const;
  int64_t total_completion_tokens() const;
  int64_t total_tokens_saved() const;

  // 费用估算
  double estimated_cost() const;
  double estimated_saved() const;

  // 延迟统计
  int64_t avg_latency_ms() const;
  int64_t max_latency_ms() const;
  int64_t avg_bypass_latency_ms() const;

 private:
  mutable std::shared_mutex mutex_;
  size_t total_ = 0;
  size_t hits_ = 0;
  size_t misses_ = 0;
  size_t bypassed_ = 0;  // 不可缓存流量（工具调用 / 流式）

  int64_t total_prompt_tokens_ = 0;
  int64_t total_completion_tokens_ = 0;
  int64_t tokens_saved_ = 0;  // 缓存命中省下的 token 估算

  int64_t total_latency_us_ = 0;
  int64_t bypass_latency_us_ = 0;
  int64_t max_latency_ = 0;
  int64_t min_latency_ = INT64_MAX;
};

}  // namespace ai_gateway
