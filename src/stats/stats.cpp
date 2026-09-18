// 统计模块实现
#include "stats/stats.h"

#include <algorithm>

#include "common/logger.h"

namespace ai_gateway {

void Stats::record_api_call(int64_t latency_ms,
                             int prompt_tokens, int completion_tokens) {
  std::lock_guard lock(mutex_);
  ++total_;
  ++misses_;
  total_prompt_tokens_ += prompt_tokens;
  total_completion_tokens_ += completion_tokens;

  auto us = latency_ms * 1000;
  total_latency_us_ += us;
  max_latency_ = std::max(max_latency_, latency_ms);
  min_latency_ = std::min(min_latency_, latency_ms);
}

void Stats::record_cache_hit(int64_t latency_ms) {
  std::lock_guard lock(mutex_);
  ++total_;
  ++hits_;

  // 估算节省的 token：取历史平均每次调用的 token 数
  auto total_tokens = total_prompt_tokens_ + total_completion_tokens_;
  int64_t avg_tokens = (misses_ > 0) ? total_tokens / misses_ : 0;
  tokens_saved_ += avg_tokens;

  auto us = latency_ms * 1000;
  total_latency_us_ += us;
  max_latency_ = std::max(max_latency_, latency_ms);
  min_latency_ = std::min(min_latency_, latency_ms);
}

void Stats::record_bypass(int64_t latency_ms,
                          int prompt_tokens, int completion_tokens) {
  std::lock_guard lock(mutex_);
  // 刻意不增加 total_：旁路流量不参与命中率计算
  ++bypassed_;
  total_prompt_tokens_ += prompt_tokens;
  total_completion_tokens_ += completion_tokens;
  bypass_latency_us_ += latency_ms * 1000;
}

void Stats::report() const {
  std::lock_guard lock(mutex_);
  if (total_ == 0 && bypassed_ == 0) return;

  double saved = estimated_saved();
  LOG_INFO("[STATS] requests={} hits={} misses={} hit_rate={:.1f}% bypassed={} "
           "tokens={} saved={} cost=¥{:.4f} saved=¥{:.4f} "
           "avg={}ms min={}ms max={}ms bypass_avg={}ms",
           total_, hits_, misses_, hit_rate() * 100, bypassed_,
           total_prompt_tokens_ + total_completion_tokens_,
           tokens_saved_,
           estimated_cost(), saved,
           avg_latency_ms(), min_latency_, max_latency_,
           avg_bypass_latency_ms());
}

double Stats::hit_rate() const {
  return total_ > 0 ? static_cast<double>(hits_) / total_ : 0.0;
}

int64_t Stats::avg_latency_ms() const {
  return total_ > 0 ? (total_latency_us_ / 1000) / static_cast<int64_t>(total_) : 0;
}

double Stats::estimated_cost() const {
  auto total_tokens = total_prompt_tokens_ + total_completion_tokens_;
  return total_tokens * kCostPer1KTokens / 1000.0;
}

double Stats::estimated_saved() const {
  return tokens_saved_ * kCostPer1KTokens / 1000.0;
}

int64_t Stats::total_prompt_tokens() const {
  std::shared_lock lock(mutex_);
  return total_prompt_tokens_;
}

int64_t Stats::total_completion_tokens() const {
  std::shared_lock lock(mutex_);
  return total_completion_tokens_;
}

int64_t Stats::total_tokens_saved() const {
  std::shared_lock lock(mutex_);
  return tokens_saved_;
}

int64_t Stats::avg_bypass_latency_ms() const {
  std::shared_lock lock(mutex_);
  return bypassed_ > 0
             ? (bypass_latency_us_ / 1000) / static_cast<int64_t>(bypassed_)
             : 0;
}

int64_t Stats::max_latency_ms() const {
  std::shared_lock lock(mutex_);
  return max_latency_;
}

}  // namespace ai_gateway
