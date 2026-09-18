// 统计模块实现
#include "stats/stats.h"

#include <algorithm>
#include <array>
#include <vector>

#include "common/logger.h"

namespace ai_gateway {

void Stats::record_api_call(int64_t latency_ms,
                             int prompt_tokens, int completion_tokens) {
  std::lock_guard lock(mutex_);
  push_latency(latency_ms);
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
  push_latency(latency_ms);
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
  push_latency(latency_ms);
  // 刻意不增加 total_：旁路流量不参与命中率计算
  ++bypassed_;
  total_prompt_tokens_ += prompt_tokens;
  total_completion_tokens_ += completion_tokens;
  bypass_latency_us_ += latency_ms * 1000;
}

void Stats::push_latency(int64_t ms) {
  latency_ring_[ring_pos_] = ms;
  ring_pos_ = (ring_pos_ + 1) % kLatencyWindow;
  if (ring_count_ < kLatencyWindow) ++ring_count_;
}

int64_t Stats::percentile(double p) const {
  if (ring_count_ == 0) return 0;
  std::vector<int64_t> xs(latency_ring_.begin(),
                          latency_ring_.begin() + static_cast<long>(ring_count_));
  std::sort(xs.begin(), xs.end());
  double idx = (p / 100.0) * static_cast<double>(xs.size() - 1);
  size_t i = static_cast<size_t>(idx < 0 ? 0 : idx);
  if (i >= xs.size()) i = xs.size() - 1;
  return xs[i];
}

void Stats::report() const {
  std::lock_guard lock(mutex_);
  if (total_ == 0 && bypassed_ == 0) return;

  double saved = estimated_saved();
  LOG_INFO("[STATS] requests={} hits={} misses={} hit_rate={:.1f}% bypassed={} "
           "tokens={} saved={} cost=¥{:.4f} saved=¥{:.4f} "
           "avg={}ms min={}ms max={}ms p50={}ms p95={}ms p99={}ms "
           "bypass_avg={}ms samples={}",
           total_, hits_, misses_, hit_rate() * 100, bypassed_,
           total_prompt_tokens_ + total_completion_tokens_,
           tokens_saved_,
           estimated_cost(), saved,
           avg_latency_ms(), min_latency_, max_latency_,
           percentile(50), percentile(95), percentile(99),
           avg_bypass_latency_ms(), ring_count_);
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
  // 不在此加锁：与 avg_latency_ms() 一致——report() 已持锁后调用它，
  // 若这里再加 shared_lock 会造成同一线程重复加锁（EDEADLK / Resource deadlock avoided）。
  return bypassed_ > 0
             ? (bypass_latency_us_ / 1000) / static_cast<int64_t>(bypassed_)
             : 0;
}

int64_t Stats::max_latency_ms() const {
  std::shared_lock lock(mutex_);
  return max_latency_;
}

}  // namespace ai_gateway
