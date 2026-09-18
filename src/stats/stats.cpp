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

  // 估算节省的 token：取历史平均每次调用的 token 数。
  // 旧实现把 prompt 与 completion 相加后再除，金额只能用统一单价；这里额外分开
  // 累计（合计值 tokens_saved_ 与旧口径一致），于是 estimated_saved() 能按
  // 输入/输出分档计价（报告 8.7 第 7 条）
  auto total_tokens = total_prompt_tokens_ + total_completion_tokens_;
  int64_t avg_tokens = (misses_ > 0) ? total_tokens / misses_ : 0;
  tokens_saved_ += avg_tokens;
  if (misses_ > 0) {
    saved_prompt_tokens_ += total_prompt_tokens_ / static_cast<int64_t>(misses_);
    saved_completion_tokens_ +=
        total_completion_tokens_ / static_cast<int64_t>(misses_);
  }

  auto us = latency_ms * 1000;
  total_latency_us_ += us;
  max_latency_ = std::max(max_latency_, latency_ms);
  min_latency_ = std::min(min_latency_, latency_ms);
}

void Stats::record_merge(int64_t latency_ms) {
  std::lock_guard lock(mutex_);
  push_latency(latency_ms);
  // 不计入 total_/hits_：合并是"共享了一次在途请求"，不是"从缓存取到内容"。
  // 若按缓存命中计数，命中率的分子会被合并行为抬高（报告 M10）
  ++merged_;
  auto us = latency_ms * 1000;
  total_latency_us_ += us;
  max_latency_ = std::max(max_latency_, latency_ms);
  min_latency_ = std::min(min_latency_, latency_ms);
}

void Stats::record_bypass(int64_t latency_ms,
                          int prompt_tokens, int completion_tokens) {
  std::lock_guard lock(mutex_);
  // 旁路样本进独立环形缓冲：旧实现把它们混进 latency_ring_，导致 report() 的
  // samples 与 requests 口径矛盾（samples=3 而 requests=1），且旁路延迟（数百 ms）
  // 会污染 avg/min/max 之外的分位数（报告 M3）
  bypass_ring_[bypass_ring_pos_] = latency_ms;
  bypass_ring_pos_ = (bypass_ring_pos_ + 1) % kLatencyWindow;
  if (bypass_ring_count_ < kLatencyWindow) ++bypass_ring_count_;
  // 刻意不增加 total_：旁路流量不参与命中率计算
  ++bypassed_;
  total_prompt_tokens_ += prompt_tokens;
  total_completion_tokens_ += completion_tokens;
  bypass_latency_us_ += latency_ms * 1000;
}

void Stats::record_stream(int64_t first_byte_ms, int64_t total_ms,
                          int prompt_tokens, int completion_tokens) {
  std::lock_guard lock(mutex_);
  // 延迟样本池里放**首字节延迟**：流式请求的"用户感知延迟"就是 TTFT。
  // 整段时长随回答长度线性增长，混进同一个分位数池会让 p95 失去意义，
  // 因此它只进日志不进池（见 report() 的口径说明）
  bypass_ring_[bypass_ring_pos_] = first_byte_ms;
  bypass_ring_pos_ = (bypass_ring_pos_ + 1) % kLatencyWindow;
  if (bypass_ring_count_ < kLatencyWindow) ++bypass_ring_count_;
  ++bypassed_;
  ++streams_;
  total_prompt_tokens_ += prompt_tokens;
  total_completion_tokens_ += completion_tokens;
  bypass_latency_us_ += first_byte_ms * 1000;
  // total_ms 目前只由调用方写进日志：把它也塞进样本池会让分位数含义混乱
  // （一半样本是 TTFT、一半是整段时长），因此这里显式丢弃
  (void)total_ms;
}

void Stats::record_stream_aborted() {
  std::lock_guard lock(mutex_);
  ++streams_aborted_;
}

void Stats::push_latency(int64_t ms) {
  latency_ring_[ring_pos_] = ms;
  ring_pos_ = (ring_pos_ + 1) % kLatencyWindow;
  if (ring_count_ < kLatencyWindow) ++ring_count_;
}

namespace {
int64_t percentile_of(const std::array<int64_t, Stats::kLatencyWindow>& ring,
                      size_t count, double p) {
  if (count == 0) return 0;
  std::vector<int64_t> xs(ring.begin(),
                          ring.begin() + static_cast<long>(count));
  std::sort(xs.begin(), xs.end());
  double idx = (p / 100.0) * static_cast<double>(xs.size() - 1);
  size_t i = static_cast<size_t>(idx < 0 ? 0 : idx);
  if (i >= xs.size()) i = xs.size() - 1;
  return xs[i];
}
}  // namespace

int64_t Stats::percentile(double p) const {
  return percentile_of(latency_ring_, ring_count_, p);
}

int64_t Stats::bypass_percentile(double p) const {
  return percentile_of(bypass_ring_, bypass_ring_count_, p);
}

void Stats::report() const {
  std::lock_guard lock(mutex_);
  if (total_ == 0 && bypassed_ == 0 && merged_ == 0 && streams_aborted_ == 0)
    return;

  double saved = estimated_saved();
  // 口径说明（报告 M3/M10/M13 + 本轮流式）：
  //   requests   = 可缓存流量（未命中 + 命中），不含旁路与合并
  //   hit_rate   = hits / requests，既不把旁路当分母，也不把合并当分子
  //   merged     = 请求合并命中，单独计数
  //   samples    = 主延迟环形缓冲里的样本数，恒等于 requests（≤ 窗口大小 1024）
  //   bypass_*   = 旁路流量自己的样本池与分位数，不混入上面的 avg/min/max。
  //                **流式请求也进这个池，进池的值是首字节延迟（TTFT）**：
  //                流式的"整段完成时间"随回答长度变化，进池会让分位数失去意义；
  //                这里另开 streams 与 streams_abort 两个计数说明样本构成
  //   cost/saved = 按输入/输出分档单价估算（默认 0.001/0.001 = 旧口径）；
  //                同时打印所用单价，避免"金额变了却查不出换没换价"
  LOG_INFO("[STATS] requests={} hits={} misses={} hit_rate={:.1f}% merged={} "
           "bypassed={} streams={} streams_abort={} tokens={} saved={} "
           "cost=¥{:.4f} saved=¥{:.4f} "
           "price_in=¥{}/1K price_out=¥{}/1K "
           "avg={}ms min={}ms max={}ms p50={}ms p95={}ms p99={}ms samples={} "
           "bypass_avg={}ms bypass_p50={}ms bypass_p95={}ms bypass_samples={}",
           total_, hits_, misses_, hit_rate() * 100, merged_, bypassed_,
           streams_, streams_aborted_,
           total_prompt_tokens_ + total_completion_tokens_,
           tokens_saved_,
           estimated_cost(), saved,
           pricing_.input_per_1k, pricing_.output_per_1k,
           avg_latency_ms(), min_latency_ms(), max_latency_,
           percentile(50), percentile(95), percentile(99), ring_count_,
           avg_bypass_latency_ms(), bypass_percentile(50), bypass_percentile(95),
           bypass_ring_count_);
}

double Stats::hit_rate() const {
  return total_ > 0 ? static_cast<double>(hits_) / total_ : 0.0;
}

int64_t Stats::avg_latency_ms() const {
  return total_ > 0 ? (total_latency_us_ / 1000) / static_cast<int64_t>(total_) : 0;
}

void Stats::set_pricing(const TokenPricing& p) {
  std::lock_guard lock(mutex_);
  pricing_ = p;
}

TokenPricing Stats::pricing() const {
  std::shared_lock lock(mutex_);
  return pricing_;
}

double Stats::estimated_cost() const {
  // 不加锁：与 avg_latency_ms() 等一致，report()/snapshot() 已在锁内调用
  return static_cast<double>(total_prompt_tokens_) *
             pricing_.input_per_1k / 1000.0 +
         static_cast<double>(total_completion_tokens_) *
             pricing_.output_per_1k / 1000.0;
}

double Stats::estimated_saved() const {
  // 单价缺省时（0.001/0.001）等价于旧口径：tokens_saved * 0.001 / 1000
  return static_cast<double>(saved_prompt_tokens_) * pricing_.input_per_1k /
             1000.0 +
         static_cast<double>(saved_completion_tokens_) * pricing_.output_per_1k /
             1000.0;
}

StatsSnapshot Stats::snapshot() const {
  std::shared_lock lock(mutex_);
  StatsSnapshot s;
  s.requests = total_;
  s.hits = hits_;
  s.misses = misses_;
  s.bypassed = bypassed_;
  s.merged = merged_;
  s.hit_rate = hit_rate();
  s.prompt_tokens = total_prompt_tokens_;
  s.completion_tokens = total_completion_tokens_;
  s.tokens_saved = tokens_saved_;
  s.input_per_1k = pricing_.input_per_1k;
  s.output_per_1k = pricing_.output_per_1k;
  s.cost_yuan = estimated_cost();
  s.saved_yuan = estimated_saved();
  s.avg_latency_ms = avg_latency_ms();
  s.min_latency_ms = min_latency_ms();
  s.max_latency_ms = max_latency_;
  s.p50_latency_ms = percentile(50);
  s.p95_latency_ms = percentile(95);
  s.p99_latency_ms = percentile(99);
  s.latency_samples = ring_count_;
  s.bypass_avg_latency_ms = avg_bypass_latency_ms();
  s.bypass_p50_latency_ms = bypass_percentile(50);
  s.bypass_p95_latency_ms = bypass_percentile(95);
  s.bypass_latency_samples = bypass_ring_count_;
  return s;
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

int64_t Stats::min_latency_ms() const {
  // 不加锁：与 avg_latency_ms()/avg_bypass_latency_ms() 一致——report() 已持锁后
  // 调用它，这里再加锁会变成同一线程重复加锁（EDEADLK / Resource deadlock avoided）
  return ring_count_ > 0 ? min_latency_ : 0;
}

int64_t Stats::max_latency_ms() const {
  std::shared_lock lock(mutex_);
  return max_latency_;
}

}  // namespace ai_gateway
