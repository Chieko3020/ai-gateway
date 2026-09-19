// 请求统计：缓存命中率、token 消耗、费用估算
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>

namespace ai_gateway {

// DeepSeek v4-flash 估算：输入 ¥1/M tok, 输出 ¥4/M tok
// 旧实现把两者压成一个单价（统一 ¥0.001/1K tokens），输出侧被低估 4 倍。
// 该常量作为**默认单价**保留：配置里不写 cost 段时仍按旧口径估算（向后兼容）
constexpr double kCostPer1KTokens = 0.001;

// 输入/输出分档单价（元 / 1K tokens）
struct TokenPricing {
  double input_per_1k = kCostPer1KTokens;
  double output_per_1k = kCostPer1KTokens;
};

// 一次读一致性快照（/metrics 用）：避免逐字段 getter 拼出互相矛盾的数
struct StatsSnapshot {
  size_t requests = 0;      // 可缓存流量 = hits + misses
  size_t hits = 0;
  size_t misses = 0;
  size_t bypassed = 0;
  size_t merged = 0;
  double hit_rate = 0.0;
  int64_t prompt_tokens = 0;
  int64_t completion_tokens = 0;
  int64_t tokens_saved = 0;
  double cost_yuan = 0.0;
  double saved_yuan = 0.0;
  double input_per_1k = kCostPer1KTokens;
  double output_per_1k = kCostPer1KTokens;
  int64_t avg_latency_ms = 0;
  int64_t min_latency_ms = 0;
  int64_t max_latency_ms = 0;
  int64_t p50_latency_ms = 0;
  int64_t p95_latency_ms = 0;
  int64_t p99_latency_ms = 0;
  size_t latency_samples = 0;
  int64_t bypass_avg_latency_ms = 0;
  int64_t bypass_p50_latency_ms = 0;
  int64_t bypass_p95_latency_ms = 0;
  size_t bypass_latency_samples = 0;
  // 流式（SSE）流量构成：正常完成 / 提前中止 / 正常完成但上游未给 usage。
  // M1 修复后"被中停"才与"正常完成"可区分，因此 streams_aborted 是有意义的信号。
  size_t streams = 0;
  size_t streams_aborted = 0;
  size_t streams_without_usage = 0;
  // 流式命中语义缓存的次数（回放上游原始 SSE 字节，零上游调用）
  size_t stream_hits = 0;
  // 回填缓存的次数（含流式正常完成后写回）
  size_t cache_writes = 0;
};

class Stats {
 public:
  // 延迟样本环形缓冲容量（内存占用固定，分位数反映"最近 N 次请求"而非全历史）
  static constexpr size_t kLatencyWindow = 1024;

  // 记录一次 LLM 调用（非缓存命中）
  // prompt_tokens + completion_tokens 来自 DeepSeek usage 字段
  void record_api_call(int64_t latency_ms,
                       int prompt_tokens, int completion_tokens);

  // 记录一次缓存命中（零 token 消耗）
  void record_cache_hit(int64_t latency_ms);

  // 记录一次"请求合并"（singleflight 命中了同义的在途请求，直接共享其结果）。
  // 与缓存命中分开计数：缓存命中是"从缓存取到内容"，合并是"共享了一次在途请求"，
  // 两者混算会把命中率分子放大（报告 M10）
  void record_merge(int64_t latency_ms);

  // 记录一次旁路请求（带工具调用 / 流式等不可缓存流量）
  // 计入 token 与费用，但不计入命中率分母——命中率只反映可缓存流量
  void record_bypass(int64_t latency_ms,
                     int prompt_tokens, int completion_tokens);

  // 记录一次流式（SSE 透传）请求。
  // 口径说明（必须显式写清，否则这些延迟数字没法解读）：
  //   first_byte_ms —— **首字节延迟（TTFT）**：从收到客户端请求到上游首个字节
  //                    写出去。对流式体验而言这才是用户感知的延迟
  //   total_ms      —— 整段完成时间（最后一个 token 写到客户端为止），
  //                    随回答长度增长，与"网关快不快"无关
  //   prompt_tokens / completion_tokens —— 从 SSE 事件的 usage 字段解析得到
  //                    （见 server/sse_usage.h）。上游未给 usage 时传 0，
  //                    并用 record_stream_no_usage() 把"缺 usage"的次数单列出来
  // 进旁路样本池（不进命中率分母，也不与缓冲式响应混进同一个 avg/min/max）
  void record_stream(int64_t first_byte_ms, int64_t total_ms,
                     int prompt_tokens, int completion_tokens);

  // 记录一次**流式命中语义缓存**：命中后不再回源，直接把当初记下的上游原始
  // SSE 字节回放给客户端（见 gateway/pipeline.cpp 的 handle_stream_request）。
  // 与非流式命中同样计入 hits_ 与命中率分母，并做同样的 saved-token 估算；
  // 额外单列 stream_hits_ —— 否则"命中的流"与"命中的缓冲式响应"在报表里分不开
  void record_stream_hit(int64_t latency_ms);

  // 记录一次**流式未命中回源**（可缓存流量）。
  // 与 record_stream() 的区别只有一处、但很关键：这类流量经过缓存查询，因此
  // 计入命中率**分母**（misses_）；而 record_stream() 收的是不可缓存流量
  // （工具请求等），它只进旁路池。把两者混起来会让命中率分子分母口径不一致
  void record_stream_miss(int64_t first_byte_ms, int64_t total_ms,
                          int prompt_tokens, int completion_tokens);

  // 记录一次缓存回填（写入成功）。用途：回答"流式流量到底有没有进缓存"——
  // 只看命中率看不出来（回填后要等下一次同义请求才可能命中）
  void record_cache_write();
  size_t cache_writes() const {
    std::shared_lock lock(mutex_);
    return cache_writes_;
  }
  size_t stream_hits() const {
    std::shared_lock lock(mutex_);
    return stream_hits_;
  }

  // 流式请求正常结束、但上游事件里**没有** usage 字段（未带
  // stream_options.include_usage，或上游不是标准 OpenAI 实现）。
  // 单列计数是为了让报表能区分"网关没解析"与"上游没给"——两者都表现为
  // token 计 0，但处置完全不同（前者是缺陷，后者是上游能力边界）
  void record_stream_no_usage();
  size_t streams_without_usage() const {
    std::shared_lock lock(mutex_);
    return streams_without_usage_;
  }

  // 流式请求提前结束的次数（客户端断开 / 写死线到点 / 上游中断）。
  // 这类请求不进延迟样本池，只单列计数——否则它们会把 avg/min 拉成
  // "一段没人收完的流的时长"，口径失真
  void record_stream_aborted();
  size_t streams_aborted() const {
    std::shared_lock lock(mutex_);
    return streams_aborted_;
  }

  // 定期输出统计摘要到日志
  void report() const;

  // 基础计数
  size_t total_requests() const { std::shared_lock lock(mutex_); return total_; }
  size_t cache_hits() const { std::shared_lock lock(mutex_); return hits_; }
  size_t cache_misses() const { std::shared_lock lock(mutex_); return misses_; }
  size_t bypassed() const { std::shared_lock lock(mutex_); return bypassed_; }
  size_t merged() const { std::shared_lock lock(mutex_); return merged_; }
  size_t cache_entries() const { std::shared_lock lock(mutex_); return hits_ + misses_; }
  double hit_rate() const;  // ODR-used, defined in .cpp

  // Token 统计
  int64_t total_prompt_tokens() const;
  int64_t total_completion_tokens() const;
  int64_t total_tokens_saved() const;

  // 费用估算
  //
  // 单价可配置：输入与输出分开计价（DeepSeek 实际是 ¥1/M 输入、¥4/M 输出）。
  // 未配置时 pricing() 是 0.001/0.001，estimated_cost() 的数值与旧口径完全一致
  void set_pricing(const TokenPricing& p);
  TokenPricing pricing() const;
  double estimated_cost() const;
  double estimated_saved() const;

  // 一次性快照（/metrics 与报表用）
  StatsSnapshot snapshot() const;

  // 延迟统计
  int64_t avg_latency_ms() const;
  // 最小延迟：无样本时返回 0（旧实现直接暴露 min_latency_ 的初值 INT64_MAX，
  // 纯旁路流量下报表会打印 min=9223372036854775807ms，见报告 M13）
  int64_t min_latency_ms() const;
  int64_t max_latency_ms() const;
  int64_t avg_bypass_latency_ms() const;
  // 旁路延迟的分位数：与主延迟共用同一个环形窗口实现，但样本池独立
  int64_t bypass_percentile(double p) const;
  size_t bypass_latency_samples() const { std::shared_lock lock(mutex_); return bypass_ring_count_; }

  // 延迟分位数（p 取 0~100），基于最近 kLatencyWindow 次请求的样本窗口
  // 不加锁：与 avg_latency_ms() 一致，仅供已持锁的 report() 调用
  int64_t percentile(double p) const;
  size_t latency_samples() const { std::shared_lock lock(mutex_); return ring_count_; }

 private:
  void push_latency(int64_t ms);  // 调用者需持锁

  mutable std::shared_mutex mutex_;
  size_t total_ = 0;
  size_t hits_ = 0;
  size_t misses_ = 0;
  size_t bypassed_ = 0;  // 不可缓存流量（工具调用）
  size_t merged_ = 0;    // 请求合并命中（不计入 total_/hits_）
  size_t streams_ = 0;   // 流式请求完成数（首字节/总时长已入旁路样本池）
  size_t streams_aborted_ = 0;  // 流式请求提前结束数（不进延迟样本池）
  size_t streams_without_usage_ = 0;  // 正常结束但上游未给 usage 的流式请求数
  size_t stream_hits_ = 0;      // 其中命中缓存（回放原始 SSE 字节）的次数
  size_t cache_writes_ = 0;     // 缓存回填次数（含流式正常完成后写回）

  int64_t total_prompt_tokens_ = 0;
  int64_t total_completion_tokens_ = 0;
  int64_t tokens_saved_ = 0;  // 缓存命中省下的 token 估算（in + out 合计）
  // 省下的部分按输入/输出分开累计，才能用分档单价估算金额。
  // tokens_saved_ 的数值与旧口径一致（= 平均单次调用的 prompt+completion）
  int64_t saved_prompt_tokens_ = 0;
  int64_t saved_completion_tokens_ = 0;

  TokenPricing pricing_{};  // 受 mutex_ 保护

  int64_t total_latency_us_ = 0;
  int64_t bypass_latency_us_ = 0;
  int64_t max_latency_ = 0;
  int64_t min_latency_ = INT64_MAX;

  std::array<int64_t, kLatencyWindow> latency_ring_{};
  size_t ring_pos_ = 0;
  size_t ring_count_ = 0;

  // 旁路样本单独一个环形缓冲：混在一起会让 samples 与 requests/avg 自相矛盾
  // （report 里 samples=3 而 requests=1、avg 是毫秒级而 p99 是数百毫秒，报告 M3）
  std::array<int64_t, kLatencyWindow> bypass_ring_{};
  size_t bypass_ring_pos_ = 0;
  size_t bypass_ring_count_ = 0;
};

}  // namespace ai_gateway
