// Prometheus 文本格式指标渲染（GET /metrics）
//
// 口径与 Stats::report() 的日志摘要保持一致，避免"日志里一个数、抓取端另一个数"：
//   ai_gateway_requests_total    可缓存流量（命中 + 未命中），不含旁路与合并
//   ai_gateway_cache_merged_total 请求合并命中，单独计数（不进命中率分子）
// 脱敏口径不变：这里只暴露聚合计数与延迟分位数，没有任何请求内容/哈希
#pragma once

#include <cstdint>
#include <string>

#include "stats/stats.h"

namespace ai_gateway {

// content_type：Prometheus 期望的 text/plain; version=0.0.4
inline constexpr const char* kMetricsContentType =
    "text/plain; version=0.0.4; charset=utf-8";

// pool_pending / pool_active / pool_threads 来自 HttpServer 的线程池观测，
// 传 -1 表示该字段不可用（渲染时跳过，避免上报假数据）。
// uptime_seconds < 0 时同样跳过 uptime 指标
std::string render_metrics(const Stats& stats, int64_t uptime_seconds = -1,
                           int64_t pool_pending = -1, int64_t pool_active = -1,
                           int64_t pool_threads = -1);

}  // namespace ai_gateway
