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
// uptime_seconds < 0 时同样跳过 uptime 指标。
// connections_accepted：累计 accept 成功的连接数（= 服务端执行的 TCP 握手次数）。
// keep-alive 生效时"同一连接上的 N 个请求"只贡献 1，这个指标是复用收益最直接的
// 观测面（-1 = 不可用）
// index_ready / index_building / index_vectors 来自 CacheEngine 的建图状态，
// 传 -1 表示不可用。为什么要暴露它们：重建已改为异步（建图可达上百秒），
// "后台在忙"期间语义检索退化为精确匹配——没有这两个布尔量，运维只能看到
// 命中率莫名下降却不知道原因
std::string render_metrics(const Stats& stats, int64_t uptime_seconds = -1,
                           int64_t pool_pending = -1, int64_t pool_active = -1,
                           int64_t pool_threads = -1,
                           int64_t connections_accepted = -1,
                           int64_t index_ready = -1,
                           int64_t index_building = -1,
                           int64_t index_vectors = -1,
                           int64_t degraded_searches = -1);

}  // namespace ai_gateway
