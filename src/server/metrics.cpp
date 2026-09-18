// Prometheus 文本格式指标渲染实现
#include "server/metrics.h"

#include <format>
#include <iterator>
#include <set>
#include <string>

namespace ai_gateway {

namespace {

// 累加器：统一控制 "# HELP/# TYPE/样本" 三段式，避免手写格式漏字段
class TextEncoder {
 public:
  void counter(const char* name, const char* help, double v) {
    head(name, help, "counter");
    std::format_to(std::back_inserter(out_), "{} {}\n", name, fmt_num(v));
  }
  void gauge(const char* name, const char* help, double v) {
    head(name, help, "gauge");
    std::format_to(std::back_inserter(out_), "{} {}\n", name, fmt_num(v));
  }
  // 分位数：同一指标名 + quantile 标签
  void quantile(const char* name, const char* help, const char* q, double v) {
    head(name, help, "gauge");
    std::format_to(std::back_inserter(out_), "{}{{quantile=\"{}\"}} {}\n", name,
                   q, fmt_num(v));
  }
  const std::string& str() const { return out_; }

 private:
  void head(const char* name, const char* help, const char* type) {
    // 同一指标名只写一次 HELP/TYPE（重复定义会被抓取端判为格式错误）
    if (!seen_.insert(name).second) return;
    std::format_to(std::back_inserter(out_), "# HELP {} {}\n# TYPE {} {}\n", name,
                   help, name, type);
  }

  // 整数不带小数点（计数器读起来更干净），小数保留 6 位
  static std::string fmt_num(double v) {
    if (v == static_cast<double>(static_cast<int64_t>(v)))
      return std::format("{}", static_cast<int64_t>(v));
    return std::format("{:.6f}", v);
  }

  std::string out_;
  std::set<std::string> seen_;  // 已输出过 HELP/TYPE 的指标名
};

}  // namespace

std::string render_metrics(const Stats& stats, int64_t uptime_seconds,
                           int64_t pool_pending, int64_t pool_active,
                           int64_t pool_threads) {
  const StatsSnapshot s = stats.snapshot();
  TextEncoder e;

  e.gauge("ai_gateway_up", "1 = 进程在跑，可被抓取", 1);

  e.counter("ai_gateway_requests_total",
            "可缓存流量请求数（命中 + 未命中，不含旁路与合并）",
            static_cast<double>(s.requests));
  e.counter("ai_gateway_cache_hits_total", "语义缓存命中次数",
            static_cast<double>(s.hits));
  e.counter("ai_gateway_cache_misses_total", "语义缓存未命中（回源上游）次数",
            static_cast<double>(s.misses));
  e.gauge("ai_gateway_cache_hit_ratio", "命中率 = hits / requests（0~1）",
          s.hit_rate);
  e.counter("ai_gateway_cache_bypassed_total",
            "旁路流量（工具调用等不可缓存请求）次数", static_cast<double>(s.bypassed));
  e.counter("ai_gateway_cache_merged_total",
            "请求合并（singleflight）命中次数，单独计数", static_cast<double>(s.merged));

  e.counter("ai_gateway_tokens_prompt_total", "上游 usage 中的 prompt_tokens 累计",
            static_cast<double>(s.prompt_tokens));
  e.counter("ai_gateway_tokens_completion_total",
            "上游 usage 中的 completion_tokens 累计",
            static_cast<double>(s.completion_tokens));
  e.counter("ai_gateway_tokens_saved_total",
            "缓存命中估算省下的 token（按平均单次调用量）",
            static_cast<double>(s.tokens_saved));

  e.counter("ai_gateway_cost_yuan_total", "估算花费（元，输入/输出分档计价）",
            s.cost_yuan);
  e.counter("ai_gateway_cost_saved_yuan_total", "估算省下的花费（元）",
            s.saved_yuan);
  e.gauge("ai_gateway_price_input_per_1k_yuan", "当前输入单价（元/1K tokens）",
          s.input_per_1k);
  e.gauge("ai_gateway_price_output_per_1k_yuan", "当前输出单价（元/1K tokens）",
          s.output_per_1k);

  e.gauge("ai_gateway_latency_avg_milliseconds", "平均延迟（主延迟池，毫秒）",
          static_cast<double>(s.avg_latency_ms));
  e.gauge("ai_gateway_latency_min_milliseconds", "最小延迟（毫秒）",
          static_cast<double>(s.min_latency_ms));
  e.gauge("ai_gateway_latency_max_milliseconds", "最大延迟（毫秒，全历史）",
          static_cast<double>(s.max_latency_ms));
  e.quantile("ai_gateway_latency_milliseconds", "延迟分位数（最近窗口，毫秒）",
             "0.5", static_cast<double>(s.p50_latency_ms));
  e.quantile("ai_gateway_latency_milliseconds", "延迟分位数（最近窗口，毫秒）",
             "0.95", static_cast<double>(s.p95_latency_ms));
  e.quantile("ai_gateway_latency_milliseconds", "延迟分位数（最近窗口，毫秒）",
             "0.99", static_cast<double>(s.p99_latency_ms));
  e.gauge("ai_gateway_latency_samples",
          "主延迟窗口内的样本数（窗口满后恒为窗口大小）",
          static_cast<double>(s.latency_samples));

  e.gauge("ai_gateway_bypass_latency_avg_milliseconds",
          "旁路流量平均延迟（毫秒，独立样本池）",
          static_cast<double>(s.bypass_avg_latency_ms));
  e.quantile("ai_gateway_bypass_latency_milliseconds",
             "旁路延迟分位数（毫秒）", "0.5",
             static_cast<double>(s.bypass_p50_latency_ms));
  e.quantile("ai_gateway_bypass_latency_milliseconds",
             "旁路延迟分位数（毫秒）", "0.95",
             static_cast<double>(s.bypass_p95_latency_ms));
  e.gauge("ai_gateway_bypass_latency_samples", "旁路延迟样本数",
          static_cast<double>(s.bypass_latency_samples));

  // 本项目没有内存池；线程池队列深度是唯一的"内部资源排队"信号
  if (pool_pending >= 0)
    e.gauge("ai_gateway_thread_pool_pending_tasks", "线程池队列中待执行的任务数",
            static_cast<double>(pool_pending));
  if (pool_active >= 0)
    e.gauge("ai_gateway_thread_pool_active_tasks", "线程池中正在执行的任务数",
            static_cast<double>(pool_active));
  if (pool_threads >= 0)
    e.gauge("ai_gateway_thread_pool_size", "线程池工作线程数",
            static_cast<double>(pool_threads));

  if (uptime_seconds >= 0)
    e.gauge("ai_gateway_uptime_seconds", "进程运行时长（秒）",
            static_cast<double>(uptime_seconds));

  return e.str();
}

}  // namespace ai_gateway
