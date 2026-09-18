// Stats 单元测试
#include "test_check.h"
#include <cmath>
#include <iostream>
#include "common/logger.h"
#include "stats/stats.h"
using namespace ai_gateway;

int main() {
  Stats st;
  st.record_cache_hit(100);
  st.record_api_call(200, 50, 30);
  st.record_cache_hit(50);

  CHECK(st.total_requests() == 3);
  CHECK(st.cache_hits() == 2);
  CHECK(st.cache_misses() == 1);
  CHECK(st.hit_rate() > 0.66 && st.hit_rate() < 0.67);
  CHECK(st.max_latency_ms() == 200);
  int ok = 5;

  // 旁路流量（工具调用 / 流式）：计入 token 与独立延迟，但不污染命中率
  {
    Stats b;
    b.record_cache_hit(10);
    b.record_api_call(20, 100, 50);
    double rate_before = b.hit_rate();
    b.record_bypass(300, 200, 80);

    CHECK(b.bypassed() == 1); ok++;
    CHECK(b.total_requests() == 2); ok++;         // 旁路不计入 total
    CHECK(b.hit_rate() == rate_before); ok++;     // 命中率不受影响
    CHECK(b.total_prompt_tokens() == 300); ok++;  // 100 + 200
    CHECK(b.avg_bypass_latency_ms() == 300); ok++;
    CHECK(b.avg_latency_ms() == 15); ok++;        // (10+20)/2，旁路不掺入
  }

  // 回归：report() 内部会调用 avg_bypass_latency_ms()，那里不可再加锁（否则 EDEADLK 终止进程）
  {
    Stats r;
    r.record_cache_hit(5);
    r.record_bypass(7, 10, 20);
    r.report();  // 若死锁，此调用会 abort
    ok++;
  }

  // 延迟分位数：环形窗口（延迟样本 1..100ms）
  {
    Stats q;
    for (int i = 1; i <= 100; ++i) q.record_api_call(i, 0, 0);
    CHECK(q.latency_samples() == 100); ok++;
    CHECK(q.percentile(0) == 1); ok++;
    CHECK(q.percentile(50) >= 49 && q.percentile(50) <= 51); ok++;
    CHECK(q.percentile(95) >= 94 && q.percentile(95) <= 96); ok++;
    CHECK(q.percentile(100) == 100); ok++;
  }

    // 报告 M3：旁路样本独立成池，samples 与 requests 口径自洽
    {
        Stats m;
        for (int i = 1; i <= 10; ++i) m.record_api_call(i * 10, 1, 0);  // 10..100
        m.record_bypass(900, 10, 10);   // 旁路延迟远大于主延迟
        CHECK(m.latency_samples() == 10); ok++;        // 只含可缓存流量
        CHECK(m.bypass_latency_samples() == 1); ok++;
        CHECK(m.percentile(100) == 100); ok++;         // 不被 900ms 的旁路污染
        CHECK(m.bypass_percentile(100) == 900); ok++;
        CHECK(m.total_requests() == 10); ok++;
        CHECK(m.bypassed() == 1); ok++;
    }

    // 报告 M10：合并命中单独计数，不进入 hits/total（命中率不被抬高）
    {
        Stats g;
        g.record_api_call(30, 10, 5);
        double rate_before = g.hit_rate();
        g.record_merge(5);
        CHECK(g.merged() == 1); ok++;
        CHECK(g.cache_hits() == 0); ok++;
        CHECK(g.total_requests() == 1); ok++;
        CHECK(g.hit_rate() == rate_before); ok++;
        CHECK(g.latency_samples() == 2); ok++;  // 合并的延迟样本仍计入主池
    }

    // 报告 M13：纯旁路流量下 min 不得是 INT64_MAX
    {
        Stats b;
        b.record_bypass(120, 1, 1);
        b.record_bypass(340, 1, 1);
        CHECK(b.min_latency_ms() == 0); ok++;   // 无主池样本 -> 0（而不是 INT64_MAX）
        CHECK(b.max_latency_ms() == 0); ok++;
        CHECK(b.avg_latency_ms() == 0); ok++;
        b.report();  // 纯旁路也要能正常输出（含 bypass_* 分位数）
        b.record_api_call(7, 0, 0);
        CHECK(b.min_latency_ms() == 7); ok++;
    }

    // 报告 8.7 第 7 条：费用估算区分输入/输出单价
    {
        // 缺省口径与旧实现完全一致：统一 0.001/1K
        Stats c;
        c.record_api_call(10, 1000, 2000);
        CHECK(c.pricing().input_per_1k == kCostPer1KTokens); ok++;
        CHECK(c.pricing().output_per_1k == kCostPer1KTokens); ok++;
        // 3000 tokens * 0.001 / 1000 = 0.003（= 旧公式的结果）
        CHECK(std::abs(c.estimated_cost() - 0.003) < 1e-12); ok++;

        // 分档：输入 0.001 / 输出 0.004（DeepSeek 的实际比例）
        Stats t;
        t.set_pricing(TokenPricing{0.001, 0.004});
        t.record_api_call(10, 1000, 2000);
        // 1000*0.001/1000 + 2000*0.004/1000 = 0.001 + 0.008 = 0.009
        CHECK(std::abs(t.estimated_cost() - 0.009) < 1e-12); ok++;
        // 判别力：统一单价的旧口径给 0.003，两者必须不同
        CHECK(std::abs(t.estimated_cost() - 0.003) > 1e-9); ok++;
        CHECK(t.pricing().output_per_1k == 0.004); ok++;

        // 命中节省也按分档算：一次未命中(1000 in / 2000 out)之后的命中，
        // 省的正是"平均一次调用"的量 = 1000 in + 2000 out
        t.record_cache_hit(5);
        CHECK(t.total_tokens_saved() == 3000); ok++;                 // 合计口径不变
        CHECK(std::abs(t.estimated_saved() - 0.009) < 1e-12); ok++;  // 0.001 + 0.008
        CHECK(std::abs(t.estimated_saved() - 0.003) > 1e-9); ok++;   // 旧口径是 0.003

        // snapshot() 一次性快照必须与逐字段 getter 一致（/metrics 用的就是它）
        auto s = t.snapshot();
        CHECK(s.requests == t.total_requests()); ok++;
        CHECK(s.hits == t.cache_hits()); ok++;
        CHECK(s.misses == t.cache_misses()); ok++;
        CHECK(s.tokens_saved == t.total_tokens_saved()); ok++;
        CHECK(std::abs(s.cost_yuan - t.estimated_cost()) < 1e-12); ok++;
        CHECK(std::abs(s.saved_yuan - t.estimated_saved()) < 1e-12); ok++;
        CHECK(s.input_per_1k == 0.001 && s.output_per_1k == 0.004); ok++;
        CHECK(s.avg_latency_ms == t.avg_latency_ms()); ok++;
        CHECK(s.latency_samples == t.latency_samples()); ok++;
        CHECK(s.bypass_latency_samples == t.bypass_latency_samples()); ok++;
    }

    // 报告 L7：热路径 INFO 采样判定（默认 1 = 全量；N = 每 N 条留 1 条）
    {
        auto& every = ::ai_gateway::detail::log_sample_every();
        auto& counter = ::ai_gateway::detail::log_line_counter();
        every.store(1);
        int kept = 0;
        for (int i = 0; i < 10; ++i) if (::ai_gateway::detail::sampled_log()) ++kept;
        CHECK(kept == 10); ok++;   // 全量

        every.store(4);
        counter.store(0);
        kept = 0;
        for (int i = 0; i < 40; ++i) if (::ai_gateway::detail::sampled_log()) ++kept;
        CHECK(kept == 10); ok++;   // 40 条里留 10 条
        every.store(1);
        counter.store(0);
        CHECK(::ai_gateway::detail::sampled_log()); ok++;
    }

    return test_check::finish("test_stats", ok);
}