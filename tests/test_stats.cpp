// Stats 单元测试
#include "test_check.h"
#include <iostream>
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

    return test_check::finish("test_stats", ok);
}