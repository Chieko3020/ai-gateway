// Stats 单元测试
#include <cassert>
#include <iostream>
#include "stats/stats.h"
using namespace ai_gateway;

int main() {
  Stats st;
  st.record_cache_hit(100);
  st.record_api_call(200, 50, 30);
  st.record_cache_hit(50);

  assert(st.total_requests() == 3);
  assert(st.cache_hits() == 2);
  assert(st.cache_misses() == 1);
  assert(st.hit_rate() > 0.66 && st.hit_rate() < 0.67);
  assert(st.max_latency_ms() == 200);
  int ok = 5;

  // 旁路流量（工具调用 / 流式）：计入 token 与独立延迟，但不污染命中率
  {
    Stats b;
    b.record_cache_hit(10);
    b.record_api_call(20, 100, 50);
    double rate_before = b.hit_rate();
    b.record_bypass(300, 200, 80);

    assert(b.bypassed() == 1); ok++;
    assert(b.total_requests() == 2); ok++;         // 旁路不计入 total
    assert(b.hit_rate() == rate_before); ok++;     // 命中率不受影响
    assert(b.total_prompt_tokens() == 300); ok++;  // 100 + 200
    assert(b.avg_bypass_latency_ms() == 300); ok++;
    assert(b.avg_latency_ms() == 15); ok++;        // (10+20)/2，旁路不掺入
  }

  std::cout << "test_stats: " << ok << "/11 passed\n";
  return 0;
}
