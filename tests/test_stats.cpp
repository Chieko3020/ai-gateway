// Stats 单元测试
#include <cassert>
#include <iostream>
#include "stats.h"
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

  std::cout << "test_stats: 5/5 passed\n";
  return 0;
}
