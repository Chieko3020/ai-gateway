// Stats 单元测试
#include <cassert>
#include <iostream>
#include "stats/stats.h"
using namespace ai_gateway;

int main() {
    Stats st;
    st.record(100, true);
    st.record(200, false);
    st.record(50, true);

    assert(st.total_requests() == 3);
    assert(st.cache_hits() == 2);
    assert(st.hit_rate() > 0.66 && st.hit_rate() < 0.67);
    assert(st.max_latency_ms() == 200);

    std::cout << "test_stats: 4/4 passed\n";
    return 0;
}
