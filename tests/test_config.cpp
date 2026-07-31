// Config 单元测试
#include <cassert>
#include <iostream>
#include "common/config.h"
using namespace ai_gateway;

int main() {
    GatewayConfig cfg;
    assert(GatewayConfig::load("config/gateway.json", cfg) == 0);

    assert(cfg.server.port >= 0);
    assert(!cfg.backend.url.empty());
    assert(cfg.cache.similarity_threshold > 0);

    std::cout << "test_config: 3/3 passed\n";
    return 0;
}
