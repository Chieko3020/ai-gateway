// Config 单元测试
//
// 依赖 config/gateway.example.json：优先用工作目录相对路径（ctest 已把
// WORKING_DIRECTORY 钉在源码根目录），直接运行导致相对路径失效时回退到
// 编译期注入的源码根目录，避免"测试因工作目录不同而失败"这一脆性
#include "test_check.h"
#include "common/config.h"
using namespace ai_gateway;

#ifndef GATEWAY_SOURCE_DIR
#define GATEWAY_SOURCE_DIR "."
#endif

int main() {
    GatewayConfig cfg;
    int rc = GatewayConfig::load("config/gateway.example.json", cfg);
    if (rc != 0) {
        rc = GatewayConfig::load(
            std::string(GATEWAY_SOURCE_DIR) + "/config/gateway.example.json", cfg);
    }

    CHECK(rc == 0);  // 示例配置必须可加载
    CHECK(cfg.server.port >= 0);
    CHECK(cfg.server.max_body_bytes > 0);
    CHECK(!cfg.backend.url.empty());
    CHECK(cfg.cache.similarity_threshold > 0);

    return test_check::finish("test_config", 5);
}
