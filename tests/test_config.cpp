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

    int ok = 0;
    CHECK(rc == 0); ok++;  // 示例配置必须可加载
    CHECK(cfg.server.port >= 0); ok++;
    CHECK(cfg.server.max_body_bytes > 0); ok++;
    CHECK(!cfg.backend.url.empty()); ok++;
    CHECK(cfg.cache.similarity_threshold > 0); ok++;

    // 报告 L10：默认阈值与文档/示例必须一致（代码为准 = 0.85）
    CHECK(cfg.cache.similarity_threshold > 0.84f &&
          cfg.cache.similarity_threshold < 0.86f); ok++;
    // 报告 L9：示例配置不再把正常回答截断到 600 字节
    CHECK(cfg.filter.max_output_chars == 0); ok++;
    // 报告 M15：embedding 段驱动模型路径与维度
    CHECK(cfg.embedding.dim > 0); ok++;
    CHECK(!cfg.embedding.model_path.empty()); ok++;
    CHECK(!cfg.embedding.vocab_path.empty()); ok++;
    // 报告 H5：连接上限与空闲超时在配置中可见且有合理默认
    CHECK(cfg.server.max_connections > 0); ok++;
    CHECK(cfg.server.idle_timeout_seconds > 0); ok++;
    // 报告 L7：采样率默认全量
    CHECK(cfg.log.sample_every >= 1); ok++;

    // 默认值兜底：空配置（不存在的键）时的默认必须自洽
    {
        GatewayConfig def;
        CHECK(def.filter.max_output_chars == 0); ok++;
        CHECK(def.server.max_connections == 256); ok++;
        CHECK(def.server.idle_timeout_seconds == 30); ok++;
        CHECK(def.embedding.dim == 512); ok++;
        CHECK(def.log.sample_every == 1); ok++;
    }

    return test_check::finish("test_config", ok);
}
