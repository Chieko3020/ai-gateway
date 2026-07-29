// 网关入口：加载配置 → 初始化各模块 → 启动 HTTP 服务

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <format>
#include <memory>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "backend/llm_client.h"
#include "cache/cache_engine.h"
#include "cache/lru_store.h"
#include "cache/vector_index.h"
#include "common/config.h"
#include "common/logger.h"
#include "common/types.h"
#include "server/filter.h"
#include "server/http_server.h"
#include "stats/stats.h"

using namespace ai_gateway;
using json = nlohmann::json;
using namespace std::chrono_literals;

// 从 OpenAI 格式请求体中提取最后一条 user message
static std::string extract_user_message(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    auto& msgs = req.at("messages");
    for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
      if ((*it).value("role", "") == "user") {
        return (*it).value("content", "");
      }
    }
  } catch (...) {}
  return "";
}

static std::atomic<bool> g_shutdown{false};

void handle_signal(int /*sig*/) {
  g_shutdown.store(true, std::memory_order_release);
}

int main(int argc, char* argv[]) {
  // ---- 1. 加载配置 ----
  const char* config_path = (argc > 1) ? argv[1] : "config/gateway.json";
  GatewayConfig cfg;
  if (GatewayConfig::load(config_path, cfg) != 0)
    return static_cast<int>(ErrorCode::kConfigError);
  if (cfg.backend.url.empty()) {
    LOG_ERROR("backend.url is required");
    return static_cast<int>(ErrorCode::kConfigError);
  }

  // ---- 2. 初始化模块（全部由 config 驱动）----
  auto lru = std::make_shared<LruStore>(cfg.cache.max_entries,
                                        cfg.cache.ttl_days * 86400);
  auto idx = std::make_shared<VectorIndex>();
  auto engine = std::make_shared<CacheEngine>(cfg.embedding, cfg.cache, lru, idx);
  auto stats = std::make_shared<Stats>();
  auto filter = std::make_shared<MessageFilter>(cfg.filter);

  // 从磁盘恢复缓存
  if (cfg.cache.enabled) {
    lru->load("cache/lru_store.json");
    idx->load("cache/vector_index.bin");
    engine->rebuild_index();  // 从 lru_store 重建向量索引
    LOG_INFO("cache restored: {} entries, {} vectors", lru->size(), idx->size());
  }

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);

  // ---- 3. 启动定期统计线程 ----
  std::thread stats_thread([stats] {
    while (!g_shutdown.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(60s);
      stats->report();
    }
  });

  // ---- 4. 构建 HTTP 服务 + 注册处理器 ----
  HttpServer server(cfg.server);
  g_shutdown.store(false, std::memory_order_release);

  server.set_handler([&](const std::string& request_body) -> std::string {
    auto t0 = std::chrono::steady_clock::now();

    // 4a. 输入过滤
    std::string user_msg = extract_user_message(request_body);
    if (!user_msg.empty()) {
      auto f_result = filter->check_input(user_msg);
      if (f_result.action == FilterAction::kReject) {
        LOG_WARN("filter: rejected input: {}", f_result.reject_msg);
        return R"({"error":"Request rejected"})";
      }
      if (f_result.action == FilterAction::kTruncate)
        user_msg = f_result.sanitized;
    }

    // 4b. 语义缓存（embedding 失败时自动降级为精确匹配）
    if (cfg.cache.enabled && !user_msg.empty()) {
      auto hit = engine->try_hit(user_msg);
      if (hit.has_value()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
        stats->record_cache_hit(elapsed.count());
        return hit->reply;
      }
    }

    // 4c. 缓存未命中 → 转发 LLM
    auto result = call_llm(cfg.backend.url, cfg.backend.api_key,
                           cfg.backend.model, request_body,
                           cfg.backend.timeout_seconds);

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);

    // 4d. 解析 token 用量
    int prompt_tokens = 0, completion_tokens = 0;
    if (result.status_code >= 200 && result.status_code < 300) {
      try {
        auto resp = json::parse(result.body);
        auto& usage = resp.at("usage");
        prompt_tokens = usage.value("prompt_tokens", 0);
        completion_tokens = usage.value("completion_tokens", 0);
      } catch (...) {}
    }

    // 4e. 写入缓存
    bool ok = (result.status_code >= 200 && result.status_code < 300);
    if (ok && cfg.cache.enabled && !user_msg.empty())
      engine->cache_reply(user_msg, result.body);

    // 4f. 统计
    stats->record_api_call(elapsed.count(), prompt_tokens, completion_tokens);

    if (ok)
      LOG_INFO("{} {} {}ms", result.status_code, result.body.size(), elapsed.count());
    else
      LOG_WARN("{} {} {}ms", result.status_code,
               result.body.size() > 0 ? result.body : "(empty)", elapsed.count());

    // 输出过滤
    auto out_result = filter->check_output(result.body);
    if (out_result.action == FilterAction::kReject) {
      LOG_WARN("filter: rejected output containing URL");
      return R"({"error":"Response filtered"})";
    }
    return out_result.sanitized;
  });

  // ---- 5. 启动 ----
  LOG_INFO("ai-gateway starting on :{}, backend={}", cfg.server.port, cfg.backend.url);
  server.run();

  // ---- 6. 清理：保存缓存 + 统计 ----
  g_shutdown.store(true, std::memory_order_release);
  if (stats_thread.joinable()) stats_thread.join();
  stats->report();
  if (cfg.cache.enabled) {
    lru->save("cache/lru_store.json");
    idx->save("cache/vector_index.bin");
    LOG_INFO("cache persisted: {} entries, {} vectors", lru->size(), idx->size());
  }
  LOG_INFO("cache hits={} misses={} hit_rate={:.1f}%",
           engine->hit_count(), engine->miss_count(), engine->hit_rate() * 100);
  LOG_INFO("ai-gateway stopped");
  return 0;
}
