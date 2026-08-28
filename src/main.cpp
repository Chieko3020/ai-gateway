// 网关入口：加载配置 初始化各模块 启动 HTTP 服务

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "backend/llm_client.h"
#include "cache/cache_engine.h"
#include "cache/lru_store.h"
#include "cache/hnsw_index.h"
#include "common/log_file.h"
#include "cache/onnx_embedding.h"
#include "common/config.h"
#include "common/logger.h"
#include "common/singleflight.h"
#include "common/types.h"
#include "server/filter.h"
#include "server/http_server.h"
#include "stats/stats.h"

using namespace ai_gateway;
using json = nlohmann::json;
using namespace std::chrono_literals;

// FNV-1a 32-bit 确定性哈希：保证跨进程一致，避免 std::hash 因随机种子导致
// 重启后同一 system prompt 算出不同 namespace 使缓存隔离失效
static uint32_t fnv1a_32(const std::string& s) {
  uint32_t h = 0x811c9dc5u;
  for (char c : s) {
    h ^= static_cast<uint8_t>(c);
    h *= 0x01000193u;
  }
  return h;
}

// 从 OpenAI 格式请求体中提取第一条 system message 内容，用于缓存隔离
// 返回 system prompt 的 FNV-1a 32-bit hex
static std::string extract_namespace(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    auto& msgs = req.at("messages");
    if (!msgs.empty() && msgs[0].value("role", "") == "system") {
      auto content = msgs[0].value("content", "");
      if (!content.empty()) {
        return std::format("{:08x}", fnv1a_32(content));
      }
    }
  } catch (...) {}
  return "";  // 客户端没有 system message 或解析失败，返回空字符串表示不做隔离
}

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

// 线程安全的关闭标志与后台线程唤醒机制
static std::atomic<bool> g_shutdown{false};
static std::mutex g_bg_mutex;
static std::condition_variable g_bg_cv;

void handle_signal(int /*sig*/) {
  g_shutdown.store(true, std::memory_order_release);
}

// 请求处理管道：过滤 + 缓存 + singleflight + LLM + 统计 + 过滤
static std::string handle_request(const std::string& request_body,
                                   const GatewayConfig& cfg,
                                   MessageFilter* filter,
                                   CacheEngine* engine,
                                   Singleflight* sf,
                                   Stats* stats) {
  auto t0 = std::chrono::steady_clock::now();

  // 缓存命中检查时带回的 embedding（避免 cache_reply 重复计算）
  std::vector<float> cached_embedding;

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

  // 4b. 语义缓存
  std::string ns;
  std::string ns_key;
  if (cfg.cache.enabled && !user_msg.empty()) {
    ns = extract_namespace(request_body);
    ns_key = ns.empty() ? user_msg : ns + ":" + user_msg;
    auto hit = engine->try_hit(user_msg, ns);
    if (hit.hit) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t0);
      stats->record_cache_hit(elapsed.count());
      return hit.reply;
    }
    cached_embedding = std::move(hit.embedding);
  }

  // 4c. 请求合并（singleflight）
  if (!user_msg.empty() && !cached_embedding.empty()) {
    auto fut = sf->try_merge(ns_key, cached_embedding);
    if (fut.has_value()) {
      auto status = fut->wait_for(
          std::chrono::seconds(cfg.backend.timeout_seconds));
      if (status == std::future_status::ready) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);
        stats->record_cache_hit(elapsed.count());
        LOG_INFO("singleflight: merged key={}", ns_key);
        return fut->get();
      }
      LOG_DEBUG("singleflight: wait timeout for key={}", ns_key);
    }
    sf->insert(ns_key, cached_embedding);
  }

  // 4d. 缓存未命中 转发 LLM
  auto result = call_llm(cfg.backend.url, cfg.backend.api_key,
                         request_body, cfg.backend.timeout_seconds);

  // 4e. singleflight 完成/取消
  bool ok = (result.status_code >= 200 && result.status_code < 300);
  if (!ns_key.empty()) {
    if (ok) sf->complete(ns_key, result.body);
    else sf->cancel(ns_key);
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);

  // 4f. 解析 token 用量
  int prompt_tokens = 0, completion_tokens = 0;
  if (result.status_code >= 200 && result.status_code < 300) {
    try {
      auto resp = json::parse(result.body);
      auto& usage = resp.at("usage");
      prompt_tokens = usage.value("prompt_tokens", 0);
      completion_tokens = usage.value("completion_tokens", 0);
    } catch (...) {}
  }

  // 4g. 写入缓存
  if (ok && cfg.cache.enabled && !user_msg.empty())
    engine->cache_reply(user_msg, result.body, cached_embedding,
                        extract_namespace(request_body));

  // 4h. 统计
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
}

int main(int argc, char* argv[]) {
  // ---- 0. 初始化日志文件（与 systemd journal 双写） ----
  ai_gateway::detail::set_log_file("gateway.log");

  // ---- 1. 加载配置 ----
  const char* config_path = (argc > 1) ? argv[1] : "config/gateway.json";
  GatewayConfig cfg;
  if (GatewayConfig::load(config_path, cfg) != 0)
    return static_cast<int>(ErrorCode::kConfigError);
  if (cfg.backend.url.empty()) {
    LOG_ERROR("backend.url is required");
    return static_cast<int>(ErrorCode::kConfigError);
  }

  // ---- 2. 初始化模块 ----
  auto lru = std::make_shared<LruStore>(cfg.cache.max_entries,
                                        cfg.cache.ttl_days * 86400);
  auto idx = std::make_shared<HnswIndex>(HnswConfig{512, 16, 100, 50});

  // 本地 ONNX 嵌入推理
  auto onnx_embed = std::make_shared<OnnxEmbedding>(
      "model/model_int8.onnx",
      "model/vocab.txt", 512);

  auto embed_fn = [onnx_embed](const std::string&, const std::string&,
                                const std::string&, const std::string& text,
                                int) -> std::vector<float> {
    return onnx_embed->ready() ? onnx_embed->encode(text)
                                : std::vector<float>{};
  };

  auto engine = std::make_shared<CacheEngine>(cfg.embedding, cfg.cache, lru, idx, embed_fn);
  auto stats = std::make_shared<Stats>();
  auto filter = std::make_shared<MessageFilter>(cfg.filter);

  // 从磁盘恢复缓存
  if (cfg.cache.enabled) {
    lru->load("cache/lru_store.json");
    // 索引由 LruStore 重建，无需独立加载;
    engine->rebuild_index();
    LOG_INFO("cache restored: {} entries, {} vectors", lru->size(), idx->size());
  }

  struct sigaction sa{};
  sa.sa_handler = handle_signal;
  sa.sa_flags = SA_RESTART;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);

  // ---- 3. 启动定期统计 + 定时持久化线程 ----
  std::thread bg_thread([stats, lru, idx, engine, &cfg] {
    while (!g_shutdown.load(std::memory_order_acquire)) {
      {
        std::unique_lock lk(g_bg_mutex);
        g_bg_cv.wait_for(lk, 60s,
                         [] { return g_shutdown.load(std::memory_order_acquire); });
      }
      if (g_shutdown.load(std::memory_order_acquire)) break;
      stats->report();
      if (cfg.cache.enabled) engine->try_rebuild_if_ghosty();
      // 定期持久化缓存，避免宕机丢了cache
      if (cfg.cache.enabled && lru->size() > 0) {
        lru->save("cache/lru_store.json");
        // 持久化空桩（HNSW 索引通过 LruStore 重建）;
      }
    }
  });

  // ---- 4. 构建 HTTP 服务 + 注册处理器 ----
  HttpServer server(cfg.server);
  g_shutdown.store(false, std::memory_order_release);

  Singleflight flight_merge;
  server.set_handler([&](const std::string& body) {
    return handle_request(body, cfg, filter.get(), engine.get(),
                          &flight_merge, stats.get());
  });

  // ---- 5. 启动 ----
  LOG_INFO("ai-gateway starting on :{}, backend={}", cfg.server.port, cfg.backend.url);
  server.run();

  // ---- 6. 清理：保存缓存 + 统计 ----
  g_shutdown.store(true, std::memory_order_release);
  g_bg_cv.notify_one();  // 唤醒 bg_thread 避免等待 60s 超时
  if (bg_thread.joinable()) bg_thread.join();
  stats->report();
  if (cfg.cache.enabled) {
    lru->save("cache/lru_store.json");
    // 持久化空桩（HNSW 索引通过 LruStore 重建）;
    LOG_INFO("cache persisted: {} entries, {} vectors", lru->size(), idx->size());
  }
  LOG_INFO("cache hits={} misses={} hit_rate={:.1f}%",
           stats->cache_hits(), stats->cache_misses(),
           stats->hit_rate() * 100);
  LOG_INFO("ai-gateway stopped");
  return 0;
}
