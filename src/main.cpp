// 网关入口：加载配置 初始化各模块 启动 HTTP 服务

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <format>
#include <future>
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

// 判断请求是否属于"不可缓存流量"：带工具调用定义/结果的请求，或流式请求。
// 这类请求与上下文强相关——缓存会丢失 tool_calls、或返回不适用答案，必须直接透传。
static bool is_uncacheable_request(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    if (req.contains("tools") || req.contains("functions")) return true;
    if (req.contains("stream") && req.value("stream", false)) return true;
    auto& msgs = req.at("messages");
    for (const auto& m : msgs) {
      auto role = m.value("role", "");
      if (role == "tool" || role == "function") return true;
      if (m.contains("tool_calls") || m.contains("function_call")) return true;
    }
  } catch (...) {}
  return false;
}

// 在 OpenAI 响应体中注入缓存状态字段，供压测脚本精确判定是否命中；
// 额外字段不影响下游对标准字段的解析。
static std::string annotate_cache_status(const std::string& body,
                                         const char* status) {
  try {
    auto resp = json::parse(body);
    if (!resp.is_object()) return body;
    resp["_cache"] = status;
    return resp.dump();
  } catch (...) {
    return body;  // 非 JSON（如错误页）原样返回
  }
}

// 线程安全的关闭标志与后台线程唤醒机制
static std::atomic<bool> g_shutdown{false};
static std::atomic<bool> g_dump_stats{false};  // SIGUSR1 置位，由后台线程输出统计
static std::mutex g_bg_mutex;
static std::condition_variable g_bg_cv;

void handle_signal(int sig) {
  if (sig == SIGUSR1) {
    // 按需导出统计：信号处理器里只置标志（report() 会加锁+写日志，不是 async-signal-safe）
    g_dump_stats.store(true, std::memory_order_release);
    return;
  }
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

  // 1. 输入过滤（所有路径都必须执行）
  //    旁路（工具调用/流式）只应跳过缓存与请求合并，不能跳过安全过滤：
  //    否则客户端加一个 "stream": true 就能绕过注入检测、URL 拦截、屏蔽词与长度截断
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

  // 2. 不可缓存流量：带工具调用或流式的请求直接转发（不查缓存、不写缓存、不参与请求合并）
  if (is_uncacheable_request(request_body)) {
    auto result = call_llm(cfg.backend.url, cfg.backend.api_key,
                           request_body, cfg.backend.timeout_seconds);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0);
    int pt = 0, ct = 0;
    if (result.status_code >= 200 && result.status_code < 300) {
      try {
        auto resp = json::parse(result.body);
        auto& usage = resp.at("usage");
        pt = usage.value("prompt_tokens", 0);
        ct = usage.value("completion_tokens", 0);
      } catch (...) {}
    }
    stats->record_bypass(elapsed.count(), pt, ct);
    LOG_INFO("cache: bypass (tool/stream) status={} {}ms",
             result.status_code, elapsed.count());

    auto out_result = filter->check_output(result.body);
    if (out_result.action == FilterAction::kReject) {
      LOG_WARN("filter: rejected output containing URL");
      return R"({"error":"Response filtered"})";
    }
    return annotate_cache_status(out_result.sanitized, "bypass");
  }

  // 缓存命中检查时带回的 embedding（避免 cache_reply 重复计算）
  std::vector<float> cached_embedding;

  // 3. 语义缓存
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
      // 命中路径与未命中路径统一口径：缓存里存的是上游原文（未过滤），取出来同样
      // 要过 check_output，否则"先让含 URL 的答案入缓存、再命中"即可绕过输出过滤
      auto hit_out = filter->check_output(hit.reply);
      if (hit_out.action == FilterAction::kReject) {
        LOG_WARN("filter: rejected cached output containing URL");
        return R"({"error":"Response filtered"})";
      }
      return annotate_cache_status(hit_out.sanitized, "hit");
    }
    cached_embedding = std::move(hit.embedding);
  }

  // 4. 请求合并（singleflight）
  // 本请求作为 leader 占用的槽位（insert 的返回值）；只有它有权 complete/cancel，
  // 这样"等待超时后被新 leader 顶替"的旧 leader 不会误写别人的 promise
  std::shared_ptr<std::promise<std::string>> sf_slot;
  if (!user_msg.empty() && !cached_embedding.empty()) {
    auto fut = sf->try_merge(ns_key, cached_embedding);
    if (fut.has_value()) {
      auto status = fut->wait_for(
          std::chrono::seconds(cfg.backend.timeout_seconds));
      if (status == std::future_status::ready) {
        // 主请求失败/被取消时等待者会拿到 SingleflightCancelled（而不是
        // broken_promise 触发的 std::future_error）：此时不共享结果、自己回源
        std::string merged;
        bool merged_ok = false;
        try {
          merged = fut->get();
          merged_ok = true;
        } catch (const std::exception& e) {
          LOG_WARN("singleflight: leader failed ({}), fallback to upstream",
                   e.what());
        }
        if (merged_ok) {
          auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t0);
          stats->record_cache_hit(elapsed.count());
          LOG_INFO("singleflight: merged key={}", ns_key);
          return annotate_cache_status(merged, "hit");
        }
      } else {
        LOG_DEBUG("singleflight: wait timeout for key={}", ns_key);
      }
    }
    sf_slot = sf->insert(ns_key, cached_embedding);
  }

  // 5. 缓存未命中 转发 LLM
  auto result = call_llm(cfg.backend.url, cfg.backend.api_key,
                         request_body, cfg.backend.timeout_seconds);

  // 6. singleflight 完成/取消（仅当本请求仍是该 key 的 leader）
  bool ok = (result.status_code >= 200 && result.status_code < 300);
  if (sf_slot) {
    if (ok) sf->complete(ns_key, result.body, sf_slot);
    else sf->cancel(ns_key, sf_slot);
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);

  // 7. 解析 token 用量
  int prompt_tokens = 0, completion_tokens = 0;
  if (result.status_code >= 200 && result.status_code < 300) {
    try {
      auto resp = json::parse(result.body);
      auto& usage = resp.at("usage");
      prompt_tokens = usage.value("prompt_tokens", 0);
      completion_tokens = usage.value("completion_tokens", 0);
    } catch (...) {}
  }

  // 8. 写入缓存
  if (ok && cfg.cache.enabled && !user_msg.empty())
    engine->cache_reply(user_msg, result.body, cached_embedding,
                        extract_namespace(request_body));

  // 9. 统计
  stats->record_api_call(elapsed.count(), prompt_tokens, completion_tokens);

  if (ok)
    LOG_INFO("{} {} {}ms", result.status_code, result.body.size(), elapsed.count());
  else
    LOG_WARN("{} {} {}ms", result.status_code,
             result.body.size() > 0 ? result.body : "(empty)", elapsed.count());

  // 10. 输出过滤
  auto out_result = filter->check_output(result.body);
  if (out_result.action == FilterAction::kReject) {
    LOG_WARN("filter: rejected output containing URL");
    return R"({"error":"Response filtered"})";
  }
  return annotate_cache_status(out_result.sanitized, "miss");
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
    // 加载后立即清理过期条目：落盘时仍有效、此后超过 TTL 的条目不应继续占用内存与索引
    size_t purged_on_load = lru->purge_expired();
    // 索引由 LruStore 重建，无需独立加载;
    engine->rebuild_index();
    if (lru->size() == 0) {
      LOG_INFO("cache restored: 0 entries (缓存为空或条目均已超过 ttl_days={})",
               cfg.cache.ttl_days);
    } else {
      LOG_INFO("cache restored: {} entries, {} vectors{}", lru->size(), idx->size(),
               purged_on_load > 0
                   ? std::format(", {} expired purged", purged_on_load)
                   : "");
    }
  }

  struct sigaction sa{};
  sa.sa_handler = handle_signal;
  sa.sa_flags = SA_RESTART;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
  // SIGUSR1：按需导出统计到日志（`kill -USR1 <pid>`，最迟 60s 由后台线程输出）
  sigaction(SIGUSR1, &sa, nullptr);

  // ---- 3. 启动定期统计 + 定时持久化线程 ----
  std::thread bg_thread([stats, lru, idx, engine, &cfg] {
    while (!g_shutdown.load(std::memory_order_acquire)) {
      {
        std::unique_lock lk(g_bg_mutex);
        g_bg_cv.wait_for(lk, 60s,
                         [] { return g_shutdown.load(std::memory_order_acquire); });
      }
      if (g_shutdown.load(std::memory_order_acquire)) break;
      if (g_dump_stats.exchange(false, std::memory_order_acq_rel))
        LOG_INFO("stats dump requested by SIGUSR1");
      stats->report();
      if (cfg.cache.enabled) engine->try_rebuild_if_ghosty();
      // 主动清理过期条目：避免失效条目长期占用内存，并让落盘内容只含有效条目
      if (cfg.cache.enabled) {
        size_t purged = lru->purge_expired();
        if (purged > 0) {
          // HNSW 无删除接口：清理后重建索引，保持索引与 LruStore 一致
          engine->rebuild_index();
          LOG_INFO("cache: purged {} expired entries, {} remaining", purged,
                   lru->size());
        }
      }
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
    // 兜底：线程池 worker 里逃出的异常会直接 std::terminate 整个进程，
    // 因此任何异常都必须在这里被拦住并转成一个普通错误响应
    try {
      return handle_request(body, cfg, filter.get(), engine.get(),
                            &flight_merge, stats.get());
    } catch (const std::exception& e) {
      LOG_ERROR("request handler threw: {}", e.what());
      return std::string(R"({"error":"Internal error"})");
    } catch (...) {
      LOG_ERROR("request handler threw non-std exception");
      return std::string(R"({"error":"Internal error"})");
    }
  });

  // ---- 5. 启动 ----
  LOG_INFO("ai-gateway starting on :{}, backend={}", cfg.server.port, cfg.backend.url);
  // 传入关闭标志：否则 SIGTERM/SIGINT 只置位而无人检查，优雅退出（保存缓存）永不执行
  server.run(&g_shutdown);

  // ---- 6. 清理：保存缓存 + 统计 ----
  g_shutdown.store(true, std::memory_order_release);
  g_bg_cv.notify_one();  // 唤醒 bg_thread 避免等待 60s 超时
  if (bg_thread.joinable()) bg_thread.join();
  stats->report();
  if (cfg.cache.enabled) {
    size_t purged = lru->purge_expired();
    lru->save("cache/lru_store.json");
    // 持久化空桩（HNSW 索引通过 LruStore 重建）;
    LOG_INFO("cache persisted: {} entries, {} vectors{}", lru->size(), idx->size(),
             purged > 0 ? std::format(", {} expired purged", purged) : "");
  }
  LOG_INFO("cache hits={} misses={} hit_rate={:.1f}%",
           stats->cache_hits(), stats->cache_misses(),
           stats->hit_rate() * 100);
  LOG_INFO("ai-gateway stopped");
  return 0;
}
