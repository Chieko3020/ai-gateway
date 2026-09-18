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

// FNV-1a 64-bit 确定性哈希：保证跨进程一致（std::hash 有随机种子，重启后同一
// system prompt 会算出不同 namespace，缓存隔离失效）。
// 32 位版本碰撞概率虽低但非零，碰撞即"不同 system prompt 共用一个缓存命名空间"，
// 后果是跨对话串答案，因此升到 64 位（报告 L4）
static uint64_t fnv1a_64(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ull;
  }
  return h;
}

// 把一条 message 的 content 拍平成纯文本：
//   content 为 string       → 原样
//   content 为数组（多模态） → 拼接各 text 片段（image_url 等非文本片段跳过）
//   content 缺失/其它类型    → 空串
// 旧实现直接 value("content","")：数组形式会抛 type_error 被吞掉，
// 于是多模态请求的 check_input 根本不执行（报告 M11 的绕过面）
static std::string message_text(const json& msg) {
  if (!msg.is_object()) return "";
  auto it = msg.find("content");
  if (it == msg.end()) return "";
  if (it->is_string()) return it->get<std::string>();
  if (it->is_array()) {
    std::string out;
    for (const auto& part : *it) {
      if (part.is_string()) {
        out += part.get<std::string>();
        continue;
      }
      if (!part.is_object()) continue;
      auto t = part.find("text");
      if (t != part.end() && t->is_string()) out += t->get<std::string>();
    }
    return out;
  }
  return "";
}

// 提取 system 消息内容用于缓存隔离：扫描全部 messages（system 不在首位时
// 旧实现会静默丢失隔离），拼接多段 system 文本后取 FNV-1a 64 位。
// 返回值形如 "ns<16 位 hex>"：
//   - 带 ns 前缀，避免与"客户端消息本身恰好是 16 位十六进制数"撞进同一 key 空间
//   - 位宽变化会让已有落盘缓存的 namespace 前缀全部改变（一次性失配，见简报）
static std::string extract_namespace(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    auto& msgs = req.at("messages");
    std::string system_text;
    for (const auto& m : msgs) {
      if (!m.is_object()) continue;
      if (m.value("role", "") != "system") continue;
      auto text = message_text(m);
      if (text.empty()) continue;
      if (!system_text.empty()) system_text += "\n";
      system_text += text;
    }
    if (!system_text.empty())
      return std::format("ns{:016x}", fnv1a_64(system_text));
  } catch (...) {}
  return "";  // 客户端没有 system message 或解析失败，返回空字符串表示不做隔离
}

// 从 OpenAI 格式请求体中提取最后一条 user message（用于缓存键与向量化）
static std::string extract_user_message(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    auto& msgs = req.at("messages");
    for (auto it = msgs.rbegin(); it != msgs.rend(); ++it) {
      if (!it->is_object()) continue;
      if (it->value("role", "") == "user") return message_text(*it);
    }
  } catch (...) {}
  return "";
}

// 拼接所有 message 的文本：输入过滤的检查面（旧实现只查最后一条 user 消息）
static std::string collect_all_text(const std::string& request_body) {
  std::string out;
  try {
    auto req = json::parse(request_body);
    auto it = req.find("messages");
    if (it == req.end() || !it->is_array()) return "";
    for (const auto& m : *it) {
      auto text = message_text(m);
      if (text.empty()) continue;
      if (!out.empty()) out += "\n";
      out += text;
    }
  } catch (...) {}
  return out;
}

// 请求分类：旁路（tool 类）与拒绝（stream）是两件事，必须分开判定。
// 旧实现把二者合并成一个 is_uncacheable_request()，于是 stream:true 也走
// "转发给上游再把整段缓冲的 SSE 文本塞进 JSON 信封"的伪透传路径。
static bool wants_stream(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    return req.contains("stream") && req.value("stream", false);
  } catch (...) {}
  return false;
}

// 工具调用类请求（tools/functions 定义、tool/function 角色的消息、tool_calls）：
// 与上下文强相关，缓存会丢失 tool_calls 或返回不适用答案，因此不缓存、不参与合并，
// 但仍然走输入/输出过滤并按上游状态码透传。
static bool is_tool_request(const std::string& request_body) {
  try {
    auto req = json::parse(request_body);
    if (req.contains("tools") || req.contains("functions")) return true;
    auto& msgs = req.at("messages");
    for (const auto& m : msgs) {
      auto role = m.value("role", "");
      if (role == "tool" || role == "function") return true;
      if (m.contains("tool_calls") || m.contains("function_call")) return true;
    }
  } catch (...) {}
  return false;
}

// stream:true 明确拒绝，而不是"转发后当 JSON 回包"。
//
// 背景：真正的 SSE 透传需要 llm_client 增量回调、response 去掉 Content-Length
// 并逐块下发、输出过滤按 SSE 事件边界判定（当前 llm_client 用 curl_easy_perform
// 整段缓冲，response.cpp 恒发 Content-Length + Connection: close）。在透传落地
// 之前，把 SSE 文本塞进 application/json 信封是"伪支持"：客户端解析失败且无法
// 增量渲染。因此这里返回 400，把不可用变成可诊断。
//
// 这也是 DSH 的 LLM 层无法用本网关做 provider 的原因：@earendil-works/pi-ai 在
// openai-completions API 里硬编码 stream: true
// （~/.dsh/profiles/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js:587），
// 于是它的每个请求都会命中这个分支。真透传是后续待办，
// 见 research/personal/ai-gateway-backlog.md 第 1 节。
static constexpr const char* kStreamUnsupported =
    R"({"error":"streaming (stream=true) is not supported yet"})";

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

// 响应体/消息的短摘要：日志里不落原文，只落"长度 + 64 位哈希"，
// 既能给排查用的关联标识，又不泄露用户内容（报告 M5）
static std::string body_digest(const std::string& s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ull;
  }
  return std::format("len={},h={:016x}", s.size(), h);
}

// 网关自身生成的 JSON 响应（默认 200；拒绝类响应给出语义正确的状态码：
// 输入被拒 400、上游内容被拦 502，见 handle_request）
static HttpReply json_reply(std::string body, int status_code = 200) {
  return HttpReply{status_code, "application/json", std::move(body)};
}

// 透传上游状态码：curl 自身失败时 llm_client 已映射为 502/504，
// 这里再兜一层，确保不会出现 status_code=0 被当成正常码发回客户端
static int upstream_status(int status_code) {
  return status_code > 0 ? status_code : 502;
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
// 返回状态码 + 响应体：上游 4xx/5xx 原样透传（见 HttpReply）
static HttpReply handle_request(const std::string& request_body,
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
  std::string filter_text = collect_all_text(request_body);
  if (!filter_text.empty()) {
    auto f_result = filter->check_input(filter_text);
    if (f_result.action == FilterAction::kReject) {
      LOG_WARN("filter: rejected input: {}", f_result.reject_msg);
      // 输入被拒是客户端错误（旧实现返回 200 + error body，调用方无法据此重试/降级）
      return json_reply(R"({"error":"Request rejected"})", 400);
    }
    if (f_result.action == FilterAction::kTruncate)
      user_msg = f_result.sanitized;
  }

  // 2. stream:true：明确拒绝（不转发到上游）。
  //    放在输入过滤之后，保证被拒请求同样经过安全过滤器。
  if (wants_stream(request_body)) {
    LOG_WARN("cache: reject (stream) stream=true not supported yet");
    return json_reply(kStreamUnsupported, 400);
  }

  // 3. 工具调用类流量：不查缓存、不写缓存、不参与请求合并，但仍走双向过滤与状态码透传
  if (is_tool_request(request_body)) {
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
    LOG_INFO_SAMPLED("cache: bypass (tool) status={} {}ms",
                     result.status_code, elapsed.count());

    auto out_result = filter->check_output(result.body);
    if (out_result.action == FilterAction::kReject) {
      LOG_WARN("filter: rejected output containing URL");
      // 上游内容无法交付给客户端：502（既非客户端错误，也不该沿用上游状态码）
      return json_reply(R"({"error":"Response filtered"})", 502);
    }
    return json_reply(annotate_cache_status(out_result.sanitized, "bypass"),
                      upstream_status(result.status_code));
  }

  // 缓存命中检查时带回的 embedding（避免 cache_reply 重复计算）
  std::vector<float> cached_embedding;

  // 4. 语义缓存
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
        return json_reply(R"({"error":"Response filtered"})", 502);
      }
      return json_reply(annotate_cache_status(hit_out.sanitized, "hit"));
    }
    cached_embedding = std::move(hit.embedding);
  }

  // 5. 请求合并（singleflight）
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
          // 合并命中不是缓存命中：单独计数、单独状态（报告 M10）。
          // 日志只落长度与哈希，不落 key 原文（= namespace:完整用户消息，报告 M5）
          stats->record_merge(elapsed.count());
          LOG_INFO_SAMPLED("singleflight: merged {}", body_digest(ns_key));
          // 合并回来的同样是上游原文，必须与未命中路径一样过输出过滤
          auto merged_out = filter->check_output(merged);
          if (merged_out.action == FilterAction::kReject) {
            LOG_WARN("filter: rejected merged output containing URL");
            return json_reply(R"({"error":"Response filtered"})", 502);
          }
          return json_reply(annotate_cache_status(merged_out.sanitized, "merged"));
        }
      } else {
        LOG_DEBUG("singleflight: wait timeout (src_len={})", ns_key.size());
      }
    }
    sf_slot = sf->insert(ns_key, cached_embedding);
  }

  // 6. 缓存未命中 转发 LLM
  auto result = call_llm(cfg.backend.url, cfg.backend.api_key,
                         request_body, cfg.backend.timeout_seconds);

  // 7. singleflight 完成/取消（仅当本请求仍是该 key 的 leader）
  bool ok = (result.status_code >= 200 && result.status_code < 300);
  if (sf_slot) {
    if (ok) sf->complete(ns_key, result.body, sf_slot);
    else sf->cancel(ns_key, sf_slot);
  }

  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0);

  // 8. 解析 token 用量
  int prompt_tokens = 0, completion_tokens = 0;
  if (result.status_code >= 200 && result.status_code < 300) {
    try {
      auto resp = json::parse(result.body);
      auto& usage = resp.at("usage");
      prompt_tokens = usage.value("prompt_tokens", 0);
      completion_tokens = usage.value("completion_tokens", 0);
    } catch (...) {}
  }

  // 9. 写入缓存
  if (ok && cfg.cache.enabled && !user_msg.empty())
    engine->cache_reply(user_msg, result.body, cached_embedding,
                        extract_namespace(request_body));

  // 10. 统计
  stats->record_api_call(elapsed.count(), prompt_tokens, completion_tokens);

  // 每请求一条 INFO 属热路径：默认全量输出，可用 log_sample_every 采样降级。
  // 只记长度与短摘要，不记上游响应体原文（可能含用户数据，报告 M5）
  if (ok)
    LOG_INFO_SAMPLED("{} {} bytes {}ms", result.status_code, result.body.size(),
                     elapsed.count());
  else
    LOG_WARN("upstream {} {} bytes {}ms body_digest={}", result.status_code,
             result.body.size(), elapsed.count(), body_digest(result.body));

  // 11. 输出过滤
  auto out_result = filter->check_output(result.body);
  if (out_result.action == FilterAction::kReject) {
    LOG_WARN("filter: rejected output containing URL");
    return json_reply(R"({"error":"Response filtered"})", 502);
  }
  // 上游错误码（4xx/5xx/502/504）原样透传，不再一律 200
  return json_reply(annotate_cache_status(out_result.sanitized, "miss"),
                    upstream_status(result.status_code));
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
  // 热路径日志采样：必须在服务开始处理请求前生效（默认 1 = 全量）
  ai_gateway::detail::set_log_sample_every(cfg.log.sample_every);
  if (cfg.log.sample_every > 1)
    LOG_WARN("log sampling enabled: 1 of every {} hot-path INFO lines is kept",
             cfg.log.sample_every);

  // ---- 2. 初始化模块 ----
  // ttl_days * 86400 是 int 乘法：ttl_days > 24855 会溢出成负数，
  // 于是"永不过期"被静默打开（报告 L5）
  constexpr int kMaxTtlDays = 3650;  // 10 年，超出视为配置错误
  if (cfg.cache.ttl_days < 0 || cfg.cache.ttl_days > kMaxTtlDays) {
    LOG_ERROR("cache.ttl_days={} out of range [0, {}]", cfg.cache.ttl_days,
              kMaxTtlDays);
    return static_cast<int>(ErrorCode::kConfigError);
  }
  const int64_t ttl_seconds = static_cast<int64_t>(cfg.cache.ttl_days) * 86400;
  auto lru = std::make_shared<LruStore>(cfg.cache.max_entries, ttl_seconds);

  // 本地 ONNX 嵌入推理：模型路径与维度来自 embedding 配置段（不再硬编码，
  // 否则示例里的模型名永远不会生效，报告 M15）
  auto onnx_embed = std::make_shared<OnnxEmbedding>(
      cfg.embedding.model_path, cfg.embedding.vocab_path, cfg.embedding.dim);

  // 维度不符属配置错误：启动即失败（旧实现按 min(dims, out_dim) 静默截断，
  // 索引维度与配置声明不一致且没有任何告警，报告 M15）。
  // 模型文件缺失等不可用情形仍降级为精确匹配，避免"没有模型就不能跑"
  if (onnx_embed->load_error() ==
      OnnxEmbedding::LoadError::kDimensionMismatch) {
    LOG_ERROR("embedding.dim={} does not match {} (model output dim={}), "
              "fix config or model", cfg.embedding.dim,
              cfg.embedding.model_path, onnx_embed->output_dim());
    return static_cast<int>(ErrorCode::kConfigError);
  }
  if (!onnx_embed->ready()) {
    LOG_WARN("onnx embedding unavailable ({}), semantic cache degrades to "
             "exact match", cfg.embedding.model_path);
  }

  auto embed_fn = [onnx_embed](const std::string& text,
                                int) -> std::vector<float> {
    return onnx_embed->ready() ? onnx_embed->encode(text)
                                : std::vector<float>{};
  };

  // 索引由缓存引擎独占持有：rebuild_index() 会替换 index_，main 不再保留副本，
  // 否则会长期持有一个已失效的旧索引对象（日志里的向量数也会取自旧对象）
  auto engine = std::make_shared<CacheEngine>(
      cfg.embedding, cfg.cache, lru,
      std::make_shared<HnswIndex>(
          HnswConfig{cfg.embedding.dim, 16, 100, 50}),
      embed_fn);
  auto stats = std::make_shared<Stats>();
  auto filter = std::make_shared<MessageFilter>(cfg.filter);

  // 从磁盘恢复缓存
  if (cfg.cache.enabled) {
    // 返回值必须检查：路径不可读/文件损坏时 load 会失败，静默忽略会让"重启不丢缓存"
    // 的说法失真（报告 M9）
    if (!lru->load("cache/lru_store.json"))
      LOG_WARN("cache: load from cache/lru_store.json failed "
               "(missing or malformed), starting empty");
    // 加载后立即清理过期条目：落盘时仍有效、此后超过 TTL 的条目不应继续占用内存与索引
    size_t purged_on_load = lru->purge_expired();
    // 索引由 LruStore 重建，无需独立加载;
    engine->rebuild_index();
    if (lru->size() == 0) {
      LOG_INFO("cache restored: 0 entries (缓存为空或条目均已超过 ttl_days={})",
               cfg.cache.ttl_days);
    } else {
      LOG_INFO("cache restored: {} entries, {} vectors{}", lru->size(),
               engine->index_size(),
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
  std::thread bg_thread([stats, lru, engine, &cfg] {
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
        if (!lru->save("cache/lru_store.json"))
          LOG_ERROR("cache: periodic save to cache/lru_store.json failed");
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
      return json_reply(R"({"error":"Internal error"})", 500);
    } catch (...) {
      LOG_ERROR("request handler threw non-std exception");
      return json_reply(R"({"error":"Internal error"})", 500);
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
    if (!lru->save("cache/lru_store.json"))
      LOG_ERROR("cache: final save to cache/lru_store.json failed "
                "({} entries were not persisted)",
                lru->size());
    // 持久化空桩（HNSW 索引通过 LruStore 重建）;
    LOG_INFO("cache persisted: {} entries, {} vectors{}", lru->size(),
             engine->index_size(),
             purged > 0 ? std::format(", {} expired purged", purged) : "");
  }
  LOG_INFO("cache hits={} misses={} hit_rate={:.1f}%",
           stats->cache_hits(), stats->cache_misses(),
           stats->hit_rate() * 100);
  LOG_INFO("ai-gateway stopped");
  return 0;
}
