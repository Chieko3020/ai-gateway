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
#include "cache/embedding_fingerprint.h"
#include "cache/lru_store.h"
#include "cache/hnsw_index.h"
#include "common/log_file.h"
#include "cache/onnx_embedding.h"
#include "common/config.h"
#include "common/logger.h"
#include "common/singleflight.h"
#include "common/types.h"
#include "server/filter.h"
#include "gateway/pipeline.h"
#include "server/http_server.h"
#include "server/metrics.h"
#include "server/response.h"
#include "server/sse_usage.h"
#include "stats/stats.h"

using namespace ai_gateway;
using json = nlohmann::json;
using namespace std::chrono_literals;

// 向量产生方式的完整标识（进 embedding 指纹）：
//   分词器实现 + 是否小写化 + 池化方式
// 这三者任何一个改变，**同一段文本产生的向量就不同**，旧落盘向量与新查询向量
// 不在同一个空间里，余弦相似度失去意义且不会报错。把它们拼进指纹，加载时会
// 判定不一致 → 丢弃旧向量、保留文本条目、按新配置重建索引。
static std::string embedding_variant_id(const EmbeddingConfig& emb) {
  return std::format("{}|lower={}|pooling={}", kTokenizerId,
                     emb.do_lower_case ? "1" : "0",
                     emb.pooling == PoolingMode::kCls ? "cls" : "mean");
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

// 进程生命周期：配置加载 → 模块装配 → 启动 HTTP 服务 → 优雅关闭（drain →
// 统计 → 落盘）。请求处理本身在 gateway/pipeline.cpp（与单测共享同一份代码）。
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
  // 日志轮转策略：必须在配置加载后立刻生效（否则配置里的 max_bytes 要等到
  // 第一条热路径日志才被读取）。若当前文件已超限，set_log_rotation 会先归档
  ai_gateway::detail::set_log_rotation(
      ai_gateway::detail::LogRotation{cfg.log.max_bytes, cfg.log.keep_files});
  // 热路径日志采样：必须在服务开始处理请求前生效（默认 1 = 全量）
  ai_gateway::detail::set_log_sample_every(cfg.log.sample_every);
  if (cfg.log.sample_every > 1)
    LOG_WARN("log sampling enabled: 1 of every {} hot-path INFO lines is kept",
             cfg.log.sample_every);
  LOG_INFO("log rotation: max_bytes={} keep_files={} (0 = disabled)",
           cfg.log.max_bytes, cfg.log.keep_files);

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
  // 分词大小写与池化方式必须在**任何 encode() 之前**设置：
  // 构造期做的那次"维度探测"推理也会用到它们，更关键的是随后计算指纹时
  // 要如实反映"这批向量是怎么产生的"（见 embedding_variant_id）
  onnx_embed->set_do_lower_case(cfg.embedding.do_lower_case);
  onnx_embed->set_pooling(cfg.embedding.pooling);

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
  // 输入/输出分档单价：配置缺省时是 0.001/0.001（与旧口径完全一致）
  stats->set_pricing(TokenPricing{cfg.cost.input_per_1k, cfg.cost.output_per_1k});
  auto filter = std::make_shared<MessageFilter>(cfg.filter);

  // ---- 2.5 embedding 指纹：向量只对"产生它的模型"有意义 ----
  // 换模型或改分词（本项目刚改过 WordPiece 规则）之后，旧向量与新查询向量不在
  // 同一个空间里，余弦相似度失去意义，且**不会报错**——只会静默返回错误答案。
  // 因此落盘时记录指纹、加载时比对；不一致就丢弃向量（保留文本）并按新模型重建。
  EmbeddingFingerprint fp;
  if (cfg.cache.enabled && onnx_embed->ready()) {
    const std::string variant = embedding_variant_id(cfg.embedding);
    fp = make_embedding_fingerprint(cfg.embedding.model_path,
                                    cfg.embedding.vocab_path, cfg.embedding.dim,
                                    variant);
    LOG_INFO("cache: embedding variant {}", variant);
    if (!fp.valid()) {
      // 模型能加载却算不出文件哈希（权限/IO 异常）：不能把空指纹当成"匹配"
      LOG_WARN("cache: embedding fingerprint unavailable "
               "(model={} vocab={}), vector origin will NOT be verified",
               cfg.embedding.model_path, cfg.embedding.vocab_path);
    } else {
      lru->set_fingerprint(fp.to_string());
      LOG_INFO("cache: embedding fingerprint {}", fp.to_string());
    }
  }

  // 从磁盘恢复缓存
  if (cfg.cache.enabled) {
    // 返回值必须检查：路径不可读/文件损坏时 load 会失败，静默忽略会让"重启不丢缓存"
    // 的说法失真（报告 M9）
    if (!lru->load("cache/lru_store.json"))
      LOG_WARN("cache: load from cache/lru_store.json failed "
               "(missing or malformed), starting empty");
    // 指纹判定结果必须显式落日志：这是"静默串答案"唯一的可观测面
    if (lru->fingerprint_mismatch()) {
      if (!lru->loaded_file_had_fingerprint()) {
        LOG_WARN("cache: lru_store.json has no embedding fingerprint (legacy "
                 "format): dropped {} vector(s), kept text entries; index will "
                 "be rebuilt with the current model",
                 lru->dropped_vectors());
      } else {
        LOG_WARN("cache: embedding fingerprint mismatch: file=[{}] current=[{}] "
                 "-> dropped {} vector(s), kept text entries, rebuilding index",
                 lru->loaded_fingerprint(),
                 fp.valid() ? fp.to_string() : std::string("<unavailable>"),
                 lru->dropped_vectors());
      }
    } else if (fp.valid() && lru->loaded_file_had_fingerprint()) {
      LOG_INFO("cache: embedding fingerprint verified ({} vector entries)",
               lru->size());
    }
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
  // handle_request 的返回值是 pair<HttpReply, keep_alive>，而路由层只接受
  // HttpReply（keep-alive 由连接处理器按请求头自行判定）。流式响应在
  // 连接处理器里走的是"writer.committed() 就直接返回"的旁路，不依赖这里返回的
  // HttpReply——因此 keep_alive 这一项在本路由上是多余的，丢弃即可
  server.set_handler([&](const std::string& body, ResponseWriter& writer,
                         const HttpRequestInfo& info) -> HttpReply {
    // 兜底：线程池 worker 里逃出的异常会直接 std::terminate 整个进程，
    // 因此任何异常都必须在这里被拦住并转成一个普通错误响应。
    // 注意：异常若发生在流式响应已经开始写之后，这里无法回退已发出的响应头
    // （客户端会看到一个被截断的流），只能保证进程存活——见简报"已知限制"
    try {
      // 客户端是否希望复用连接：由连接处理器从版本 + Connection 头解析后传入。
      // 响应头的 Connection 与"是否真的复用"必须用同一份意愿
      return handle_request(body, cfg, filter.get(), engine.get(),
                            &flight_merge, stats.get(), writer, info.keep_alive)
          .first;
    } catch (const std::exception& e) {
      LOG_ERROR("request handler threw: {}", e.what());
      return HttpReply{500, "application/json", R"({"error":"Internal error"})"};
    } catch (...) {
      LOG_ERROR("request handler threw non-std exception");
      return HttpReply{500, "application/json", R"({"error":"Internal error"})"};
    }
  });

  // GET /metrics：Prometheus 文本格式抓取端点。
  //   - 与 POST /v1/chat/completions 是两条独立路由（key = "METHOD /path"），
  //     新增它不改变既有行为（连接处理器只对"没有对应 method 路由的非 POST"
  //     返回 405，见 connection_handler.cpp）
  //   - 只暴露聚合计数与分位数，不含任何请求内容／哈希，脱敏口径与日志一致
  //   - 不读 active_connections()：conns_ 由 reactor 线程独占，worker 里读是数据竞争
  const auto process_start = std::chrono::steady_clock::now();
  server.add_route("GET", "/metrics",
                   [stats, &server, process_start](const std::string&,
                                                   ResponseWriter&,
                                                   const HttpRequestInfo&) {
                     auto uptime =
                         std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - process_start)
                             .count();
                     auto body = render_metrics(
                         *stats, uptime,
                         static_cast<int64_t>(server.pending_tasks()),
                         static_cast<int64_t>(server.active_tasks()),
                         static_cast<int64_t>(server.worker_threads()),
                         static_cast<int64_t>(server.accepted_connections()));
                     return HttpReply{200, kMetricsContentType, std::move(body)};
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
