// 配置加载实现：nlohmann/json 解析 + 环境变量读取
#include "common/config.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "common/logger.h"

using json = nlohmann::json;

namespace ai_gateway {

namespace {

std::string resolve_api_key(const json& section) {
  // 1. 环境变量名（LLM_API_KEY_EMB / LLM_API_KEY）
  if (section.contains("api_key_env")) {
    const char* val = std::getenv(
        section["api_key_env"].get<std::string>().c_str());
    if (val && val[0] != '\0') return val;
  }
  // 2. 配置文件直写
  if (section.contains("api_key")) {
    auto key = section["api_key"].get<std::string>();
    if (!key.empty()) return key;
  }
  // 3. 通用环境变量
  const char* fallback = std::getenv("LLM_API_KEY");
  if (fallback && fallback[0] != '\0') return fallback;
  return "";
}

}  // namespace

int GatewayConfig::load(const std::string& path, GatewayConfig& out) {
  try {
    std::ifstream ifs(path);
    if (!ifs) {
      LOG_ERROR("cannot open config file: {}", path);
      return 1;
    }
    json root = json::parse(ifs);

    // --- server ---
    if (root.contains("server")) {
      auto& s = root["server"];
      out.server.port = s.value("port", 9000);
      out.server.max_body_bytes = s.value("max_body_bytes", 20 * 1024 * 1024ull);
      out.server.max_connections = s.value("max_connections", size_t{256});
      out.server.idle_timeout_seconds = s.value("idle_timeout_seconds", 30);
      out.server.max_header_bytes = s.value("max_header_bytes", size_t{65536});
      out.server.write_timeout_seconds = s.value("write_timeout_seconds", 60);
      // 0/负数会让"写超时"退化成立刻放弃（等价于旧的截断行为），夹到最小值 1
      if (out.server.write_timeout_seconds < 1) out.server.write_timeout_seconds = 1;
      out.server.stream_idle_timeout_seconds =
          s.value("stream_idle_timeout_seconds", 60);
      if (out.server.stream_idle_timeout_seconds < 1)
        out.server.stream_idle_timeout_seconds = 1;
    }

    // --- backend ---
    if (root.contains("backend")) {
      auto& b = root["backend"];
      out.backend.url = b.value("url", "");
      out.backend.model = b.value("model", "");
      out.backend.timeout_seconds = b.value("timeout_seconds", 60);
      out.backend.api_key = resolve_api_key(b);
    }

    // --- embedding ---
    if (root.contains("embedding")) {
      auto& e = root["embedding"];
      out.embedding.model_path = e.value("model_path", "model/model_int8.onnx");
      out.embedding.vocab_path = e.value("vocab_path", "model/vocab.txt");
      out.embedding.dim = e.value("dim", 512);
      if (out.embedding.dim <= 0) {
        LOG_ERROR("embedding.dim={} must be > 0", out.embedding.dim);
        return 1;
      }
      // 默认 true = 官方 sentence_bert_config.json 的取值（见 config.h 的说明）
      out.embedding.do_lower_case = e.value("do_lower_case", true);
      // 池化：只接受 "cls" / "mean"（大小写不敏感）。拼错时**启动即失败**，
      // 不静默退回默认值——池化方式决定了向量的语义，配错了只会表现为
      // "命中率变差"这种没有任何报错的慢性问题
      if (e.contains("pooling")) {
        if (!e["pooling"].is_string()) {
          LOG_ERROR("embedding.pooling must be a string (\"cls\" or \"mean\")");
          return 1;
        }
        std::string mode = e["pooling"].get<std::string>();
        for (char& c : mode)
          c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (mode == "cls") {
          out.embedding.pooling = PoolingMode::kCls;
        } else if (mode == "mean") {
          out.embedding.pooling = PoolingMode::kMean;
        } else {
          LOG_ERROR("embedding.pooling=\"{}\" is not one of \"cls\"/\"mean\"",
                    e["pooling"].get<std::string>());
          return 1;
        }
      }
    }

    // --- cache ---
    if (root.contains("cache")) {
      auto& c = root["cache"];
      out.cache.enabled = c.value("enabled", true);
      out.cache.similarity_threshold = c.value("similarity_threshold", 0.85f);
      out.cache.entity_veto = c.value("entity_veto", true);
      out.cache.max_entries = c.value("max_entries", 10000);
      out.cache.ttl_days = c.value("ttl_days", 7);
      out.cache.store_vectors = c.value("store_vectors", true);
    }

    // --- filter ---
    if (root.contains("filter")) {
      auto& f = root["filter"];
      out.filter.max_input_chars = f.value("max_input_chars", 500);
      out.filter.max_output_chars = f.value("max_output_chars", 0);
      out.filter.block_urls = f.value("block_urls", true);
      if (f.contains("blocked_keywords") && f["blocked_keywords"].is_array()) {
        out.filter.blocked_keywords.clear();  // 防止重复 load 累加
        for (auto& kw : f["blocked_keywords"]) {
          out.filter.blocked_keywords.push_back(kw.get<std::string>());
        }
      }
    }

    // --- log ---
    if (root.contains("log")) {
      auto& l = root["log"];
      out.log.sample_every = l.value("sample_every", uint64_t{1});
      if (out.log.sample_every == 0) out.log.sample_every = 1;
      out.log.max_bytes = l.value("max_bytes", size_t{10 * 1024 * 1024});
      out.log.keep_files = l.value("keep_files", 5);
      // 负份数会让"删除最老一份"的下标越界，夹到最小值 1
      if (out.log.keep_files < 1) out.log.keep_files = 1;
    }

    // --- cost ---
    // 缺省 = 旧的统一单价（0.001/0.001），配置只写一边时另一边沿用缺省；
    // 负价会让"费用"变成负数（等于给缓存刷收益），直接判为配置错误
    if (root.contains("cost")) {
      auto& c = root["cost"];
      out.cost.input_per_1k = c.value("input_per_1k", 0.001);
      out.cost.output_per_1k = c.value("output_per_1k", 0.001);
      if (out.cost.input_per_1k < 0 || out.cost.output_per_1k < 0) {
        LOG_ERROR("cost price must be >= 0 (input_per_1k={}, output_per_1k={})",
                  out.cost.input_per_1k, out.cost.output_per_1k);
        return 1;
      }
    }

    LOG_INFO("config loaded: port={}, backend={}, model={}, max_conn={}, "
             "idle_timeout={}s, log_rotate={} bytes keep={}, "
             "price=¥{}/1K in, ¥{}/1K out",
             out.server.port, out.backend.url, out.backend.model,
             out.server.max_connections, out.server.idle_timeout_seconds,
             out.log.max_bytes, out.log.keep_files,
             out.cost.input_per_1k, out.cost.output_per_1k);
    // 影响向量的三个开关单独落一行：它们决定"旧向量还能不能用"，
    // 出问题时这一行是唯一能还原现场的证据
    LOG_INFO("embedding: pooling={} do_lower_case={} dim={} | cache threshold={:.3f} "
             "entity_veto={} | write_deadline={}s stream_idle={}s",
             out.embedding.pooling == PoolingMode::kCls ? "cls" : "mean",
             out.embedding.do_lower_case ? "true" : "false", out.embedding.dim,
             out.cache.similarity_threshold, out.cache.entity_veto ? "on" : "off",
             out.server.write_timeout_seconds,
             out.server.stream_idle_timeout_seconds);
    return 0;
  } catch (const std::exception& e) {
    LOG_ERROR("config parse error: {}", e.what());
    return 1;
  }
}

}  // namespace ai_gateway
