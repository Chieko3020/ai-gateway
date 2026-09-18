// 配置加载实现：nlohmann/json 解析 + 环境变量读取
#include "common/config.h"

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
    }

    // --- cache ---
    if (root.contains("cache")) {
      auto& c = root["cache"];
      out.cache.enabled = c.value("enabled", true);
      out.cache.similarity_threshold = c.value("similarity_threshold", 0.85f);
      out.cache.max_entries = c.value("max_entries", 10000);
      out.cache.ttl_days = c.value("ttl_days", 7);
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
    }

    LOG_INFO("config loaded: port={}, backend={}, model={}, max_conn={}, "
             "idle_timeout={}s",
             out.server.port, out.backend.url, out.backend.model,
             out.server.max_connections, out.server.idle_timeout_seconds);
    return 0;
  } catch (const std::exception& e) {
    LOG_ERROR("config parse error: {}", e.what());
    return 1;
  }
}

}  // namespace ai_gateway
