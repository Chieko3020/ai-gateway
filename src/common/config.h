// JSON 配置加载：从文件读取网关配置，支持环境变量覆盖 API Key
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai_gateway {

// 服务端配置
struct ServerConfig {
  int port = 9000;
};

// LLM 后端配置（OpenAI 兼容）
struct BackendConfig {
  std::string url;         // https://api.deepseek.com/v1/chat/completions
  std::string api_key;     // 从环境变量或配置文件读取
  std::string model;       // deepseek-v4-flash
  int timeout_seconds = 60;
};

// Embedding 后端配置
struct EmbeddingConfig {
  std::string url;         // http://127.0.0.1:8081/v1/embeddings
  std::string api_key;
  std::string model;       // bge-small-zh-v1.5
};

// 缓存配置
struct CacheConfig {
  bool enabled = true;
  float similarity_threshold = 0.85f;
  int max_entries = 10000;
  int ttl_days = 7;
};

// 安全过滤器配置
struct FilterConfig {
  int max_input_chars = 500;
  int max_output_chars = 600;
  bool block_urls = true;
  std::vector<std::string> blocked_keywords;
};

// 网关总配置
struct GatewayConfig {
  ServerConfig server;
  BackendConfig backend;
  EmbeddingConfig embedding;
  CacheConfig cache;
  FilterConfig filter;

  static int load(const std::string& path, GatewayConfig& out);
};

}  // namespace ai_gateway
