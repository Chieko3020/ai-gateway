// JSON 配置加载：从文件读取网关配置，支持环境变量覆盖 API Key
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ai_gateway {

// 服务端配置
struct ServerConfig {
  int port = 9000;
  uint64_t max_body_bytes = 20 * 1024 * 1024;  // 请求体最大字节数，默认 20MB
  // 并发连接上限：超出时对新连接直接回 503 并关闭，避免 fd 被耗尽后 accept 全面失败
  size_t max_connections = 256;
  // 单连接空闲超时（秒）：超时未收到任何新字节即关闭。
  // 没有这个上限时，"只发一半请求就半关闭写端"的连接会永久驻留 fd 与缓冲区（报告 H5）。
  // 取 30s：真实客户端的首字节与慢速上传都可能停顿数秒，默认值过小会误杀正常请求
  // （半关闭与已收齐的请求不依赖它——分别在 EOF 与提交时立刻处理）
  int idle_timeout_seconds = 30;
  // 头部区段最大字节数（slowloris 防护）
  size_t max_header_bytes = 65536;
};

// LLM 后端配置（OpenAI 兼容）
struct BackendConfig {
  std::string url;         // https://api.deepseek.com/v1/chat/completions
  std::string api_key;     // 从环境变量或配置文件读取
  std::string model;       // deepseek-v4-flash
  int timeout_seconds = 60;
};

// Embedding 配置：真正驱动进程内 ONNX 推理的模型路径与维度。
// 旧实现保留了 url/api_key/model 三个"死配置"——main 完全忽略它们，照样加载
// 硬编码的 model/model_int8.onnx，示例里却写着 BAAI/bge-large-zh-v1.5（1024 维），
// 用户照抄配置不会生效（报告 M15）
struct EmbeddingConfig {
  std::string model_path = "model/model_int8.onnx";  // ONNX 模型文件
  std::string vocab_path = "model/vocab.txt";        // WordPiece 词表
  int dim = 512;  // 期望的向量维度：与模型实际输出不符时启动即失败，不静默截断
};

// 缓存配置
struct CacheConfig {
  bool enabled = true;
  float similarity_threshold = 0.85f;
  int max_entries = 10000;
  int ttl_days = 7;
};

// 安全过滤器配置
// 注意：max_input_chars / max_output_chars <= 0 表示不截断。
// max_output_chars 的旧默认值 600 会把绝大多数正常回答静默截断到 600 字节，
// 对"透明代理"定位是错误默认值，因此默认改为 0（不截断，报告 L9）
struct FilterConfig {
  int max_input_chars = 500;
  int max_output_chars = 0;
  bool block_urls = true;
  std::vector<std::string> blocked_keywords;
};

// 日志配置
struct LogConfig {
  // 每请求 INFO 的采样率：1 = 全量（默认），N > 1 = 每 N 条只留 1 条。
  // WARN/ERROR 永不采样。热路径日志是全局串行热点，高并发下应调大该值
  uint64_t sample_every = 1;
};

// 网关总配置
struct GatewayConfig {
  ServerConfig server;
  BackendConfig backend;
  EmbeddingConfig embedding;
  CacheConfig cache;
  FilterConfig filter;
  LogConfig log;

  static int load(const std::string& path, GatewayConfig& out);
};

}  // namespace ai_gateway
