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
  // 非流式响应的写超时（秒，必须 > 0）：worker 在大响应 + 慢客户端（非阻塞 fd 上
  // EAGAIN）时最多等这么久把响应写完，超时则断开。
  // 取值就是"一个慢客户端最多占住一个 worker 多久"。注意它作用于**响应头 + 正文
  // 写完**这一段（对非流式响应是整条响应的总死线），流式响应不走它——
  // 流式用 stream_idle_timeout_seconds 逐次续期（见下）
  //
  // 默认 60s：旧默认 10s 是"整条响应"的死线，会把生成超过 10s 的长回答直接切断
  // （上游 timeout_seconds 默认也是 60s，两者同量级才自洽）
  int write_timeout_seconds = 60;
  // 流式（SSE）响应的**空闲死线**（秒，必须 > 0）：两次成功写入之间的间隔超过
  // 它就中停上游并断开，而**不是**给整条流设总时限。
  // 为什么要区分：LLM 长回答的流式输出动辄几十秒到几分钟（输出 4K token 就超过
  // 60s），任何"整条流的总死线"都等于给回答长度设上限。真正需要防的是客户端
  // 长时间不读（一个连接把 worker 占死），那属于"写入之间没有进展"。
  // 每次实际写出一个 chunk 就续期，因此只要客户端在跟读，流要多长都行
  int stream_idle_timeout_seconds = 60;
};

// LLM 后端配置（OpenAI 兼容）
struct BackendConfig {
  std::string url;         // https://api.deepseek.com/v1/chat/completions
  std::string api_key;     // 从环境变量或配置文件读取
  std::string model;       // deepseek-v4-flash
  int timeout_seconds = 60;
};

// 向量池化方式：取 BERT last_hidden_state 的哪一部分作为句向量。
// 官方 bge-small-zh-v1.5 用的是 CLS（见 EmbeddingConfig::pooling 的注释），
// Mean 保留用于复现历史行为与对照实验
enum class PoolingMode {
  kCls = 0,   // last_hidden_state[:, 0, :]（[CLS] 位）
  kMean = 1,  // 按 attention_mask 加权的 token 均值
};

// Embedding 配置：真正驱动进程内 ONNX 推理的模型路径与维度。
// 旧实现保留了 url/api_key/model 三个"死配置"——main 完全忽略它们，照样加载
// 硬编码的 model/model_int8.onnx，示例里却写着 BAAI/bge-large-zh-v1.5（1024 维），
// 用户照抄配置不会生效（报告 M15）
struct EmbeddingConfig {
  std::string model_path = "model/model_int8.onnx";  // ONNX 模型文件
  std::string vocab_path = "model/vocab.txt";        // WordPiece 词表
  int dim = 512;  // 期望的向量维度：与模型实际输出不符时启动即失败，不静默截断

  // 是否做小写化（BERT BasicTokenizer 的 do_lower_case）。
  //
  // 默认 true：模型自带的 sentence_bert_config.json 写的就是 `do_lower_case: true`，
  // 也即 sentence-transformers 加载 BAAI/bge-small-zh-v1.5 时的实际行为。
  // 取值不是"风格偏好"，而是**信息是否在分词层就丢失**：
  //   本词表只有小写英文（`dns` 在、`dma`/`DNS`/`DMA` 都不在）。
  //   do_lower_case=false 时 `什么是DMA` 与 `什么是DNS` 的 token 序列**完全相同**
  //   （"DMA"/"DNS" 都整词回退成 [UNK]）→ 余弦 1.0000，任何阈值都拦不住；
  //   do_lower_case=true 时同一对是 0.5797（`dma`/`dns` 分别命中 `##ma`/`##ns` 子词），
  //   安全低于阈值。
  // 保留 false 可切换：切换会改变分词结果 → 旧向量失效，由 embedding 指纹处理
  bool do_lower_case = true;

  // 池化方式（官方 1_Pooling/config.json 是 pooling_mode_cls_token=true、
  // pooling_mode_mean_tokens=false，modules.json 是 Transformer→Pooling→Normalize，
  // 模型卡明确写 "select the last hidden state of the first token ([CLS])"）。
  // 默认 kCls 对齐官方；kMean 保留是因为旧落盘向量是按 mean 产生的，便于对照复现。
  // 实测差异（LCQMC 验证集，阈值 0.85）：lower+cls 召回 88.6% / 误命中 30.4% / F1 80.5%；
  // lower+mean 召回 85.4% / 误命中 33.1% / F1 79.1%（数值见 scripts/eval_semantic_cache.py）
  PoolingMode pooling = PoolingMode::kCls;
};

// 缓存配置
struct CacheConfig {
  bool enabled = true;
  // 余弦相似度阈值。默认 0.85 而非 0.80：LCQMC 验证集上的阈值扫描
  // （case+mean / lower+cls 两种设置都跑过）显示 0.85 是 F1 最高点
  // （召回 88.6% / 误命中 30.4% / F1 80.5%），0.80 时误命中率接近 50%、
  // 0.90 时召回掉到 72.1%。数值见 scripts/eval_semantic_cache.py
  float similarity_threshold = 0.85f;
  // 实体一致性否决（见 cache/entity_tokens.h）：命中候选与查询在数字/大写缩略语/
  // 混合标识符上不一致时**否决**这次命中。阈值救不了这类反例
  // （`继续下一题` ↔ `继续12题` 余弦 0.885；`什么是DMA` ↔ `什么是DNS` 旧配置下 1.0000），
  // 但实体硬约束能拦住。默认开启：误伤面实测很小（LCQMC 3000 对上 F1 不降反升）
  bool entity_veto = true;
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
  // 单文件大小上限：达到后 gateway.log → gateway.log.1，旧的依次后移，
  // 最老的一份删除。0 = 关闭轮转（回到"单文件一直追加"的旧行为）
  size_t max_bytes = 10 * 1024 * 1024;  // 10MB
  int keep_files = 5;                   // 保留历史份数（不含当前文件）
};

// 费用估算配置（元 / 1K tokens）。
// 缺省值与旧口径一致（0.001/0.001 统一单价），因此不配置时报表金额不变；
// 要按真实分档计价就显式写 cost 段（DeepSeek v4-flash：输入 ¥0.001、输出 ¥0.004）
struct CostConfig {
  double input_per_1k = 0.001;
  double output_per_1k = 0.001;
};

// 网关总配置
struct GatewayConfig {
  ServerConfig server;
  BackendConfig backend;
  EmbeddingConfig embedding;
  CacheConfig cache;
  FilterConfig filter;
  LogConfig log;
  CostConfig cost;

  static int load(const std::string& path, GatewayConfig& out);
};

}  // namespace ai_gateway
