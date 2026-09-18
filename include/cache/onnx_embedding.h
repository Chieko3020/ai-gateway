// 本地 ONNX Runtime 嵌入推理：替代之前 python 开启的 HTTP Embedding 服务
//
// 使用 ONNX Runtime C API 直接加载量化模型、分词、推理
// 比 Flask HTTP 服务省 ~20ms（无 HTTP/JSON 序列化开销）
//
// 依赖：libonnxruntime.so（系统动态库，pip 安装）
// 模型：bge-small-zh-v1.5-onnx/model_int8.onnx (INT8 量化, ~23MB)
// 词表：bge-small-zh-v1.5/vocab.txt (WordPiece 词表, 21128 词条)

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <onnxruntime_c_api.h>

namespace ai_gateway {
// 先分词后推理
// 贪心最长子串匹配分词器
// 基于 BERT WordPiece 词表，逐位置向后扫描取最长匹配 token
// 与标准 WordPiece 的区别：标准 WordPiece 从开头逐 token 贪心匹配并跳过未登录字，
// 本实现从每个位置尝试最长子串，未匹配则标记为 [UNK]
class GreedyTokenizer {
 public:
  static constexpr size_t kMinVocabSize = 1000;
  bool load(const std::string& vocab_path);
  size_t size() const { return vocab_.size(); }
  // 返回 [CLS](101) + token_ids + [SEP](102)，截断到 max_len
  std::vector<int64_t> encode(std::string_view text, int max_len = 512);

 private:
  std::unordered_map<std::string, int> vocab_;
};

// ONNX Runtime 嵌入推理
class OnnxEmbedding {
 public:
  OnnxEmbedding(const std::string& model_path,
                const std::string& vocab_path,
                int dims = 512);
  ~OnnxEmbedding();

  OnnxEmbedding(const OnnxEmbedding&) = delete;
  OnnxEmbedding& operator=(const OnnxEmbedding&) = delete;

  bool ready() const { return session_ != nullptr; }
  std::vector<float> encode(std::string_view text);

  // 构造失败的原因：调用方据此决定"降级继续"还是"启动即失败"
  enum class LoadError {
    kNone = 0,           // 就绪
    kUnavailable,        // 模型/词表不存在或 ORT 失败 -> 可降级为精确匹配
    kDimensionMismatch,  // 模型可加载但输出维度 != 配置 -> 配置错误，启动即失败
  };
  LoadError load_error() const { return load_error_; }

  // 模型的真实输出维度（构造期探测得到；未就绪时为 0）。
  // 与构造参数 dims 不符时构造失败（ready()==false），不静默截断
  int output_dim() const { return output_dim_; }
  // 配置里期望的维度
  int dims() const { return dims_; }

 private:
  static const OrtApi* g_api_;  // 进程级单例
  OrtSession* session_ = nullptr;
  OrtEnv* env_ = nullptr;
  OrtMemoryInfo* mem_info_ = nullptr;
  GreedyTokenizer tokenizer_;
  int dims_ = 512;
  int output_dim_ = 0;  // 模型实际输出维度（探测得到）
  LoadError load_error_ = LoadError::kNone;
};

}  // namespace ai_gateway
