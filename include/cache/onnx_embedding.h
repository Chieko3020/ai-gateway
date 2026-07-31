// 本地 ONNX Runtime 嵌入推理：替代之前 python 开启的 HTTP Embedding 服务
//
// 使用 ONNX Runtime C API 直接加载量化模型、分词、推理
// 比 Flask HTTP 服务省 ~20ms（无 HTTP/JSON 序列化开销）
//
// 依赖：libonnxruntime.so（系统动态库，pip 安装）
// 模型：bge-small-zh-v1.5-onnx/model_int8.onnx (需要INT8 量化, ~23MB)
// 词表：bge-small-zh-v1.5/vocab.txt

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <onnxruntime_c_api.h>

namespace ai_gateway {
// 先分词后推理
// 最小 BPE 分词器：加载 vocab.txt，UTF-8 逐字匹配
class BpeTokenizer {
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

 private:
  static const OrtApi* g_api_;  // 进程级单例
  OrtSession* session_ = nullptr;
  OrtEnv* env_ = nullptr;
  OrtMemoryInfo* mem_info_ = nullptr;
  BpeTokenizer tokenizer_;
  int dims_ = 512;
};

}  // namespace ai_gateway
