// ONNX Runtime 嵌入推理 & BPE 分词器实现
#include "cache/onnx_embedding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "common/logger.h"

namespace ai_gateway {

// ── BpeTokenizer ──────────────────────────────────────────

bool BpeTokenizer::load(const std::string& vocab_path) {
  std::ifstream f(vocab_path);
  if (!f) return false;
  vocab_.clear();
  std::string line;
  int id = 0;
  while (std::getline(f, line)) {
    // vocab.txt 每行一个 token，第一行通常是 [PAD]
    if (!line.empty()) vocab_[line] = id++;
  }
  return vocab_.size() > kMinVocabSize;  // 至少要有基本词条并且数量足够
}

std::vector<int64_t> BpeTokenizer::encode(std::string_view text, int max_len) {
  std::vector<int64_t> ids;
  ids.reserve(max_len);
  ids.push_back(101);  // [CLS]

  auto try_add = [&](std::string_view piece) {
    auto it = vocab_.find(std::string(piece));
    if (it != vocab_.end()) {
      ids.push_back(it->second);
      return true;
    }
    return false;
  };

  const auto* data = text.data();
  size_t i = 0, len = text.size();

  while (i < len && static_cast<int>(ids.size()) < max_len - 1) {
    // 尝试最长子串匹配（WordPiece 启发）
    bool found = false;
    int end = std::min(i + 10, len);
    for (int j = end; j > static_cast<int>(i); --j) {
      if (try_add(text.substr(i, j - i))) {
        i = j;
        found = true;
        break;
      }
    }
    if (!found) {
      ids.push_back(100);  // [UNK]
      i += 1;
    }
  }

  ids.push_back(102);  // [SEP]
  return ids;
}

// ── OnnxEmbedding ─────────────────────────────────────────

const OrtApi* OnnxEmbedding::g_api_ = nullptr;

OnnxEmbedding::OnnxEmbedding(const std::string& model_path,
                              const std::string& vocab_path, int dims)
    : dims_(dims) {
  // 加载 Tokenizer
  if (!tokenizer_.load(vocab_path)) {
    LOG_ERROR("onnx: tokenizer load failed: {}", vocab_path);
    return;
  }
  LOG_INFO("onnx: tokenizer loaded, vocab={}", tokenizer_.size());

  // 初始化 ORT API（进程级单例模式）
  if (!g_api_) {
    g_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_api_) {
      LOG_ERROR("onnx: OrtGetApiBase failed");
      return;
    }
  }

  // 创建环境
  auto status = g_api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "embed", &env_);
  if (status) {
    LOG_ERROR("onnx: CreateEnv: {}", g_api_->GetErrorMessage(status));
    g_api_->ReleaseStatus(status);
    return;
  }

  // Session options（CPU + 内存优化）
  OrtSessionOptions* opts = nullptr;
  g_api_->CreateSessionOptions(&opts);
  g_api_->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_EXTENDED);
  g_api_->SetIntraOpNumThreads(opts, 1);
  g_api_->SetInterOpNumThreads(opts, 1);
  g_api_->EnableCpuMemArena(opts);

  status = g_api_->CreateSession(env_, model_path.c_str(), opts, &session_);
  g_api_->ReleaseSessionOptions(opts);
  if (status) {
    LOG_ERROR("onnx: CreateSession: {}", g_api_->GetErrorMessage(status));
    g_api_->ReleaseStatus(status);
    session_ = nullptr;
    return;
  }

  status = g_api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info_);
  if (status) {
    LOG_ERROR("onnx: CreateCpuMemoryInfo: {}", g_api_->GetErrorMessage(status));
    g_api_->ReleaseStatus(status);
    return;
  }
  LOG_INFO("onnx: model loaded dims={}", dims_);
}

OnnxEmbedding::~OnnxEmbedding() {
  if (session_) g_api_->ReleaseSession(session_);
  if (env_) g_api_->ReleaseEnv(env_);
  if (mem_info_) g_api_->ReleaseMemoryInfo(mem_info_);
}

std::vector<float> OnnxEmbedding::encode(std::string_view text) {
  if (!session_ || !g_api_) return {};

  auto input_ids = tokenizer_.encode(text, 512);
  int64_t seq_len = static_cast<int64_t>(input_ids.size());
  if (seq_len < 2) return {};

  // 注意力掩码 + token 类型 ID
  std::vector<int64_t> mask(seq_len, 1);
  std::vector<int64_t> seg(seq_len, 0);

  // 创建输入 tensors
  const int64_t shape[] = {1, seq_len};
  size_t byte_size = seq_len * sizeof(int64_t);

  OrtValue* inputs[3] = {};
  g_api_->CreateTensorWithDataAsOrtValue(
      mem_info_, input_ids.data(), byte_size, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[0]);
  g_api_->CreateTensorWithDataAsOrtValue(
      mem_info_, mask.data(), byte_size, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]);
  g_api_->CreateTensorWithDataAsOrtValue(
      mem_info_, seg.data(), byte_size, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[2]);

  // 推理
  const char* in_names[] = {"input_ids", "attention_mask", "token_type_ids"};
  const char* out_names[] = {"last_hidden_state"};
  OrtValue* output = nullptr;
  auto status = g_api_->Run(session_, nullptr, in_names, inputs, 3,
                            out_names, 1, &output);

  for (int i = 0; i < 3; ++i) g_api_->ReleaseValue(inputs[i]);

  if (status) {
    LOG_WARN("onnx: Run error: {}", g_api_->GetErrorMessage(status));
    g_api_->ReleaseStatus(status);
    return {};
  }

  // 获取输出数据
  OrtTensorTypeAndShapeInfo* info = nullptr;
  status = g_api_->GetTensorTypeAndShape(output, &info);
  if (status) {
    LOG_WARN("onnx: GetTensorTypeAndShape: {}", g_api_->GetErrorMessage(status));
    g_api_->ReleaseStatus(status);
    g_api_->ReleaseValue(output);
    return {};
  }

  size_t elem_count = 0;
  g_api_->GetTensorShapeElementCount(info, &elem_count);
  int out_dim = static_cast<int>(elem_count / seq_len);
  g_api_->ReleaseTensorTypeAndShapeInfo(info);

  if (out_dim == 0) {
    g_api_->ReleaseValue(output);
    return {};
  }

  float* out_data = nullptr;
  status = g_api_->GetTensorMutableData(output, reinterpret_cast<void**>(&out_data));
  if (status || !out_data) {
    LOG_WARN("onnx: GetTensorMutableData failed");
    if (status) {
      LOG_WARN("onnx: GetTensorMutableData: {}", g_api_->GetErrorMessage(status));
      g_api_->ReleaseStatus(status);
    }
    g_api_->ReleaseValue(output);
    return {};
  }

  // 沿序列长度做均值池化
  std::vector<float> result(dims_, 0.0f);
  for (int64_t t = 0; t < seq_len; ++t) {
    for (int d = 0; d < dims_ && d < out_dim; ++d) {
      result[d] += out_data[t * out_dim + d];
    }
  }
  for (int d = 0; d < dims_; ++d) result[d] /= static_cast<float>(seq_len);

  // L2 normalize（bge 需要归一化向量用于余弦相似度）
  float norm = 0.0f;
  for (float v : result) norm += v * v;
  norm = std::sqrt(norm) + 1e-12f;
  for (float& v : result) v /= norm;

  g_api_->ReleaseValue(output);
  return result;
}

}  // namespace ai_gateway
