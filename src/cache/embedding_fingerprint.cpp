// Embedding 指纹实现
#include "cache/embedding_fingerprint.h"

#include <cstdio>
#include <format>

namespace ai_gateway {

std::string hash_file_fnv1a64(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return {};
  uint64_t h = 0xcbf29ce484222325ull;
  // 1MB 缓冲：文件是 23MB 级的 ONNX 模型，逐块读取避免大块分配
  static constexpr size_t kChunk = 1 << 20;
  static thread_local std::string buf;
  buf.resize(kChunk);
  size_t n = 0;
  while ((n = std::fread(buf.data(), 1, kChunk, f)) > 0) {
    for (size_t i = 0; i < n; ++i) {
      h ^= static_cast<unsigned char>(buf[i]);
      h *= 0x100000001b3ull;
    }
  }
  bool failed = std::ferror(f) != 0;
  std::fclose(f);
  if (failed) return {};
  return std::format("{:016x}", h);
}

std::string EmbeddingFingerprint::to_string() const {
  return std::format("model={};vocab={};dim={};tok={}", model_digest,
                     vocab_digest, dim, tokenizer_id);
}

EmbeddingFingerprint make_embedding_fingerprint(const std::string& model_path,
                                                const std::string& vocab_path,
                                                int dim,
                                                const std::string& tokenizer_id) {
  EmbeddingFingerprint fp;
  fp.model_digest = hash_file_fnv1a64(model_path);
  fp.vocab_digest = hash_file_fnv1a64(vocab_path);
  fp.dim = dim;
  fp.tokenizer_id = tokenizer_id;
  return fp;
}

}  // namespace ai_gateway
