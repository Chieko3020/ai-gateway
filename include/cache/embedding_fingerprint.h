// Embedding 指纹：回答"这份落盘向量是哪个模型、哪个分词器、多少维产生的"
//
// 为什么必须有它：落盘缓存只存向量，换了模型或改了分词（本项目刚改过 WordPiece
// 规则）之后，旧向量与新查询向量处在**不同的向量空间**里，余弦相似度失去意义。
// 更糟的是它不会报错——只会静默给出错误命中（返回另一个问题的答案），
// 因此必须把"产生向量的模型身份"一起落盘并在加载时比对。
#pragma once

#include <cstdint>
#include <string>

namespace ai_gateway {

// 指纹的组成（全部是"会改变向量语义"的东西）：
//   model=<模型文件内容哈希(16 hex)>   模型权重变了，向量空间就变了
//   vocab=<词表文件内容哈希(16 hex)>   分词规则/词表变了，同一个字符串也会得到不同向量
//   dim=<向量维度>                     维度不同根本不能互相比对
//   tok=<分词器标识>                   实现版本标识（如 bert-wordpiece@v2），
//                                      用于"词表和模型都没换、但分词实现改了"的情况
struct EmbeddingFingerprint {
  std::string model_digest;  // 16 hex，空 = 文件读取失败
  std::string vocab_digest;  // 16 hex，空 = 文件读取失败
  int dim = 0;
  std::string tokenizer_id;

  // 形如 "model=abc...;vocab=def...;dim=512;tok=bert-wordpiece@v2"
  std::string to_string() const;

  // 是否可用（模型与词表都读到了）。不可用时调用方应回退到"不校验"并告警，
  // 而不是把空指纹当成"匹配"
  bool valid() const { return !model_digest.empty() && !vocab_digest.empty(); }
};

// 文件内容哈希：FNV-1a 64 位遍历**全部字节**（不使用采样式哈希——只读头部
// 会让"模型换了但头部相同"的文件算出同一个指纹），输出 16 位十六进制。
// 读失败返回空串（调用方据此判定指纹不可用）
std::string hash_file_fnv1a64(const std::string& path);

// 组装指纹。tokenizer_id 由调用方给出（如 "bert-wordpiece@v2"）
EmbeddingFingerprint make_embedding_fingerprint(const std::string& model_path,
                                                const std::string& vocab_path,
                                                int dim,
                                                const std::string& tokenizer_id);

}  // namespace ai_gateway
