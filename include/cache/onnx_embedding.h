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
// BERT WordPiece 分词器（bge-small-zh-v1.5 / bert-base-chinese 词表）
//
// 对齐官方实现（transformers.BertTokenizer 的 BasicTokenizer + WordPiece），
// 逐条规则见 .cpp 与 scripts/tokenizer_reference.py：
//   1. clean_text：控制字符丢弃、空白统一成空格
//   2. tokenize_chinese_chars：每个 CJK 字符两侧补空格（汉字因此逐个成词）
//   3. 大小写：官方 tokenizer_config.json 对本词表是 do_lower_case=false，
//      因此**默认不做**小写化（开关 set_do_lower_case，打开后与 do_lower_case=true 对齐）
//   4. 按空白与标点切分（ASCII 段 + Unicode 类别 P*，与官方 _is_punctuation 一致）
//   5. WordPiece：从词首开始最长匹配，续接子词带 "##" 前缀；
//      单词长度 > 100 或任一位置无法匹配 → 整词一个 [UNK]（不逐字符回退）
//
// 实测一致性（scripts/tokenizer_parity.py，711 条样本 = 压测数据集 412 条 + 定种子随机 300 条）：
//   do_lower_case=false 100%（711/711）、do_lower_case=true 100%（711/711）；
//   修复前的贪心实现在同一批样本上是 41.35% / 40.65%。
//
// 已知未覆盖（只在打开 lowercase 时出现，默认配置下与官方一致）：
//   - 官方 strip_accents 在 do_lower_case=true 时默认跟随打开（对整串做 NFD 去音标、
//     分解韩文音节等），本项目不做 NFD："café" 官方 lowercase=true 得到 "cafe"(8377)、
//     本项目 [UNK]；"한국어" 官方会分解成字母，本项目整词 [UNK]（tests/test_tokenizer.cpp
//     把这三条偏差固定成了断言）
//   - 小写化只处理 ASCII（'İ'/'Ä' 这类非 ASCII 大写字母不做映射）
//   - 空白只覆盖 Zs 的常见码位；类别 Cn（未分配）/Co（私用区）官方会当控制符丢弃，
//     本项目保留（真实语料里不出现）
class BertWordPieceTokenizer {
 public:
  static constexpr size_t kMinVocabSize = 1000;
  // 官方 WordPiece 的 max_input_chars_per_word：超过就整词 [UNK]
  static constexpr size_t kMaxCharsPerWord = 100;
  static constexpr int64_t kUnkId = 100;  // 与词表前几行固定位置一致（[PAD]=0）
  static constexpr int64_t kClsId = 101;
  static constexpr int64_t kSepId = 102;

  bool load(const std::string& vocab_path);
  size_t size() const { return vocab_.size(); }

  // 是否做小写化。默认 false = 与官方 tokenizer_config.json 一致
  void set_do_lower_case(bool on) { do_lower_case_ = on; }
  bool do_lower_case() const { return do_lower_case_; }

  // BasicTokenizer：clean_text + 汉字补空格 + 大小写 + 空白/ASCII 标点切分
  std::vector<std::string> basic_tokenize(std::string_view text) const;
  // WordPiece：对单个词做最长匹配 + ## 续接；失败返回单个 "[UNK]"
  std::vector<std::string> wordpiece_tokenize(std::string_view word) const;
  std::vector<std::string> tokenize(std::string_view text) const;

  // 返回 [CLS](101) + token_ids + [SEP](102)，总长不超过 max_len
  std::vector<int64_t> encode(std::string_view text, int max_len = 512) const;

 private:
  int id_of(std::string_view piece, std::string& scratch) const;

  std::unordered_map<std::string, int> vocab_;
  bool do_lower_case_ = false;
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
  BertWordPieceTokenizer tokenizer_;
  int dims_ = 512;
  int output_dim_ = 0;  // 模型实际输出维度（探测得到）
  LoadError load_error_ = LoadError::kNone;
};

}  // namespace ai_gateway
