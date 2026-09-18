// ONNX Runtime 嵌入推理 & BERT WordPiece 分词器实现
#include "cache/onnx_embedding.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "common/logger.h"

namespace ai_gateway {

// ── BertWordPieceTokenizer ───────────────────────────────────
//
// 规则来源：transformers.BertTokenizer（BasicTokenizer + WordPieceTokenizer）。
// 对照脚本：scripts/tokenizer_reference.py（生成金标准 id 表）、
//           scripts/tokenizer_parity.py（批量与 HuggingFace 逐条比对）。

namespace {

// 把 UTF-8 字节序列解成码点。非法字节按单字节码点处理（不抛异常，
// 分词器宁可产出 [UNK] 也不能让请求路径崩掉）
struct Utf8Decoder {
  std::string_view s;
  size_t pos = 0;

  bool next(uint32_t& cp) {
    if (pos >= s.size()) return false;
    const unsigned char c0 = static_cast<unsigned char>(s[pos]);
    size_t len = 1;
    uint32_t v = c0;
    if (c0 >= 0xF0) { len = 4; v = c0 & 0x07u; }
    else if (c0 >= 0xE0) { len = 3; v = c0 & 0x0Fu; }
    else if (c0 >= 0xC0) { len = 2; v = c0 & 0x1Fu; }
    if (pos + len > s.size()) { len = 1; v = c0; }
    for (size_t k = 1; k < len; ++k) {
      const unsigned char cc = static_cast<unsigned char>(s[pos + k]);
      if ((cc & 0xC0u) != 0x80u) { len = 1; v = c0; break; }
      v = (v << 6) | (cc & 0x3Fu);
    }
    pos += len;
    cp = v;
    return true;
  }
};

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// BERT 的 _is_whitespace：'\t' '\n' '\r' ' ' 以及 Unicode 类别 Zs。
// 这里列出 Zs 的码位。注意 U+2028/U+2029 是 Zl/Zp 而不是 Zs——官方把它们当
// 普通字符保留，所以这里也不能当成空白
bool is_whitespace(uint32_t cp) {
  switch (cp) {
    case 0x09: case 0x0A: case 0x0D: case 0x20:
    case 0xA0: case 0x1680:
    case 0x202F: case 0x205F: case 0x3000:
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;
  }
}

// BERT 的 _is_control：类别 Cc / Cf（whitespace 已在前面拦掉）。
// 注意官方对 '\t' '\n' '\r' 显式返回 false（它们是空白而不是控制符，会被换成空格；
// 若在这里当成控制符丢掉，"a\tb" 会变成 "ab" 这种错误的词边界）。
// Cf 只覆盖常见码位（文档化近似）
bool is_control(uint32_t cp) {
  if (cp == 0x09 || cp == 0x0A || cp == 0x0D) return false;
  if (cp < 0x20 || cp == 0x7F) return true;           // C0 控制符
  if (cp >= 0x80 && cp <= 0x9F) return true;           // C1 控制符
  if (cp == 0xAD) return true;
  if (cp >= 0x200B && cp <= 0x200F) return true;       // ZWSP/方向标记
  if (cp >= 0x202A && cp <= 0x202E) return true;
  if (cp >= 0x2060 && cp <= 0x2064) return true;
  if (cp >= 0x2066 && cp <= 0x206F) return true;
  if (cp == 0xFEFF) return true;
  return false;
}

// BERT 的 _is_chinese_char：CJK 统一表意文字各区 + 兼容区
bool is_chinese_char(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// BERT 的 _is_punctuation：ASCII 段（33-47, 58-64, 91-96, 123-126）**或**
// Unicode 类别 P*。注意两段都要：'$'(0x24) 属于 Sc 不在 P* 里，但官方 ASCII 段
// 覆盖它；反过来 "‑"(U+2011, Pd) 只靠 P* 才能识别。
// 下表由 `python3 scripts/tokenizer_reference.py --emit-punctuation` 生成
// （与官方 _is_punctuation 用的是同一份 unicodedata 数据）
struct CodeRange {
  uint32_t lo;
  uint32_t hi;
};

// unicodedata 15.0.0：P* 类别共 842 个码点，合并为 191 个区间
constexpr CodeRange kPunctRanges[] = {
    {0x21, 0x23}, {0x25, 0x2A}, {0x2C, 0x2F}, {0x3A, 0x3B}, {0x3F, 0x40}, {0x5B, 0x5D},
    {0x5F, 0x5F}, {0x7B, 0x7B}, {0x7D, 0x7D}, {0xA1, 0xA1}, {0xA7, 0xA7}, {0xAB, 0xAB},
    {0xB6, 0xB7}, {0xBB, 0xBB}, {0xBF, 0xBF}, {0x37E, 0x37E}, {0x387, 0x387}, {0x55A, 0x55F},
    {0x589, 0x58A}, {0x5BE, 0x5BE}, {0x5C0, 0x5C0}, {0x5C3, 0x5C3}, {0x5C6, 0x5C6},
    {0x5F3, 0x5F4}, {0x609, 0x60A}, {0x60C, 0x60D}, {0x61B, 0x61B}, {0x61D, 0x61F},
    {0x66A, 0x66D}, {0x6D4, 0x6D4}, {0x700, 0x70D}, {0x7F7, 0x7F9}, {0x830, 0x83E},
    {0x85E, 0x85E}, {0x964, 0x965}, {0x970, 0x970}, {0x9FD, 0x9FD}, {0xA76, 0xA76},
    {0xAF0, 0xAF0}, {0xC77, 0xC77}, {0xC84, 0xC84}, {0xDF4, 0xDF4}, {0xE4F, 0xE4F},
    {0xE5A, 0xE5B}, {0xF04, 0xF12}, {0xF14, 0xF14}, {0xF3A, 0xF3D}, {0xF85, 0xF85},
    {0xFD0, 0xFD4}, {0xFD9, 0xFDA}, {0x104A, 0x104F}, {0x10FB, 0x10FB}, {0x1360, 0x1368},
    {0x1400, 0x1400}, {0x166E, 0x166E}, {0x169B, 0x169C}, {0x16EB, 0x16ED}, {0x1735, 0x1736},
    {0x17D4, 0x17D6}, {0x17D8, 0x17DA}, {0x1800, 0x180A}, {0x1944, 0x1945}, {0x1A1E, 0x1A1F},
    {0x1AA0, 0x1AA6}, {0x1AA8, 0x1AAD}, {0x1B5A, 0x1B60}, {0x1B7D, 0x1B7E}, {0x1BFC, 0x1BFF},
    {0x1C3B, 0x1C3F}, {0x1C7E, 0x1C7F}, {0x1CC0, 0x1CC7}, {0x1CD3, 0x1CD3}, {0x2010, 0x2027},
    {0x2030, 0x2043}, {0x2045, 0x2051}, {0x2053, 0x205E}, {0x207D, 0x207E}, {0x208D, 0x208E},
    {0x2308, 0x230B}, {0x2329, 0x232A}, {0x2768, 0x2775}, {0x27C5, 0x27C6}, {0x27E6, 0x27EF},
    {0x2983, 0x2998}, {0x29D8, 0x29DB}, {0x29FC, 0x29FD}, {0x2CF9, 0x2CFC}, {0x2CFE, 0x2CFF},
    {0x2D70, 0x2D70}, {0x2E00, 0x2E2E}, {0x2E30, 0x2E4F}, {0x2E52, 0x2E5D}, {0x3001, 0x3003},
    {0x3008, 0x3011}, {0x3014, 0x301F}, {0x3030, 0x3030}, {0x303D, 0x303D}, {0x30A0, 0x30A0},
    {0x30FB, 0x30FB}, {0xA4FE, 0xA4FF}, {0xA60D, 0xA60F}, {0xA673, 0xA673}, {0xA67E, 0xA67E},
    {0xA6F2, 0xA6F7}, {0xA874, 0xA877}, {0xA8CE, 0xA8CF}, {0xA8F8, 0xA8FA}, {0xA8FC, 0xA8FC},
    {0xA92E, 0xA92F}, {0xA95F, 0xA95F}, {0xA9C1, 0xA9CD}, {0xA9DE, 0xA9DF}, {0xAA5C, 0xAA5F},
    {0xAADE, 0xAADF}, {0xAAF0, 0xAAF1}, {0xABEB, 0xABEB}, {0xFD3E, 0xFD3F}, {0xFE10, 0xFE19},
    {0xFE30, 0xFE52}, {0xFE54, 0xFE61}, {0xFE63, 0xFE63}, {0xFE68, 0xFE68}, {0xFE6A, 0xFE6B},
    {0xFF01, 0xFF03}, {0xFF05, 0xFF0A}, {0xFF0C, 0xFF0F}, {0xFF1A, 0xFF1B}, {0xFF1F, 0xFF20},
    {0xFF3B, 0xFF3D}, {0xFF3F, 0xFF3F}, {0xFF5B, 0xFF5B}, {0xFF5D, 0xFF5D}, {0xFF5F, 0xFF65},
    {0x10100, 0x10102}, {0x1039F, 0x1039F}, {0x103D0, 0x103D0}, {0x1056F, 0x1056F},
    {0x10857, 0x10857}, {0x1091F, 0x1091F}, {0x1093F, 0x1093F}, {0x10A50, 0x10A58},
    {0x10A7F, 0x10A7F}, {0x10AF0, 0x10AF6}, {0x10B39, 0x10B3F}, {0x10B99, 0x10B9C},
    {0x10EAD, 0x10EAD}, {0x10F55, 0x10F59}, {0x10F86, 0x10F89}, {0x11047, 0x1104D},
    {0x110BB, 0x110BC}, {0x110BE, 0x110C1}, {0x11140, 0x11143}, {0x11174, 0x11175},
    {0x111C5, 0x111C8}, {0x111CD, 0x111CD}, {0x111DB, 0x111DB}, {0x111DD, 0x111DF},
    {0x11238, 0x1123D}, {0x112A9, 0x112A9}, {0x1144B, 0x1144F}, {0x1145A, 0x1145B},
    {0x1145D, 0x1145D}, {0x114C6, 0x114C6}, {0x115C1, 0x115D7}, {0x11641, 0x11643},
    {0x11660, 0x1166C}, {0x116B9, 0x116B9}, {0x1173C, 0x1173E}, {0x1183B, 0x1183B},
    {0x11944, 0x11946}, {0x119E2, 0x119E2}, {0x11A3F, 0x11A46}, {0x11A9A, 0x11A9C},
    {0x11A9E, 0x11AA2}, {0x11B00, 0x11B09}, {0x11C41, 0x11C45}, {0x11C70, 0x11C71},
    {0x11EF7, 0x11EF8}, {0x11F43, 0x11F4F}, {0x11FFF, 0x11FFF}, {0x12470, 0x12474},
    {0x12FF1, 0x12FF2}, {0x16A6E, 0x16A6F}, {0x16AF5, 0x16AF5}, {0x16B37, 0x16B3B},
    {0x16B44, 0x16B44}, {0x16E97, 0x16E9A}, {0x16FE2, 0x16FE2}, {0x1BC9F, 0x1BC9F},
    {0x1DA87, 0x1DA8B}, {0x1E95E, 0x1E95F},
};

bool is_unicode_punctuation(uint32_t cp) {
  // 区间按 lo 升序，二分查找
  size_t lo = 0, hi = sizeof(kPunctRanges) / sizeof(kPunctRanges[0]);
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (cp < kPunctRanges[mid].lo)
      hi = mid;
    else if (cp > kPunctRanges[mid].hi)
      lo = mid + 1;
    else
      return true;
  }
  return false;
}

bool is_punctuation(uint32_t cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
    return true;
  return cp > 127 && is_unicode_punctuation(cp);
}

// ASCII 小写化（官方 lower() 是 Unicode 的，这里只处理 ASCII；
// 非 ASCII 大小写字符本词表里基本不存在）
uint32_t to_ascii_lower(uint32_t cp) {
  return (cp >= 'A' && cp <= 'Z') ? cp - 'A' + 'a' : cp;
}

}  // namespace

bool BertWordPieceTokenizer::load(const std::string& vocab_path) {
  std::ifstream f(vocab_path);
  if (!f) return false;
  vocab_.clear();
  std::string line;
  int id = 0;
  while (std::getline(f, line)) {
    // 与官方 load_vocab 一致：**每一行**都占一个 id（空行也不例外），
    // 否则后面所有 token 的 id 都会整体前移（官方只 rstrip("\n")）
    vocab_[line] = id++;
  }
  return vocab_.size() > kMinVocabSize;
}

int BertWordPieceTokenizer::id_of(std::string_view piece,
                                  std::string& scratch) const {
  // unordered_map<std::string,int> 不支持异质查找，用调用方复用的缓冲避免
  // 每次 probe 都分配（assign 复用已有容量）
  scratch.assign(piece.data(), piece.size());
  auto it = vocab_.find(scratch);
  return it == vocab_.end() ? -1 : it->second;
}

std::vector<std::string> BertWordPieceTokenizer::basic_tokenize(
    std::string_view text) const {
  // 1. clean_text：丢控制符、空白统一成单个空格
  std::string cleaned;
  cleaned.reserve(text.size() + 8);
  {
    Utf8Decoder d{text};
    uint32_t cp = 0;
    while (d.next(cp)) {
      if (cp == 0 || cp == 0xFFFD || is_control(cp)) continue;
      if (is_whitespace(cp)) {
        cleaned.push_back(' ');
        continue;
      }
      append_utf8(cleaned, cp);
    }
  }

  // 2. tokenize_chinese_chars：每个 CJK 字符两侧补空格
  std::string padded;
  padded.reserve(cleaned.size() + 8);
  {
    Utf8Decoder d{cleaned};
    uint32_t cp = 0;
    while (d.next(cp)) {
      if (is_chinese_char(cp)) padded.push_back(' ');
      append_utf8(padded, cp);
      if (is_chinese_char(cp)) padded.push_back(' ');
    }
  }

  // 3. 可选小写化
  if (do_lower_case_) {
    std::string lowered;
    lowered.reserve(padded.size());
    Utf8Decoder d{padded};
    uint32_t cp = 0;
    while (d.next(cp)) append_utf8(lowered, to_ascii_lower(cp));
    padded.swap(lowered);
  }

  // 4. 按空白切分，再按 ASCII 标点切分（官方 _run_split_on_punc 的语义：
  //    标点各自成词，且逐字符判断，因此连续标点会切成多个单字符 token）
  std::vector<std::string> out;
  std::string cur;
  auto flush = [&] {
    if (!cur.empty()) {
      out.push_back(cur);
      cur.clear();
    }
  };
  {
    Utf8Decoder d{padded};
    uint32_t cp = 0;
    while (d.next(cp)) {
      if (is_whitespace(cp)) {
        flush();
        continue;
      }
      if (is_punctuation(cp)) {
        flush();
        std::string p;
        append_utf8(p, cp);
        out.push_back(std::move(p));
        continue;
      }
      append_utf8(cur, cp);
    }
  }
  flush();
  return out;
}

std::vector<std::string> BertWordPieceTokenizer::wordpiece_tokenize(
    std::string_view word) const {
  std::vector<std::string> out;
  if (word.empty()) return out;

  // 官方：超过 max_input_chars_per_word 直接整词 [UNK]。
  // 这里按码点计数（官方 len(token) 是 Python 字符数）
  size_t chars = 0;
  {
    Utf8Decoder d{word};
    uint32_t cp = 0;
    while (d.next(cp)) ++chars;
  }
  if (chars > kMaxCharsPerWord) {
    out.emplace_back("[UNK]");
    return out;
  }

  std::string scratch;
  size_t start = 0;
  while (start < word.size()) {
    size_t end = word.size();
    int found = -1;
    std::string found_piece;
    while (start < end) {
      std::string_view sub = word.substr(start, end - start);
      std::string probe;
      if (start > 0) {
        probe.reserve(sub.size() + 2);
        probe += "##";
        probe.append(sub.data(), sub.size());
        if (id_of(probe, scratch) >= 0) {
          found = 1;
          found_piece = probe;
          break;
        }
      } else if (id_of(sub, scratch) >= 0) {
        found = 1;
        found_piece.assign(sub.data(), sub.size());
        break;
      }
      // 官方按字符（不是字节）回退长度：这里按 UTF-8 边界回退，
      // 否则会在多字节字符中间截断，得到一个永远查不到的字节前缀
      --end;
      while (end > start &&
             (static_cast<unsigned char>(word[end]) & 0xC0u) == 0x80u)
        --end;
    }
    if (found < 0) {
      // 只要有一个位置匹配不上，整词回退为单个 [UNK]（官方行为：不逐字符回退）
      out.clear();
      out.emplace_back("[UNK]");
      return out;
    }
    out.push_back(std::move(found_piece));
    start = end;
  }
  return out;
}

std::vector<std::string> BertWordPieceTokenizer::tokenize(
    std::string_view text) const {
  std::vector<std::string> out;
  for (const auto& basic : basic_tokenize(text)) {
    auto pieces = wordpiece_tokenize(basic);
    for (auto& p : pieces) out.push_back(std::move(p));
  }
  return out;
}

std::vector<int64_t> BertWordPieceTokenizer::encode(std::string_view text,
                                                    int max_len) const {
  std::vector<int64_t> ids;
  if (max_len < 2) return ids;  // 放不下 [CLS]/[SEP]
  ids.reserve(static_cast<size_t>(max_len));
  ids.push_back(kClsId);

  std::string scratch;
  for (const auto& piece : tokenize(text)) {
    if (static_cast<int>(ids.size()) >= max_len - 1) break;  // 给 [SEP] 留位
    int id = piece == "[UNK]" ? kUnkId : id_of(piece, scratch);
    if (id < 0) id = kUnkId;
    ids.push_back(id);
  }

  ids.push_back(kSepId);
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
    load_error_ = LoadError::kUnavailable;
    return;
  }
  LOG_INFO("onnx: tokenizer loaded, vocab={}", tokenizer_.size());

  // 初始化 ORT API（进程级单例模式，call_once 防竞态）
  static std::once_flag api_init;
  std::call_once(api_init, [] {
    g_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  });
  if (!g_api_) {
    LOG_ERROR("onnx: OrtGetApiBase failed");
    load_error_ = LoadError::kUnavailable;
    return;
  }

  // 创建环境
  auto status = g_api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "embed", &env_);
  if (status) {
    LOG_ERROR("onnx: CreateEnv: {}", g_api_->GetErrorMessage(status));
    g_api_->ReleaseStatus(status);
    load_error_ = LoadError::kUnavailable;
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
    load_error_ = LoadError::kUnavailable;
    return;
  }

  status = g_api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info_);
  if (status) {
    LOG_ERROR("onnx: CreateCpuMemoryInfo: {}", g_api_->GetErrorMessage(status));
    g_api_->ReleaseStatus(status);
    load_error_ = LoadError::kUnavailable;
    return;
  }
  // 用一次探测推理拿到模型的**真实**输出维度。
  // 关键：探测时把 dims_ 临时放到"不可能被截断"的极大值，否则 encode() 里的
  // min(dims_, out_dim) 会让"配置 256 的模型输出 512"被截成 256，探测结果就变成
  // 配置值本身（自证式校验，什么也验证不了——这正是旧日志"dims=512"的由来）
  // 旧行为：不符时静默截断，日志还照打配置值（报告 M15）
  const int saved_dims = dims_;
  dims_ = 1 << 20;
  auto probe = encode("dimension probe");
  dims_ = saved_dims;
  if (probe.empty()) {
    LOG_ERROR("onnx: probe encode failed, refusing to use {}", model_path);
    g_api_->ReleaseSession(session_);
    session_ = nullptr;
    load_error_ = LoadError::kUnavailable;
    return;
  }
  output_dim_ = static_cast<int>(probe.size());
  LOG_INFO("onnx: model loaded configured_dim={} model_output_dim={} "
           "pooling={} do_lower_case={}",
           dims_, output_dim_,
           pooling_ == PoolingMode::kCls ? "cls" : "mean",
           tokenizer_.do_lower_case() ? "true" : "false");
  if (output_dim_ != dims_) {
    LOG_ERROR("onnx: model output dim {} != configured embedding.dim {}, "
              "refusing to load (the index would silently drop every vector)",
              output_dim_, dims_);
    g_api_->ReleaseSession(session_);
    session_ = nullptr;
    load_error_ = LoadError::kDimensionMismatch;
    return;
  }
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
  OrtStatus* st = nullptr;

  auto release_inputs = [&](int count) {
    for (int i = 0; i < count; ++i)
      if (inputs[i]) g_api_->ReleaseValue(inputs[i]);
  };

  st = g_api_->CreateTensorWithDataAsOrtValue(
      mem_info_, input_ids.data(), byte_size, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[0]);
  if (st) {
    LOG_WARN("onnx: CreateTensor input_ids failed: {}", g_api_->GetErrorMessage(st));
    g_api_->ReleaseStatus(st);
    return {};
  }

  st = g_api_->CreateTensorWithDataAsOrtValue(
      mem_info_, mask.data(), byte_size, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]);
  if (st) {
    LOG_WARN("onnx: CreateTensor attention_mask failed: {}", g_api_->GetErrorMessage(st));
    g_api_->ReleaseStatus(st);
    release_inputs(1);
    return {};
  }

  st = g_api_->CreateTensorWithDataAsOrtValue(
      mem_info_, seg.data(), byte_size, shape, 2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[2]);
  if (st) {
    LOG_WARN("onnx: CreateTensor token_type_ids failed: {}", g_api_->GetErrorMessage(st));
    g_api_->ReleaseStatus(st);
    release_inputs(2);
    return {};
  }

  // 推理
  const char* in_names[] = {"input_ids", "attention_mask", "token_type_ids"};
  const char* out_names[] = {"last_hidden_state"};
  OrtValue* output = nullptr;
  auto status = g_api_->Run(session_, nullptr, in_names, inputs, 3,
                            out_names, 1, &output);

  release_inputs(3);

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

  // 池化：把 [1, seq_len, out_dim] 的 last_hidden_state 压成 [out_dim] 句向量。
  //   官方 bge-small-zh-v1.5 的 1_Pooling/config.json 是 pooling_mode_cls_token=true、
  //   pooling_mode_mean_tokens=false，模型卡写 "select the last hidden state of the
  //   first token ([CLS])"，因此默认走 CLS。
  //   构造期已保证 dims_ == 模型实际输出维度，因此这里的 min() 不会真的截断
  //   （保留以防御性处理异常模型）
  int effective_dim = std::min(dims_, out_dim);
  std::vector<float> result(effective_dim, 0.0f);
  if (pooling_ == PoolingMode::kCls) {
    // last_hidden_state[:, 0, :]：[CLS] 位已经过整个序列的自注意力，
    // 是 BERT 类模型训练时约定的句表示（不做 mask 加权——[CLS] 恒在 mask 内）
    for (int d = 0; d < effective_dim; ++d) result[d] = out_data[d];
  } else {
    // 按 attention_mask 加权的均值。本实现的 input_ids 里 [PAD] 只在 max_len
    // 截断时出现（单条文本、无 padding），mask 因此全为 1 —— 加权与不加权结果相同，
    // 这里仍按加权写，是为了与官方 mean 池化的定义逐字对应，不依赖"恰好没有 pad"
    int64_t mask_sum = 0;
    for (int64_t t = 0; t < seq_len; ++t) mask_sum += mask[static_cast<size_t>(t)];
    if (mask_sum <= 0) mask_sum = 1;
    for (int64_t t = 0; t < seq_len; ++t) {
      if (mask[static_cast<size_t>(t)] == 0) continue;
      for (int d = 0; d < effective_dim; ++d) {
        result[d] += out_data[t * out_dim + d];
      }
    }
    for (int d = 0; d < effective_dim; ++d)
      result[d] /= static_cast<float>(mask_sum);
  }

  // L2 normalize（bge 需要归一化向量用于余弦相似度；两种池化都在归一化之前）
  // 注：归一化对 CLS 与 mean 都必要——余弦相似度在下面的 HNSW 检索里按点积算，
  // 没有这一步"向量长度"会混进相似度
  float norm = 0.0f;
  for (float v : result) norm += v * v;
  norm = std::sqrt(norm) + 1e-12f;
  for (float& v : result) v /= norm;

  g_api_->ReleaseValue(output);
  return result;
}

}  // namespace ai_gateway
