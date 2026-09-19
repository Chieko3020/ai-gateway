// float 向量的 base64 序列化：落盘时替代"JSON 数字数组"
//
// 为什么需要它：向量是 512 个 float（2048 字节），但 nlohmann 会把每个 float
// 按**17 位有效数字**写进 JSON（与 double 完全同形），平均 21 字符/元素 ——
// 一条 512 维条目光向量就 10.5KB。1 万条实测落盘 107.7MB，而且 save 期间要构建
// 等量的 DOM 与 dump 字符串，把内存峰值推到 373MB。
// 换成"原始 float 字节 + base64"后：2048 字节 → 2731 字符，体积降到约 1/4，
// DOM 里也只是一个字符串。
//
// 字节序：直接按本机内存布局写出，因此**不跨字节序通用**（只在小端机器间可读）。
// 这是有意的取舍：换成显式大端需要逐元素转换，而向量落盘本来就是"同机重启"
// 场景；真要跨平台迁移，重建索引比搬向量更省事（向量可由文本重算）。
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ai_gateway {

namespace detail {

inline constexpr char kB64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// '=' 用 -2 标记（区别于 -1 的"非法字符"）
inline const std::array<int8_t, 256>& b64_decode_table() {
  static const std::array<int8_t, 256> table = [] {
    std::array<int8_t, 256> t{};
    t.fill(-1);
    for (int i = 0; i < 64; ++i)
      t[static_cast<unsigned char>(kB64Alphabet[i])] = static_cast<int8_t>(i);
    t[static_cast<unsigned char>('=')] = -2;
    return t;
  }();
  return table;
}

}  // namespace detail

// 把 float 向量编码成 base64（长度为 4 的倍数，不足处补 '='）
inline std::string encode_float_vector(const std::vector<float>& values) {
  if (values.empty()) return {};
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  const size_t n = values.size() * sizeof(float);

  std::string out;
  out.reserve((n + 2) / 3 * 4);
  for (size_t i = 0; i < n; i += 3) {
    const uint32_t b0 = bytes[i];
    const uint32_t b1 = (i + 1 < n) ? bytes[i + 1] : 0u;
    const uint32_t b2 = (i + 2 < n) ? bytes[i + 2] : 0u;
    const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
    out += detail::kB64Alphabet[(triple >> 18) & 0x3F];
    out += detail::kB64Alphabet[(triple >> 12) & 0x3F];
    out += (i + 1 < n) ? detail::kB64Alphabet[(triple >> 6) & 0x3F] : '=';
    out += (i + 2 < n) ? detail::kB64Alphabet[triple & 0x3F] : '=';
  }
  return out;
}

// 解码；任何非法字符、长度不是 4 的倍数、或解出的元素数不等于 expected_count
// 都返回空 vector（调用方据此把这条目当"没有向量"处理，而不是拿半截向量去检索）
inline std::vector<float> decode_float_vector(const std::string& b64,
                                              size_t expected_count) {
  if (b64.empty()) return {};
  if (b64.size() % 4 != 0) return {};

  const auto& table = detail::b64_decode_table();
  std::vector<unsigned char> bytes;
  bytes.reserve(b64.size() / 4 * 3);
  for (size_t i = 0; i < b64.size(); i += 4) {
    int8_t v[4];
    int pad = 0;
    for (int j = 0; j < 4; ++j) {
      v[j] = table[static_cast<unsigned char>(b64[i + j])];
      if (v[j] == -1) return {};  // 非法字符
      if (v[j] == -2) ++pad;      // '='
    }
    if (pad > 2) return {};
    const uint32_t triple = (static_cast<uint32_t>(v[0] & 0x3F) << 18) |
                            (static_cast<uint32_t>(v[1] & 0x3F) << 12) |
                            (static_cast<uint32_t>(v[2] & 0x3F) << 6) |
                            static_cast<uint32_t>(v[3] & 0x3F);
    bytes.push_back(static_cast<unsigned char>((triple >> 16) & 0xFF));
    if (pad < 2) bytes.push_back(static_cast<unsigned char>((triple >> 8) & 0xFF));
    if (pad < 1) bytes.push_back(static_cast<unsigned char>(triple & 0xFF));
  }

  if (bytes.size() != expected_count * sizeof(float)) return {};
  std::vector<float> out(expected_count);
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

}  // namespace ai_gateway
