// SSE 事件流里的 usage 解析（流式请求的 token 统计）
//
// 背景：流式请求此前 token 计数为 0。OpenAI 兼容的流式响应里，token 用量只在
// **最后一个** SSE 事件里出现（形如 `data: {"usage":{"prompt_tokens":..,
// "completion_tokens":..},"choices":[]}`），且**只有当请求带了
// `stream_options.include_usage: true` 时上游才会给**。网关此前只是把字节原样
// 透传，从不解析事件，于是 Stats 里流式请求的 token/费用恒为 0
// （/metrics 与 journal 报表都少算这块）。
//
// 设计约束（为什么是一个专门的类而不是"把 body 攒起来再 json::parse"）：
//  1. **不能改变透传语义**：解析器只旁路观察一份**有界**副本，透传给客户端的字节
//     完全由原路径负责（一个字节都不能被这里影响）
//  2. **有界内存**：SSE 流可以无限长（几 MB 的思考链），把整条流缓存下来再解析
//     就等于把"真透传"退回"整段缓冲"，还多一份内存峰值。这里只留
//     kMaxLineBytes 的窗口：足够容纳最后一个 usage 事件，超长的单行直接丢弃
//  3. **跨 chunk 的 key/value 分离**：TCP 分块边界与 SSE 事件边界无关，`"usage"` 与
//     它的数字完全可能落在两个 chunk 里。这里把"最近见过的 key"记在成员里
//     （而不是同一块内配对），因此跨块也能正确配对
//  4. **只认 `data:` 行的行首**：不解析裸 JSON 行，避免把上游错误体、注释行里的
//     同名字段误当成用量
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ai_gateway {

// 一次流式请求的 token 用量（上游没给 usage 时全部保持 0）
struct StreamUsage {
  int64_t prompt_tokens = 0;
  int64_t completion_tokens = 0;
  // 是否真的在上游事件里见到过 usage 字段。
  // 与"计数为 0"必须区分：上游可能给了 usage 但两个数都是 0（罕见但合法），
  // 也可能压根没给（未开 include_usage）——日志与 /metrics 要能分清这两种情况
  bool seen = false;

  int64_t total() const { return prompt_tokens + completion_tokens; }
};

class SseUsageParser {
 public:
  // 单行（含 data: 前缀）最多观察的字节数。usage 事件只有几百字节，
  // 16KB 的窗口足够；超过就放弃这一行（绝不无限增长）
  static constexpr size_t kMaxLineBytes = 16384;
  // 已处理部分额外保留的字节数：覆盖"`"usage"` 所在的 data 行正好结束在 chunk
  // 边界、数字落在下一个 chunk"的情形（见 sse_usage.cpp 的 feed 说明）。
  // 取值远大于 `data: {"...,"usage":{...}}` 的头部长度，又足够小到重扫可忽略
  static constexpr size_t kCarryBytes = 256;

  // 喂入一块上游字节（可以是任意切片，不需要与事件边界对齐）
  void feed(const char* data, size_t len);
  void feed(std::string_view chunk) { feed(chunk.data(), chunk.size()); }

  const StreamUsage& usage() const { return usage_; }

 private:
  // 处理一条已收齐的行（不含行尾 '\n'）。同一条行可能被重复喂入（见 kCarryBytes），
  // 因此必须是**幂等**的：只做赋值，不做累加
  void process_line(std::string_view line);

  std::string buf_;  // 待处理窗口（有界：kCarryBytes + 一行）
  StreamUsage usage_;
};

}  // namespace ai_gateway
