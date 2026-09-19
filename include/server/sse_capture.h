// 原始 SSE 字节的旁路解析：把一条已经流完的 SSE 响应还原成"回答文本"。
//
// 为什么需要它（而不是让流式路径也存非流式 JSON）：
//   流式命中要回放**上游原样的字节**（data: 前缀、多事件结构、usage、[DONE]），
//   因此缓存条目里存的是原始 SSE 字节；但缓存的其余机制（实体提取、精确匹配
//   降级、命中率统计、按条目的 saved-token 估算）都建立在"回答文本"上，所以
//   两者必须一起入库。本文件只做识别，不重排、不修改任何字节。
//
// 与 server/sse_usage.h 的分工：那边解析的是"用量"（token 数），这边解析的是
// "内容"（delta.content 拼接）。两者都在旁路读同一份字节，互不影响透传。
#pragma once

#include <string>
#include <string_view>

namespace ai_gateway {

// 从原始 SSE 字节里拼接 assistant 回答正文（所有 `choices[].delta.content`；
// 也认非流式形态的 `choices[].message.content`）。
// 解析规则：
//   - 按 SSE 事件边界（空行）切分，`data:` 行取有效载荷（多行 data 以 \n 连接）
//   - 跳过 `[DONE]`；单个事件 JSON 解析失败时跳过该事件（不因一个坏事件放弃整段）
//   - 只取 content，不取 reasoning_content（推理过程不是"回答"，把它当缓存文本
//     会让实体提取与命中率口径偏离真实答案）
std::string extract_sse_content(std::string_view sse);

// 流里是否出现过终止事件 `data: [DONE]`。
// 用途：只有拿到终止事件的流才算"完整地流完了"，这种流才有资格写进缓存——
// 否则会把客户端中途断开留下的半截内容固化 30 天。
bool sse_has_done(std::string_view sse);

}  // namespace ai_gateway
