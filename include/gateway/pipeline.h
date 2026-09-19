// 网关请求管道：从"客户端请求体"到"给客户端的一份回复"的全部生产逻辑。
//
// 为什么单独成一个编译单元（而不是留在 main.cpp 里）：
//   本项目的请求管道（输入过滤 → 旁路/流式分支 → 语义缓存 → 请求合并 → 上游
//   转发 → 写缓存 → 统计 → 输出过滤）此前只存在于 main.cpp，而 main.cpp 只被
//   编译进 ai-gateway 可执行文件 —— 于是 tests/ 里**没有任何测试目标能链接到
//   这段代码**，所有缺陷都落在零覆盖的代码层（对抗性审查的报告 T2：对 drain、
//   合并 ns 隔离、keep_alive 决策做单变量变异，4 个测试目标全绿）。
//   把管道抽到这里后，main.cpp（生产入口）与 tests/test_pipeline.cpp（单测）
//   链接的是**同一份对象代码**，不再需要"测试自己注册一个路由"那种假覆盖
//   （见 research/personal/ops-incident-log.md 第 13 条）。
//
// 与 HTTP 层的边界：
//   本层不做 socket I/O。流式路径的字节由 ResponseWriter 直接写出（边收边发），
//   缓冲式路径把完整响应体作为返回值交给连接处理器。
#pragma once

#include <chrono>
#include <string>
#include <utility>

#include "cache/cache_engine.h"
#include "common/config.h"
#include "common/singleflight.h"
#include "server/filter.h"
#include "server/http_server.h"  // ResponseWriter 完整定义
#include "stats/stats.h"

namespace ai_gateway {

// 一次请求的处理结果：pair<响应, 本条连接是否**允许**复用>
//
// 第二个字段是 handler 这一侧的决策（"这个响应本身适不适合留在同一条连接上"），
// 与客户端意愿（Connection 头/版本）是两件事：连接处理器把二者取合取。
// 为 false 的典型情形：
//   - 输入被拒 400 / 输出被拦 502：请求语义已经出错，复用只会让客户端把下一个
//     请求排在一条状态可疑的连接上；
//   - 上游 5xx：上游故障期间客户端多半会重试，让它重新握手能错开到别的实例；
//   - 流式请求走了非流式兜底（stream_fallback）：客户端是按流式发起的，异常
//     路径上少一层连接状态歧义更稳妥。
// 流式真透传路径上响应头已经写出去了，此时以管道返回的值为准（见
// connection_handler 的 writer.committed() 旁路）。
using HandleOutcome = std::pair<HttpReply, bool>;

// 分词器实现标识：**改了分词规则就必须改这个字符串**。它进 embedding 指纹，
// 用来覆盖"模型与词表文件都没变、但分词实现改了"这一情形（本项目 2026-09 刚补过
// Bert WordPiece 的 lowercase/## 续接规则，正属此类）。
// 历史值：v1 = 贪心 10 字符窗口（与官方不一致率 58.7%）；v2 = 对齐官方 WordPiece
inline constexpr const char* kTokenizerId = "bert-wordpiece@v2";

// 网关注解缓存状态用的字段值（供压测脚本与单测精确判定是否命中）
inline constexpr const char* kCacheStatusHit = "hit";
inline constexpr const char* kCacheStatusMiss = "miss";
inline constexpr const char* kCacheStatusMerged = "merged";
inline constexpr const char* kCacheStatusBypass = "bypass";
inline constexpr const char* kCacheStatusStreamFallback = "stream_fallback";

// 处理 stream:true：把上游 SSE 逐块透传给客户端。
//
// writer 在这一条路径上被真正使用（边收边发）；缓冲式路径只用返回值。
HandleOutcome handle_stream_request(const std::string& request_body,
                                    const GatewayConfig& cfg,
                                    MessageFilter* filter, Stats* stats,
                                    ResponseWriter& writer,
                                    std::chrono::steady_clock::time_point t0,
                                    bool client_wants_keep_alive);

// 处理一次 POST /v1/chat/completions：输入过滤 → 旁路 → 流式 → 缓存 → 合并 →
// 上游 → 统计 → 输出过滤。所有分支都必须经过输入过滤（这是安全边界）。
HandleOutcome handle_request(const std::string& request_body,
                             const GatewayConfig& cfg, MessageFilter* filter,
                             CacheEngine* engine, Singleflight* sf, Stats* stats,
                             ResponseWriter& writer,
                             bool client_wants_keep_alive);

// ---- 供 main 使用的请求体解析辅助（与管道共用同一份实现） ----
//
// 从 OpenAI 格式请求体里提取带 ns 前缀的缓存命名空间（system prompt 的
// FNV-1a 64 位哈希）；没有 system message 或解析失败时返回空串（不做隔离）
std::string extract_namespace(const std::string& request_body);

}  // namespace ai_gateway
