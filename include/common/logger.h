// stderr 输出 + 写入文件日志，NDEBUG 切除 DEBUG
// 使用 std::vformat + try-catch 包裹格式化，异常时回退 raw 输出
//
// 热路径日志策略（报告 L7）：
//   - INFO 不再每行 std::flush：flush 是一次 write 系统调用，4 个 worker 每请求 2 条
//     INFO 时日志写入会成为全局串行热点（且更早的实现还在全局 log_mutex 内）
//   - WARN/ERROR 仍然 flush：故障信息不能因为进程异常退出而留在缓冲区
//   - 每请求的 INFO 可用 log_sample_every 降级/采样（见 sampled_log）
#pragma once
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

#include "common/log_file.h"

namespace ai_gateway {

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

#ifndef LOG_ACTIVE_LEVEL
#ifdef NDEBUG
inline constexpr LogLevel kActiveLevel = LogLevel::INFO;
#else
inline constexpr LogLevel kActiveLevel = LogLevel::DEBUG;
#endif
#else
inline constexpr LogLevel kActiveLevel = LOG_ACTIVE_LEVEL;
#endif

namespace detail {

inline std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto t = std::chrono::system_clock::to_time_t(now);
  char buf[20]{};
  struct tm tm_buf{};
  std::strftime(buf, sizeof(buf), "%H:%M:%S", localtime_r(&t, &tm_buf));
  return buf;
}

inline const char* level_tag(LogLevel lv) {
  switch (lv) {
    case LogLevel::DEBUG: return "DEBUG";
    case LogLevel::INFO:  return "INFO";
    case LogLevel::WARN:  return "WARN";
    case LogLevel::ERROR: return "ERROR";
  }
  return "????";
}

// 每请求 INFO 的采样率：1 = 全量（默认），N > 1 = 每 N 条只留 1 条。
// 通过 set_log_sample_every() 从配置覆盖
inline std::atomic<uint64_t>& log_sample_every() {
  static std::atomic<uint64_t> n{1};
  return n;
}

inline std::atomic<uint64_t>& log_line_counter() {
  static std::atomic<uint64_t> c{0};
  return c;
}

inline void set_log_sample_every(uint64_t every) {
  log_sample_every().store(every == 0 ? 1 : every, std::memory_order_relaxed);
}

// 热路径 INFO 的采样判定：返回 true 表示本条应当输出。
// 计数用原子自增（无锁），采样率 > 1 时只有第 1、N+1、2N+1… 条落盘
inline bool sampled_log() {
  const uint64_t every = log_sample_every().load(std::memory_order_relaxed);
  if (every <= 1) return true;
  const uint64_t n = log_line_counter().fetch_add(1, std::memory_order_relaxed);
  return n % every == 0;
}

// 核心输出函数：同时写 stderr 和文件
template <typename... Args>
void emit(LogLevel lv, std::string_view fmt_str, Args&&... args) {
  if (lv < kActiveLevel) return;
  std::lock_guard lock(log_mutex());
  // WARN 及以上必须落盘（flush），INFO/DEBUG 交给流缓冲批量写出
  const bool flush_now = lv >= LogLevel::WARN;
  try {
    auto line = std::format("[{} {}] {}\n", timestamp(), level_tag(lv),
                            std::vformat(fmt_str, std::make_format_args(args...)));
    std::cerr << line;
    auto& f = detail::log_file();
    if (f.is_open()) {
      f << line;
      if (flush_now) f.flush();
    }
  } catch (const std::exception& e) {
    auto fallback = std::format("[{} {}] (fmt error: {})\n", timestamp(), level_tag(lv), e.what());
    std::cerr << fallback;
    auto& f = detail::log_file();
    if (f.is_open()) { f << fallback; f.flush(); }
  }
}

}  // namespace detail
}  // namespace ai_gateway

// 用 do-while(0) 确保宏在任何控制流中行为正确
// if constexpr 在编译期丢弃 false 分支，包括所有函数参数求值
// LOG_DEBUG(expensive()) 在 Release 下零开销
#define LOG_DEBUG(fmt, ...)                                       \
  do {                                                            \
    if constexpr (::ai_gateway::kActiveLevel <= ::ai_gateway::LogLevel::DEBUG) \
      ::ai_gateway::detail::emit(::ai_gateway::LogLevel::DEBUG, fmt __VA_OPT__(,) __VA_ARGS__); \
  } while (0)

#define LOG_INFO(fmt, ...)                                        \
  do {                                                            \
    if constexpr (::ai_gateway::kActiveLevel <= ::ai_gateway::LogLevel::INFO) \
      ::ai_gateway::detail::emit(::ai_gateway::LogLevel::INFO, fmt __VA_OPT__(,) __VA_ARGS__); \
  } while (0)

// 热路径 INFO：受 log_sample_every 采样控制（默认 1 = 全量输出）
#define LOG_INFO_SAMPLED(fmt, ...)                                \
  do {                                                            \
    if constexpr (::ai_gateway::kActiveLevel <= ::ai_gateway::LogLevel::INFO) \
      if (::ai_gateway::detail::sampled_log())                     \
        ::ai_gateway::detail::emit(::ai_gateway::LogLevel::INFO, fmt __VA_OPT__(,) __VA_ARGS__); \
  } while (0)

#define LOG_WARN(fmt, ...)                                        \
  do {                                                            \
    if constexpr (::ai_gateway::kActiveLevel <= ::ai_gateway::LogLevel::WARN) \
      ::ai_gateway::detail::emit(::ai_gateway::LogLevel::WARN, fmt __VA_OPT__(,) __VA_ARGS__); \
  } while (0)

#define LOG_ERROR(fmt, ...)                                       \
  do {                                                            \
    ::ai_gateway::detail::emit(::ai_gateway::LogLevel::ERROR, fmt __VA_OPT__(,) __VA_ARGS__); \
  } while (0)
