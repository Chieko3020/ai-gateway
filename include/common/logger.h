// stderr 输出 + 写入文件日志，NDEBUG 切除 DEBUG
// 使用 std::vformat + try-catch 包裹格式化，异常时回退 raw 输出
#pragma once

#include <chrono>
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

// 核心输出函数：同时写 stderr 和文件
template <typename... Args>
void emit(LogLevel lv, std::string_view fmt_str, Args&&... args) {
  if (lv < kActiveLevel) return;
  std::lock_guard lock(log_mutex());
  try {
    auto line = std::format("[{} {}] {}\n", timestamp(), level_tag(lv),
                            std::vformat(fmt_str, std::make_format_args(args...)));
    std::cerr << line;
    auto& f = detail::log_file();
    if (f.is_open()) { f << line << std::flush; }
  } catch (const std::exception& e) {
    auto fallback = std::format("[{} {}] (fmt error: {})\n", timestamp(), level_tag(lv), e.what());
    std::cerr << fallback;
    auto& f = detail::log_file();
    if (f.is_open()) { f << fallback << std::flush; }
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

#define LOG_WARN(fmt, ...)                                        \
  do {                                                            \
    if constexpr (::ai_gateway::kActiveLevel <= ::ai_gateway::LogLevel::WARN) \
      ::ai_gateway::detail::emit(::ai_gateway::LogLevel::WARN, fmt __VA_OPT__(,) __VA_ARGS__); \
  } while (0)

#define LOG_ERROR(fmt, ...)                                       \
  do {                                                            \
    ::ai_gateway::detail::emit(::ai_gateway::LogLevel::ERROR, fmt __VA_OPT__(,) __VA_ARGS__); \
  } while (0)
