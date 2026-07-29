// 极简日志宏：零依赖，stderr 输出，NDEBUG 切除 DEBUG
//
// 设计决策：不使用 std::vlog + make_format_args（右值引用问题在宏中难以处理）
// 改为 try-catch 包裹 std::format，异常时回退到 raw 输出
#pragma once

#include <chrono>
#include <ctime>
#include <format>
#include <iostream>
#include <string>

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
  std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
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

// 核心输出函数：format 可能抛异常（参数类型不匹配等），兜底输出 raw
template <typename... Args>
void emit(LogLevel lv, std::string_view fmt_str, Args&&... args) {
  if (lv < kActiveLevel) return;
  try {
    std::cerr << std::format("[{} {}] {}\n", timestamp(), level_tag(lv),
                             std::vformat(fmt_str, std::make_format_args(args...)));
  } catch (const std::exception& e) {
    std::cerr << std::format("[{} {}] (fmt error: {})\n",
                             timestamp(), level_tag(lv), e.what());
  }
}

}  // namespace detail
}  // namespace ai_gateway

// 用 do-while(0) 确保宏在任何控制流中行为正确
// __VA_OPT__(,) 在无额外参数时省略逗号，避免 "emit(fmt,)" 语法错误
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
