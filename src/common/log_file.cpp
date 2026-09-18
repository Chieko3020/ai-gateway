// 日志文件实现
#include "common/log_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>

namespace ai_gateway::detail {

std::ofstream& log_file() {
  static std::ofstream file;
  return file;
}

std::mutex& log_mutex() {
  static std::mutex mtx;
  return mtx;
}

void set_log_file(const std::string& path) {
  std::lock_guard lock(log_mutex());
  if (log_file().is_open()) log_file().close();

  // 日志里可能出现用户消息片段、上游错误体等敏感内容，权限必须与 API key 文件
  // 一致（0600）。std::ofstream 不能指定权限位，因此先用 open(2) 以 0600 创建/
  // 打开，再用 /proc/self/fd 把它交给流（避免 ofstream 以默认 0644 新建）。
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (fd < 0) {
    // 目录不存在时补建一级父目录后重试（旧实现只打印一行错误就静默丢日志）
    auto slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > 0 &&
        ::mkdir(path.substr(0, slash).c_str(), 0700) == 0)
      fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  }
  if (fd < 0) {
    std::cerr << "[logger] cannot open: " << path << " (" << std::strerror(errno)
              << ")\n";
    return;
  }
  // 已存在的文件可能仍是旧权限（0644）：显式收紧
  ::fchmod(fd, 0600);

  char proc[64];
  std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
  log_file().open(proc, std::ios::app);
  ::close(fd);  // ofstream 已持有自己的 fd
  if (!log_file().is_open())
    std::cerr << "[logger] cannot attach stream: " << path << "\n";
  // 不做任何 pubsetbuf 干预：libstdc++ 里在 open() 之后调用它无效，
  // 反而会让人误以为改了缓冲策略。缓冲行为是流默认的，INFO 由 WARN+ 的
  // 显式 flush 与进程正常退出时的析构共同保证落盘
}

}  // namespace ai_gateway::detail
