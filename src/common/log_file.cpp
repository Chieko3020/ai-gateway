// 日志文件实现
#include "common/log_file.h"
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
  log_file().open(path, std::ios::app);
  if (!log_file().is_open())
    std::cerr << "[logger] cannot open: " << path << "\n";
}

}  // namespace ai_gateway::detail
