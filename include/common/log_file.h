// 日志文件：static 单例模式 ofstream
#pragma once
#include <fstream>
#include <mutex>
#include <string>

namespace ai_gateway::detail {

std::ofstream& log_file();
std::mutex& log_mutex();
void set_log_file(const std::string& path);

}  // namespace ai_gateway::detail
