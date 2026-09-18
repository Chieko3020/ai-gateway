// 日志文件：static 单例模式 ofstream + 按大小轮转
#pragma once
#include <cstddef>
#include <fstream>
#include <mutex>
#include <string>

namespace ai_gateway::detail {

// 日志轮转策略：单文件达到 max_bytes 时把当前文件改名为 <path>.1，
// 旧的 .1 → .2 …，最老的（第 keep_files 份）被删除。
struct LogRotation {
  size_t max_bytes = 10 * 1024 * 1024;  // 0 = 关闭轮转（回到"单文件一直追加"）
  int keep_files = 5;                   // 保留的历史份数（不含当前文件）
};

std::ofstream& log_file();
std::mutex& log_mutex();
void set_log_file(const std::string& path);

// 设置轮转策略。必须在 set_log_file() 之后调用（内部会按当前策略重建状态）
void set_log_rotation(const LogRotation& rot);

// 记录本次写入文件的字节数：达到阈值时切分。
// **调用者必须持有 log_mutex()**（emit() 已经在锁内写文件，切分也在锁内完成，
// 否则两个线程会同时 rename 同一个文件）
void account_log_bytes(size_t n);

}  // namespace ai_gateway::detail
