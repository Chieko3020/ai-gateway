// 日志文件实现（含按大小轮转）
#include "common/log_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <format>
#include <iostream>

namespace ai_gateway::detail {

namespace {

// 路径与轮转状态：只在持有 log_mutex() 时访问
std::string& log_path() {
  static std::string p;
  return p;
}

LogRotation& log_rotation() {
  static LogRotation r;
  return r;
}

size_t& log_bytes_written() {
  static size_t n = 0;
  return n;
}

// 以 0600 打开并挂到单例流上，返回是否成功；成功时按文件当前大小重置字节计数
bool open_log_stream(const std::string& path) {
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
    return false;
  }
  // 已存在的文件可能仍是旧权限（0644）：显式收紧
  ::fchmod(fd, 0600);

  // 轮转判定要用"文件真实大小"起步：进程重启后是接着旧文件追加的，
  // 从 0 起算会让一个已经接近上限的文件再涨一整个 max_bytes
  struct stat st{};
  log_bytes_written() = (::fstat(fd, &st) == 0) ? static_cast<size_t>(st.st_size) : 0;

  char proc[64];
  std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
  log_file().open(proc, std::ios::app);
  ::close(fd);  // ofstream 已持有自己的 fd
  if (!log_file().is_open()) {
    std::cerr << "[logger] cannot attach stream: " << path << "\n";
    return false;
  }
  // 不做任何 pubsetbuf 干预：libstdc++ 里在 open() 之后调用它无效，
  // 反而会让人误以为改了缓冲策略。缓冲行为是流默认的，INFO 由 WARN+ 的
  // 显式 flush 与进程正常退出时的析构共同保证落盘
  return true;
}

std::string backup_name(const std::string& path, int index) {
  return std::format("{}.{}", path, index);
}

// 切分：当前文件 → .1，旧的依次后移，最老的一份删除。
// 调用者必须持有 log_mutex()
void rotate_locked() {
  const auto& rot = log_rotation();
  const std::string path = log_path();
  if (path.empty()) return;

  // 先把用户态缓冲区落盘，否则 close() 之前已写入但仍在缓冲区的日志行
  // 会随着 rename 之后的"新文件"一起错位（甚至丢内容）
  if (log_file().is_open()) log_file().flush();
  log_file().close();

  // 最老的一份直接删除：rename 不会自动回收
  const std::string oldest = backup_name(path, rot.keep_files);
  if (::unlink(oldest.c_str()) != 0 && errno != ENOENT)
    std::cerr << "[logger] cannot remove " << oldest << ": "
              << std::strerror(errno) << "\n";

  for (int i = rot.keep_files - 1; i >= 1; --i) {
    const std::string from = backup_name(path, i);
    const std::string to = backup_name(path, i + 1);
    if (::rename(from.c_str(), to.c_str()) != 0 && errno != ENOENT)
      std::cerr << "[logger] cannot rotate " << from << " -> " << to << ": "
                << std::strerror(errno) << "\n";
  }
  if (::rename(path.c_str(), backup_name(path, 1).c_str()) != 0) {
    std::cerr << "[logger] cannot rotate " << path << ": "
              << std::strerror(errno) << "\n";
    // 改名失败时不要放弃：重新打开原文件继续写，至少不丢日志
    open_log_stream(path);
    return;
  }

  const size_t previous = log_bytes_written();
  if (!open_log_stream(path)) return;
  // 轮流后在日志里留一行可追溯的标记（只记大小与份数，不涉及用户数据）
  if (log_file().is_open()) {
    auto marker = std::format(
        "[rotation] previous log reached {} bytes (>= log.max_bytes={}), "
        "rotated to {}.1, keeping {} backup(s)\n",
        previous, rot.max_bytes, path, rot.keep_files);
    log_file() << marker;
    log_file().flush();
    log_bytes_written() += marker.size();
  }
}

}  // namespace

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
  log_path() = path;
  open_log_stream(path);
}

void set_log_rotation(const LogRotation& rot) {
  std::lock_guard lock(log_mutex());
  auto& r = log_rotation();
  r.max_bytes = rot.max_bytes;
  r.keep_files = rot.keep_files < 1 ? 1 : rot.keep_files;
  if (r.max_bytes == 0) return;  // 关闭轮转
  // 设置策略时若发现当前文件已经超过上限，立刻切一次（不等下一次写入）：
  // 线上 gateway.log 已是 7.7MB，重启后应先把它归档再写新文件
  if (!log_path().empty() && log_bytes_written() >= r.max_bytes) rotate_locked();
}

void account_log_bytes(size_t n) {
  auto& rot = log_rotation();
  log_bytes_written() += n;
  if (rot.max_bytes == 0) return;
  if (log_bytes_written() >= rot.max_bytes) rotate_locked();
}

}  // namespace ai_gateway::detail
