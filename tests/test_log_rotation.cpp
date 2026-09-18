// 日志轮转回归（报告 8.7 第 4 条：线上 gateway.log 已 7.7MB 无切分）
//
// 覆盖点：
//   1. 单文件达到 max_bytes 后切分：gateway.log → gateway.log.1，旧的依次后移
//   2. 只保留 keep_files 份历史，最老的一份被删除（不会无限堆积）
//   3. 切分点落在行边界：不出现被 rename 截断的半行
//   4. max_bytes=0 关闭轮转（回到"单文件一直追加"的旧行为）
//   5. 启动时文件已超限 → 立刻归档（重启后不继续往超限文件里追加）
//   6. 轮转不改变权限（新文件仍是 0600）
//
// 判别力说明：把 log_file.cpp 的 account_log_bytes() 改成空实现（等价于修复前的
// "只追加不切分"）后，第 1/2/5/6 段断言失败：只有 gateway.log 一个文件、
// 大小远超 max_bytes，且 .1 不存在。
#include <sys/stat.h>
#include <unistd.h>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>

#include "common/log_file.h"
#include "common/logger.h"
#include "test_check.h"

using namespace ai_gateway;

namespace {

std::string g_dir;

std::string path_of(const std::string& name) { return g_dir + "/" + name; }

bool file_exists(const std::string& p) { return ::access(p.c_str(), F_OK) == 0; }

size_t file_size(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) return 0;
  auto pos = f.tellg();
  return pos > 0 ? static_cast<size_t>(pos) : 0;
}

std::string read_all(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

// 残行计数：读回来的每一行都应以 '[' 开头；文件尾字节必须是 '\n'
// （切分发生在写完一整行之后，所以任何被切出去的文件都不可能以半行结尾）
struct FileShape {
  size_t bad_prefix = 0;
  bool ends_with_newline = true;
};

FileShape shape_of(const std::string& p) {
  FileShape s;
  std::string all = read_all(p);
  if (!all.empty() && all.back() != '\n') s.ends_with_newline = false;
  size_t start = 0;
  while (start < all.size()) {
    auto nl = all.find('\n', start);
    if (nl == std::string::npos) {  // 无换行的尾段 = 残行
      ++s.bad_prefix;
      break;
    }
    if (nl > start && all[start] != '[') ++s.bad_prefix;
    start = nl + 1;
  }
  return s;
}

// 写一条长度固定的日志行（约 190 字节），便于精确推算切分点
void write_line(int i) {
  LOG_INFO("rotation filler line {:06d} {}", i, std::string(160, 'x'));
}

}  // namespace

int main() {
  int ok = 0;

  g_dir = "/tmp/agw-log-rotate-" + std::to_string(::getpid());
  ::mkdir(g_dir.c_str(), 0700);

  const std::string cur = path_of("gateway.log");
  const size_t kMax = 4096;
  const int kKeep = 3;
  const size_t kLineMax = 400;  // 单行上限（用于判断"超出但不超过一行"）

  // ── 1/2/3. 达到上限就切分，只留 keep 份，且不切断行 ──────────────────
  {
    ai_gateway::detail::set_log_file(cur);
    ai_gateway::detail::set_log_rotation({kMax, kKeep});
    const int kLines = 60;  // 60 × ~190B ≈ 11.4KB ≈ 3 个 4KB 文件
    for (int i = 0; i < kLines; ++i) write_line(i);

    CHECK(file_exists(cur));
    ok++;
    CHECK(file_exists(cur + ".1"));
    ok++;
    CHECK(file_exists(cur + ".2"));
    ok++;
    // keep_files=3：最老的一份（.4）不能存在
    CHECK(!file_exists(cur + ".4"));
    ok++;

    // 每个归档文件不得超过 max_bytes + 一行（判定发生在写完一行之后）
    for (const std::string& p : {cur + ".1", cur + ".2", cur + ".3"}) {
      if (!file_exists(p)) continue;
      CHECK(file_size(p) <= kMax + kLineMax);
      ok++;
    }

    size_t bad = 0, fillers = 0, markers = 0;
    bool all_end_nl = true;
    for (const std::string& p : {cur, cur + ".1", cur + ".2", cur + ".3"}) {
      if (!file_exists(p)) continue;
      auto shape = shape_of(p);
      bad += shape.bad_prefix;
      if (!shape.ends_with_newline) all_end_nl = false;
      std::string content = read_all(p);
      for (size_t pos = content.find("rotation filler line");
           pos != std::string::npos;
           pos = content.find("rotation filler line", pos + 1))
        ++fillers;
      for (size_t pos = content.find("[rotation]"); pos != std::string::npos;
           pos = content.find("[rotation]", pos + 1))
        ++markers;
    }
    CHECK(bad == 0);
    ok++;
    CHECK(all_end_nl);
    ok++;
    // 归档+当前文件里的填充行数不超过写过的行数（最老的文件被删掉，所以可能更少）
    CHECK(fillers > 0 && fillers <= static_cast<size_t>(kLines));
    ok++;
    CHECK(markers >= 1);
    ok++;

    // 轮流后在当前文件里留下可追溯的标记（只含大小与份数，不含用户数据）
    CHECK(read_all(cur).find("[rotation]") != std::string::npos);
    ok++;
  }

  // ── 4. max_bytes=0：关闭轮转（旧行为，单文件一直追加）─────────────────
  {
    const std::string p2 = path_of("norotate.log");
    ai_gateway::detail::set_log_file(p2);
    ai_gateway::detail::set_log_rotation({0, 3});
    for (int i = 0; i < 60; ++i) write_line(i);
    CHECK(file_exists(p2));
    ok++;
    CHECK(!file_exists(p2 + ".1"));
    ok++;
    CHECK(file_size(p2) > kMax);  // 超过 kMax 也不切分
    ok++;
  }

  // ── 5. 启动时文件已超限：立刻归档，不继续往超限文件里追加 ─────────────
  {
    const std::string p3 = path_of("preexisting.log");
    {
      // 手工造一个 ~5.8KB 的"上次运行残留"（kMax=4096）
      std::ofstream out(p3, std::ios::binary);
      for (int i = 0; i < 30; ++i) out << "[" << std::string(190, 'y') << "]\n";
    }
    CHECK(file_size(p3) > kMax);
    ok++;
    ai_gateway::detail::set_log_file(p3);
    ai_gateway::detail::set_log_rotation({kMax, kKeep});
    CHECK(file_exists(p3 + ".1"));
    ok++;
    CHECK(file_size(p3 + ".1") > kMax);  // 旧内容完整归档
    ok++;
    CHECK(file_size(p3) < kMax);         // 当前文件从 0 起算
    ok++;
  }

  // ── 6. 权限仍是 0600（轮转不得让新文件退回 0644）─────────────────────
  {
    const std::string p4 = path_of("perm.log");
    ai_gateway::detail::set_log_file(p4);
    ai_gateway::detail::set_log_rotation({512, 2});
    for (int i = 0; i < 20; ++i) write_line(i);
    CHECK(file_exists(p4 + ".1"));
    ok++;
    for (const std::string& p : {p4, p4 + ".1"}) {
      struct stat st {};
      if (::stat(p.c_str(), &st) != 0) { CHECK(false); ok++; continue; }
      CHECK((st.st_mode & 0777) == 0600);
      ok++;
    }
  }

  // 收尾：删掉测试产生的日志文件与临时目录
  for (const char* base : {"gateway.log", "norotate.log", "preexisting.log",
                           "perm.log"}) {
    for (const char* suffix : {"", ".1", ".2", ".3", ".4"})
      ::unlink((path_of(base) + suffix).c_str());
  }
  ::rmdir(g_dir.c_str());

  return test_check::finish("test_log_rotation", ok);
}
