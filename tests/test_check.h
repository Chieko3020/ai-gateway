// 测试断言工具
//
// 不用 <cassert> 的 assert()：Release 构建（-DCMAKE_BUILD_TYPE=Release）会定义
// NDEBUG，assert 整体被预处理剔除，测试于是"全绿"但一条断言也没执行。
// CHECK 与 NDEBUG 无关：失败时打印表达式与 文件:行号 并累计失败数，
// 由 finish() 转成非零退出码供 ctest 判定失败。
#pragma once

#include <cstdio>

namespace test_check {

inline int& failures() {
  static int n = 0;
  return n;
}

inline int& checks() {
  static int n = 0;
  return n;
}

inline void fail(const char* expr, const char* file, int line) {
  ++failures();
  std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", expr, file, line);
  std::fflush(stderr);
}

// 打印汇总：返回进程退出码（0 表示全部通过）
// executed = 本次实际执行的断言条数（调用方在每条断言后 ok++，与断言结果无关）
inline int finish(const char* name, int executed) {
  if (failures() == 0) {
    std::printf("%s: %d/%d passed\n", name, executed, executed);
    return 0;
  }
  std::printf("%s: FAILED (%d of %d checks failed)\n", name, failures(),
              executed);
  return 1;
}

}  // namespace test_check

#define CHECK(cond)                                                   \
  do {                                                                \
    ++::test_check::checks();                                         \
    if (!(cond)) ::test_check::fail(#cond, __FILE__, __LINE__);        \
  } while (0)
