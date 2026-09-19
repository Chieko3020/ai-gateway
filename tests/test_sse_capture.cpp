// 原始 SSE 字节 → 回答文本（include/server/sse_capture.h）
//
// 为什么值得单测：这条解析路径决定"流式回源后写进缓存的是什么内容"。写错了不会
// 报错，只会让命中率、实体提取与 saved-token 估算悄悄偏离真实答案。
//
// 判别力（每条用例都能被一种错误实现弄红）：
//   2   event 里 content 为 null 时不写空串 —— 真上游的第一个事件就是
//       `"content":null`（本次真上游复测抓到的原始字节）
//   3   只取 content、不取 reasoning_content —— 推理模型把思考过程也放在 delta 里
//   4   跳过 [DONE] 而不是拿它当 JSON 解析
//   5   CRLF 分行也要按事件切 —— 用 "找 \n\n" 的简化实现会整条漏掉
//   6   多行 data 按规范以 \n 连接
//   7   单个坏事件跳过，不影响同一段里的其它事件
//   8   断流（最后事件没有空行结尾）也要收下
//   9   非流式形态（choices[].message.content）也认
#include <string>

#include "server/sse_capture.h"
#include "test_check.h"

using ai_gateway::extract_sse_content;
using ai_gateway::sse_has_done;

namespace {

std::string ev(const std::string& payload) { return "data: " + payload + "\n\n"; }

}  // namespace

int main() {
  int ok = 0;

  // 1. 多个 delta 事件按顺序拼接
  {
    std::string sse = ev(R"({"choices":[{"delta":{"content":"你好"}}]})") +
                      ev(R"({"choices":[{"delta":{"content":"，世界"}}]})") +
                      ev("[DONE]");
    CHECK(extract_sse_content(sse) == "你好，世界");
    ok++;
    CHECK(sse_has_done(sse));
    ok++;
  }

  // 2. content 为 null 的事件（真上游首个事件的形态）不产生内容，也不炸
  {
    std::string sse =
        ev(R"({"choices":[{"delta":{"content":null,"reasoning_content":"想想"}}]})") +
        ev(R"({"choices":[{"delta":{"content":"答案"}}]})");
    CHECK(extract_sse_content(sse) == "答案");
    ok++;
  }

  // 3. reasoning_content 不算回答（推理过程不是"答案"）
  {
    std::string sse = ev(R"({"choices":[{"delta":{"reasoning_content":"推理中"}}]})") +
                      ev(R"({"choices":[{"delta":{"content":"结论"}}]})");
    CHECK(extract_sse_content(sse) == "结论");
    ok++;
  }

  // 4. finish_reason 事件（content 为空串）不污染结果
  {
    std::string sse = ev(R"({"choices":[{"delta":{},"finish_reason":"stop"}]})");
    CHECK(extract_sse_content(sse).empty());
    ok++;
  }

  // 5. CRLF 分行：事件边界是 \r\n\r\n，简化实现会整条漏掉
  {
    std::string sse =
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\r\n\r\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\r\n\r\n"
        "data: [DONE]\r\n\r\n";
    CHECK(extract_sse_content(sse) == "AB");
    ok++;
    CHECK(sse_has_done(sse));
    ok++;
  }

  // 6. 多行 data：按 SSE 规范以 \n 连接后再解析（JSON 允许多行）
  {
    std::string sse = "data: {\"choices\":[{\"delta\":\ndata: {\"content\":\"分行的\"}}]}\n\n";
    CHECK(extract_sse_content(sse) == "分行的");
    ok++;
  }

  // 7. 坏事件只跳过自己，同段里其它事件照常解析
  {
    std::string sse = ev(R"({"choices":[{"delta":{"content":"前"}}]})") +
                      ev("{ 这不是 JSON") +
                      ev(R"({"choices":[{"delta":{"content":"后"}}]})");
    CHECK(extract_sse_content(sse) == "前后");
    ok++;
  }

  // 8. 断流：最后一个事件没有以空行结尾，也要收下（否则缓存里会少最后一段）
  {
    std::string sse = ev(R"({"choices":[{"delta":{"content":"前"}}]})") +
                      R"(data: {"choices":[{"delta":{"content":"尾"}}]})";
    CHECK(extract_sse_content(sse) == "前尾");
    ok++;
    CHECK(!sse_has_done(sse));  // 没有 [DONE] —— 调用方据此拒绝回填缓存
    ok++;
  }

  // 9. 非流式形态也认（choices[].message.content）
  {
    std::string sse = ev(R"({"choices":[{"message":{"content":"完整回复"}}]})");
    CHECK(extract_sse_content(sse) == "完整回复");
    ok++;
  }

  // 10. 注释行 / 其它字段（event:/id:）不影响解析
  {
    std::string sse = ": cache hit\n\nevent: message\nid: 1\ndata: "
                      R"({"choices":[{"delta":{"content":"X"}}]})"
                      "\n\n";
    CHECK(extract_sse_content(sse) == "X");
    ok++;
  }

  // 11. 空输入
  {
    CHECK(extract_sse_content("").empty());
    ok++;
    CHECK(!sse_has_done(""));
    ok++;
  }

  return test_check::finish("test_sse_capture", ok);
}
