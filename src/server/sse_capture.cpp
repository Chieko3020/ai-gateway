#include "server/sse_capture.h"

#include <nlohmann/json.hpp>

namespace ai_gateway {

bool sse_has_done(std::string_view sse) {
  // 不做事件级解析：终止事件在任何合法写法下都含这段字面量
  // （`data: [DONE]` / `data:[DONE]` 都会含 "data:" 后的 "[DONE]"）
  return sse.find("[DONE]") != std::string_view::npos;
}

std::string extract_sse_content(std::string_view sse) {
  std::string out;
  std::string payload;  // 当前事件累积的 data 载荷

  // 按 SSE 规范逐行处理：空行结束一个事件，`data:` 行累积载荷。
  // 这里不用 "找 \n\n" 的简化写法：合法上游可能用 \r\n 分行，那样
  // `\r\n\r\n` 里并不存在连续的 "\n\n"，会整条漏掉（真上游复测用的是 CRLF 容忍路）
  auto flush_event = [&out, &payload]() {
    if (payload.empty()) return;
    if (payload != "[DONE]") {
      auto j = nlohmann::json::parse(payload, nullptr, /*allow_exceptions=*/false);
      if (!j.is_discarded() && j.contains("choices") && j["choices"].is_array()) {
        for (const auto& choice : j["choices"]) {
          const nlohmann::json* container = nullptr;
          if (choice.contains("delta") && choice["delta"].is_object())
            container = &choice["delta"];
          else if (choice.contains("message") && choice["message"].is_object())
            container = &choice["message"];
          if (container == nullptr) continue;
          auto it = container->find("content");
          if (it != container->end() && it->is_string())
            out += it->get<std::string>();
        }
      }
    }
    payload.clear();
  };

  size_t pos = 0;
  while (pos <= sse.size()) {
    size_t nl = sse.find('\n', pos);
    std::string_view line =
        (nl == std::string_view::npos) ? sse.substr(pos) : sse.substr(pos, nl - pos);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    pos = (nl == std::string_view::npos) ? sse.size() + 1 : nl + 1;

    if (line.empty()) {
      flush_event();
      continue;
    }
    if (line.starts_with("data:")) {
      std::string_view part = line.substr(5);
      if (!part.empty() && part.front() == ' ') part.remove_prefix(1);
      if (!payload.empty()) payload += '\n';  // 多行 data 按规范以 \n 连接
      payload.append(part);
    }
    // 其它字段（event:/id:/retry:/注释行）与解析无关，直接忽略
  }
  flush_event();  // 最后一段没有以空行结尾（上游断流）时也要收
  return out;
}

}  // namespace ai_gateway
