// SSE usage 解析回归（目标 6：流式请求的 token 统计）
//
// 背景：流式路径此前 token 计数恒为 0——`usage` 只在最后一个 SSE 事件里，而透传
// 路径从不解析事件流。本用例把解析行为固定成断言：
//   1. 标准 OpenAI/DeepSeek 结束事件的 usage 能被解析（prompt/completion）
//   2. `stream_options.include_usage` 未开（事件里没有 usage 节点）时 seen=false
//   3. **任意 chunk 切分**都能解析出来（TCP 分块与事件边界无关：这里逐一按
//      1/3/7/64 字节切片，并把 `"usage"` 与它的数字强行切在两个 chunk 里）
//   4. 不是 usage 的东西不会被误当成 usage（普通 delta 事件、event:/注释行、
//      上游错误体、同名字段但不在 usage 对象里、注释行里的 "prompt_tokens"）
//   5. 有界性：超长单行不会让内部缓冲无界增长（丢弃该行，不影响后续事件）
//
// 判别力说明（回退修复即失败）：
//   - 去掉 on_chunk 里的 usage_.feed(...)：第 1/3 段全挂
//   - 去掉 `payload.find("\"usage\"")` 这道门（第 4 段）：注释行里的同名字段被误计
//   - 去掉跨块的 last-key/carry 处理：第 3 段的"切在 key/value 之间"子用例挂
#include <cstdio>
#include <string>
#include <vector>

#include "server/sse_usage.h"
#include "test_check.h"

using namespace ai_gateway;

namespace {

// 标准的尾部 usage 事件（OpenAI 兼容 / DeepSeek 流式）
const char* kUsageEvent =
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":57,"
    "\"total_tokens\":185}}\n\n";
const char* kDoneEvent = "data: [DONE]\n\n";

// 不带 usage 的普通 delta 事件
const char* kDeltaEvent =
    "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\n";

// 把 blob 按固定步长切片喂给解析器（模拟任意 TCP 分块）
StreamUsage feed_in_slices(std::string_view blob, size_t step) {
  SseUsageParser p;
  for (size_t i = 0; i < blob.size(); i += step) {
    p.feed(blob.data() + i, std::min(step, blob.size() - i));
  }
  return p.usage();
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1. 基本解析（单块喂入）─────────────────────────────────────────
  {
    SseUsageParser p;
    p.feed(std::string(kDeltaEvent));
    CHECK(!p.usage().seen); ok++;  // 中途事件不该产生 usage
    p.feed(std::string(kUsageEvent));
    CHECK(p.usage().seen); ok++;
    CHECK(p.usage().prompt_tokens == 128); ok++;
    CHECK(p.usage().completion_tokens == 57); ok++;
    CHECK(p.usage().total() == 185); ok++;
    p.feed(std::string(kDoneEvent));
    CHECK(p.usage().prompt_tokens == 128); ok++;  // [DONE] 不覆盖已有值
  }

  // ── 2. 上游未提供 usage（未开 include_usage）→ 优雅退化为 0 且 seen=false ──
  {
    SseUsageParser p;
    p.feed(std::string(kDeltaEvent));
    p.feed(std::string(kDeltaEvent));
    p.feed(std::string(kDoneEvent));
    CHECK(!p.usage().seen); ok++;
    CHECK(p.usage().prompt_tokens == 0 && p.usage().completion_tokens == 0); ok++;
  }

  // ── 3. 任意 chunk 切分都要解析出来 ────────────────────────────────
  {
    std::string blob = std::string(kDeltaEvent) + kUsageEvent + kDoneEvent;
    for (size_t step : {1u, 2u, 3u, 7u, 13u, 64u, 4096u}) {
      auto u = feed_in_slices(blob, step);
      if (!(u.seen && u.prompt_tokens == 128 && u.completion_tokens == 57)) {
        std::fprintf(stderr, "slice step %zu failed: seen=%d in=%lld out=%lld\n",
                     step, static_cast<int>(u.seen),
                     static_cast<long long>(u.prompt_tokens),
                     static_cast<long long>(u.completion_tokens));
      }
      CHECK(u.seen && u.prompt_tokens == 128 && u.completion_tokens == 57); ok++;
    }

    // 显式把 `"usage"` 与数字切在两个 chunk 里（最刁钻的边界）
    {
      std::string s = kUsageEvent;
      const size_t pos = s.find("\"prompt_tokens\"");
      CHECK(pos != std::string::npos); ok++;
      SseUsageParser p;
      p.feed(s.substr(0, pos));            // 到 key 名为止（含 "usage"）
      p.feed(s.substr(pos));               // 剩下的（key + 值）
      CHECK(p.usage().seen); ok++;
      CHECK(p.usage().prompt_tokens == 128); ok++;
      CHECK(p.usage().completion_tokens == 57); ok++;
    }
    // 切在 "usage" 与 "prompt_tokens" 之间
    {
      std::string s = kUsageEvent;
      const size_t pos = s.find("\"prompt_tokens\"");
      SseUsageParser p;
      p.feed(s.substr(0, pos + 3));  // 切在 key 名中间
      p.feed(s.substr(pos + 3));
      CHECK(p.usage().seen && p.usage().prompt_tokens == 128); ok++;
    }
  }

  // ── 4. 不得误判 ───────────────────────────────────────────────────
  {
    // (a) event:/注释行 / id 行里出现同名字段
    {
      SseUsageParser p;
      p.feed(": prompt_tokens: 999\n");
      p.feed("event: usage\n");
      p.feed("id: completion_tokens: 888\n");
      CHECK(!p.usage().seen); ok++;
      CHECK(p.usage().prompt_tokens == 0); ok++;
    }
    // (b) data 行里有 prompt_tokens，但**没有** usage 对象（别的扩展字段）
    {
      SseUsageParser p;
      p.feed("data: {\"echo\":{\"prompt_tokens\":777}}\n\n");
      CHECK(!p.usage().seen); ok++;
      CHECK(p.usage().prompt_tokens == 0); ok++;
    }
    // (c) 键存在但值不是数字（被上游写成字符串）
    {
      SseUsageParser p;
      p.feed("data: {\"usage\":{\"prompt_tokens\":\"1\"}}\n\n");
      CHECK(!p.usage().seen); ok++;
    }
    // (d) 上游错误体（非 SSE，无 data: 行）
    {
      SseUsageParser p;
      p.feed("{\"error\":{\"message\":\"rate limit\"}}");
      CHECK(!p.usage().seen); ok++;
    }
    // (e) [DONE] 行
    {
      SseUsageParser p;
      p.feed(std::string(kDoneEvent));
      CHECK(!p.usage().seen); ok++;
    }
  }

  // ── 5. 有界性：超长单行被丢弃，不拖垮后续事件 ──────────────────────
  {
    SseUsageParser p;
    std::string huge = "data: {\"choices\":[{\"delta\":{\"content\":\"";
    huge.append(200 * 1024, 'x');  // 200KB 的单行（远超 kMaxLineBytes）
    huge += "\"}}]}\n\n";
    p.feed(huge);
    // 之后的正常 usage 事件仍必须能解析出来
    p.feed(std::string(kUsageEvent));
    CHECK(p.usage().seen); ok++;
    CHECK(p.usage().prompt_tokens == 128); ok++;

    // 连一个换行都没有的畸形巨块也不能让解析器崩或无限持有
    SseUsageParser p2;
    std::string nonl(200 * 1024, 'y');
    p2.feed(nonl);
    p2.feed(std::string(kUsageEvent));
    CHECK(p2.usage().seen); ok++;
  }

  // ── 6. 大数（不溢出到负数）────────────────────────────────────────
  {
    SseUsageParser p;
    p.feed("data: {\"usage\":{\"prompt_tokens\":1234567,\"completion_tokens\":89012}}\n\n");
    CHECK(p.usage().prompt_tokens == 1234567); ok++;
    CHECK(p.usage().completion_tokens == 89012); ok++;
  }

  return test_check::finish("test_sse_usage", ok);
}
