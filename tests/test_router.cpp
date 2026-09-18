// Router 路由分发单元测试
//
// handler 签名本轮加了 ResponseWriter&（SSE 透传要边收边发）：本用例只测路由
// 分发本身，因此给一个写不进去的 writer（fd=-1 + 已过期死线，任何写都会失败），
// 顺带断言"缓冲式 handler 确实用不到 writer"
#include "test_check.h"
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>

#include "server/http_server.h"
#include "server/router.h"
using namespace ai_gateway;

int main() {
    int ok = 0;
    Router r;

    // 一个永远不会被写成功的 writer：fd=-1、死线已过
    std::atomic<uint64_t> eagain{0};
    ResponseWriter writer(-1, /*write_deadline_ms=*/1,
                          std::chrono::steady_clock::now(), &eagain);
    (void)writer;
    HttpRequestInfo info;
    info.keep_alive = true;

    int called_a = 0, called_b = 0;
    r.add("POST", "/v1/chat/completions", [&](auto, auto&, auto&) { called_a++; return HttpReply{200, "application/json", "a"}; });
    r.add("POST", "/v1/embeddings", [&](auto, auto&, auto&) { called_b++; return HttpReply{200, "application/json", "b"}; });

    auto* h1 = r.find("POST", "/v1/chat/completions");
    CHECK(h1 != nullptr); (*h1)("x", writer, info); CHECK(called_a == 1); ok++;

    auto* h2 = r.find("POST", "/v1/embeddings");
    CHECK(h2 != nullptr); (*h2)("x", writer, info); CHECK(called_b == 1); ok++;

    // 状态码由 handler 决定（上游 4xx/5xx 透传的落点）
    r.add("POST", "/v1/errors", [&](auto, auto&, auto&) { return HttpReply{429, "application/json", "e"}; });
    auto* h3 = r.find("POST", "/v1/errors");
    CHECK(h3 != nullptr); CHECK((*h3)("x", writer, info).status_code == 429); ok++;

    // Not found
    CHECK(r.find("GET", "/v1/chat/completions") == nullptr); ok++;
    CHECK(r.find("POST", "/nonexistent") == nullptr); ok++;

    return test_check::finish("test_router", ok);
}
