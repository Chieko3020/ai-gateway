// Router 路由分发单元测试
#include "test_check.h"
#include <iostream>
#include "server/router.h"
using namespace ai_gateway;

int main() {
    int ok = 0;
    Router r;

    int called_a = 0, called_b = 0;
    r.add("POST", "/v1/chat/completions", [&](auto) { called_a++; return HttpReply{200, "application/json", "a"}; });
    r.add("POST", "/v1/embeddings", [&](auto) { called_b++; return HttpReply{200, "application/json", "b"}; });

    auto* h1 = r.find("POST", "/v1/chat/completions");
    CHECK(h1 != nullptr); (*h1)("x"); CHECK(called_a == 1); ok++;

    auto* h2 = r.find("POST", "/v1/embeddings");
    CHECK(h2 != nullptr); (*h2)("x"); CHECK(called_b == 1); ok++;

    // 状态码由 handler 决定（上游 4xx/5xx 透传的落点）
    r.add("POST", "/v1/errors", [&](auto) { return HttpReply{429, "application/json", "e"}; });
    auto* h3 = r.find("POST", "/v1/errors");
    CHECK(h3 != nullptr); CHECK((*h3)("x").status_code == 429); ok++;

    // Not found
    CHECK(r.find("GET", "/v1/chat/completions") == nullptr); ok++;
    CHECK(r.find("POST", "/nonexistent") == nullptr); ok++;

    return test_check::finish("test_router", ok);
}
