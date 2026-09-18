// Router 路由分发单元测试
#include <cassert>
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
    assert(h1 != nullptr); (*h1)("x"); assert(called_a == 1); ok++;

    auto* h2 = r.find("POST", "/v1/embeddings");
    assert(h2 != nullptr); (*h2)("x"); assert(called_b == 1); ok++;

    // 状态码由 handler 决定（上游 4xx/5xx 透传的落点）
    r.add("POST", "/v1/errors", [&](auto) { return HttpReply{429, "application/json", "e"}; });
    auto* h3 = r.find("POST", "/v1/errors");
    assert(h3 != nullptr); assert((*h3)("x").status_code == 429); ok++;

    // Not found
    assert(r.find("GET", "/v1/chat/completions") == nullptr); ok++;
    assert(r.find("POST", "/nonexistent") == nullptr); ok++;

    std::cout << "test_router: " << ok << "/5 passed\n";
    return 0;
}
