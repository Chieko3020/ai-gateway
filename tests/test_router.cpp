// Router 路由分发单元测试
#include <cassert>
#include <iostream>
#include "server/router.h"
using namespace ai_gateway;

int main() {
    int ok = 0;
    Router r;

    int called_a = 0, called_b = 0;
    r.add("/v1/chat/completions", [&](auto) { called_a++; return std::string("a"); });
    r.add("/v1/embeddings", [&](auto) { called_b++; return std::string("b"); });

    auto* h1 = r.find("POST", "/v1/chat/completions");
    assert(h1 != nullptr); (*h1)("x"); assert(called_a == 1); ok++;

    auto* h2 = r.find("POST", "/v1/embeddings");
    assert(h2 != nullptr); (*h2)("x"); assert(called_b == 1); ok++;

    // Not found
    assert(r.find("GET", "/v1/chat/completions") == nullptr); ok++;
    assert(r.find("POST", "/nonexistent") == nullptr); ok++;

    std::cout << "test_router: " << ok << "/4 passed\n";
    return 0;
}
