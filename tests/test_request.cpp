// HTTP 请求解析单元测试
#include <cassert>
#include <cstring>
#include <iostream>
#include "server/request.h"
using namespace ai_gateway;

int main() {
    int ok = 0;

    // POST with body
    const char* raw =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 27\r\n"
        "\r\n"
        "{\"model\":\"test\",\"msg\":\"hi\"}";
    auto req = parse_request(raw, std::strlen(raw));
    assert(req.valid && req.method == "POST"); ok++;
    assert(req.path == "/v1/chat/completions"); ok++;
    assert(req.content_length == 27); ok++;
    assert(req.body == R"({"model":"test","msg":"hi"})"); ok++;
    assert(req.header("Content-Type") == "application/json"); ok++;

    // GET request (no body)
    auto get = parse_request("GET /health HTTP/1.1\r\n\r\n", 23);
    assert(get.valid && get.method == "GET"); ok++;
    assert(get.path == "/health"); ok++;
    assert(get.content_length == 0); ok++;

    // Garbage → invalid
    assert(!parse_request("garbage\r\n", 9).valid); ok++;

    std::cout << "test_request: " << ok << "/9 passed\n";
    return 0;
}
