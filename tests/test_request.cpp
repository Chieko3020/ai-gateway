// HTTP 请求解析单元测试
#include "test_check.h"
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
    CHECK(req.valid && req.method == "POST"); ok++;
    CHECK(req.path == "/v1/chat/completions"); ok++;
    CHECK(req.content_length == 27); ok++;
    CHECK(req.body == R"({"model":"test","msg":"hi"})"); ok++;
    CHECK(req.header("Content-Type") == "application/json"); ok++;

    // GET request (no body)
    auto get = parse_request("GET /health HTTP/1.1\r\n\r\n", 24);
    CHECK(get.valid && get.method == "GET"); ok++;
    CHECK(get.path == "/health"); ok++;
    CHECK(get.content_length == 0); ok++;

    // Garbage means invalid
    CHECK(!parse_request("garbage\r\n", 9).valid); ok++;

    return test_check::finish("test_request", ok);
}
