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

    // 报告 L6：path 必须剥离 query string（路由表是精确匹配）
    {
        const char* q =
            "POST /v1/chat/completions?x=1&y=2 HTTP/1.1\r\n"
            "Content-Length: 2\r\n\r\n{}";
        auto rq = parse_request(q, std::strlen(q));
        CHECK(rq.valid); ok++;
        CHECK(rq.path == "/v1/chat/completions"); ok++;
        CHECK(rq.body == "{}"); ok++;
    }

    // 报告 M7：header 查表大小写不敏感（与服务器侧 parse_content_length 口径统一）。
    // 判别力：改回精确匹配的 unordered_map 查表时这条会失败（同时 content_length
    // 也解析不出来，因为 request.cpp 内部就是通过 header() 取 Content-Length）
    {
        const char* mixed =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "content-length: 2\r\n"
            "CONTENT-TYPE: application/json\r\n"
            "X-Custom-Header: v\r\n"
            "\r\n{}";
        auto rq = parse_request(mixed, std::strlen(mixed));
        CHECK(rq.valid); ok++;
        CHECK(rq.content_length == 2); ok++;
        CHECK(rq.body == "{}"); ok++;
        CHECK(rq.header("Content-Length") == "2"); ok++;
        CHECK(rq.header("content-type") == "application/json"); ok++;
        CHECK(rq.header("X-CUSTOM-HEADER") == "v"); ok++;
        CHECK(rq.header("X-Not-Present").empty()); ok++;
    }

    // 报告 M7：Transfer-Encoding: chunked 显式拒绝（不再把原始 chunked 报文
    // 当成 body 转发给上游）
    {
        const char* chunked =
            "POST /v1/chat/completions HTTP/1.1\r\n"
            "Transfer-Encoding: chunked\r\n"
            "\r\n"
            "5\r\nhello\r\n0\r\n\r\n";
        auto rq = parse_request(chunked, std::strlen(chunked));
        CHECK(!rq.valid); ok++;
    }

    return test_check::finish("test_request", ok);
}
