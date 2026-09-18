// HTTP 响应构造单元测试
#include "test_check.h"
#include <iostream>
#include <string>
#include "common/types.h"
#include "server/response.h"
using namespace ai_gateway;

int main() {
    int ok = 0;

    auto r = make_ok_json(R"({"reply":"hello"})");
    CHECK(r.find("200 OK") != std::string::npos); ok++;
    CHECK(r.find("Content-Type: application/json") != std::string::npos); ok++;
    CHECK(r.find("Content-Length:") != std::string::npos); ok++;
    CHECK(r.find(R"({"reply":"hello"})") != std::string::npos); ok++;

    auto e400 = make_bad_request("{}");
    CHECK(e400.find("400 Bad Request") != std::string::npos); ok++;

    auto e503 = make_service_unavailable("{}");
    CHECK(e503.find("503 Service Unavailable") != std::string::npos); ok++;

    // 报告 8.7-7：413 缺文案时响应行是 "HTTP/1.1 413 Unknown"
    auto e413 = make_payload_too_large("{}");
    CHECK(e413.find("413 Payload Too Large") != std::string::npos); ok++;
    CHECK(e413.find("Unknown") == std::string::npos); ok++;
    CHECK(status_text(413) == "Payload Too Large"); ok++;

    // 上游状态码是原样透传的，常见的 4xx 也必须有文案（判别力：旧实现这里全是 Unknown）
    const int upstream_codes[] = {200, 201, 204, 301, 302, 304, 400, 401, 402, 403,
                                  404, 405, 406, 408, 409, 410, 411, 413, 414, 415,
                                  422, 429, 431, 500, 501, 502, 503, 504, 505};
    int missing = 0;
    for (int code : upstream_codes)
        if (status_text(code) == "Unknown") ++missing;
    CHECK(missing == 0); ok++;
    CHECK(status_text(401) == "Unauthorized"); ok++;
    CHECK(status_text(422) == "Unprocessable Entity"); ok++;

    // 未逐条列出的码按类别兜底，仍然不能是 Unknown
    CHECK(status_text(418) == "Client Error"); ok++;
    CHECK(status_text(499) == "Client Error"); ok++;
    CHECK(status_text(507) == "Server Error"); ok++;
    CHECK(status_text(299) == "Success"); ok++;
    // 真正越界的码保持 Unknown（不当成合法状态码）
    CHECK(status_text(0) == "Unknown"); ok++;
    CHECK(status_text(999) == "Unknown"); ok++;

    // ---- 本轮新增：chunked 长度语义与 Connection 头 ----
    // 流式响应用 chunked 承载（见 response.h）。判别力：把 chunked 分支去掉，
    // 这里会看到 Content-Length: 18446744073709551615（kChunkedLength 哨兵）
    {
        ResponseHeader h;
        h.status_code = 200;
        h.content_type = "text/event-stream";
        h.chunked = true;
        h.keep_alive = true;
        auto head = build_response_head(h);
        CHECK(head.find("200 OK") != std::string::npos); ok++;
        CHECK(head.find("Content-Type: text/event-stream") != std::string::npos); ok++;
        CHECK(head.find("Transfer-Encoding: chunked") != std::string::npos); ok++;
        CHECK(head.find("Content-Length") == std::string::npos); ok++;
        CHECK(head.find("Connection: keep-alive") != std::string::npos); ok++;
        // 头必须以空行结束（否则第一个 chunk 会被当成头的一部分）
        CHECK(head.size() >= 4 && head.compare(head.size() - 4, 4, "\r\n\r\n") == 0); ok++;

        // Content-Type 原样透传：上游给什么就是什么（本轮要求，不能统一成 JSON）
        ResponseHeader h2;
        h2.status_code = 200;
        h2.content_type = "text/event-stream; charset=utf-8";
        h2.content_length = 3;
        h2.keep_alive = false;
        auto head2 = build_response_head(h2);
        CHECK(head2.find("Content-Type: text/event-stream; charset=utf-8") !=
              std::string::npos); ok++;
        CHECK(head2.find("Connection: close") != std::string::npos); ok++;
        CHECK(head2.find("Content-Length: 3") != std::string::npos); ok++;

        // 空 Content-Type 兜底成 application/json（不能发出 "Content-Type: \r\n"）
        ResponseHeader h3;
        h3.content_type.clear();
        auto head3 = build_response_head(h3);
        CHECK(head3.find("Content-Type: application/json") != std::string::npos); ok++;
    }

    // chunk 编码：长度是十六进制 + CRLF 包裹；0 块只用于终止
    {
        CHECK(encode_chunk("hello") == "5\r\nhello\r\n"); ok++;
        CHECK(encode_chunk("") == ""); ok++;  // 空块不合法，不得产生 "0\r\n\r\n" 之外的块
        CHECK(encode_chunk(std::string(16, 'x')) == "10\r\n" + std::string(16, 'x') + "\r\n"); ok++;
        CHECK(std::string(kChunkedTerminator) == "0\r\n\r\n"); ok++;
    }

    return test_check::finish("test_response", ok);
}
