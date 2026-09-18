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

    return test_check::finish("test_response", ok);
}
