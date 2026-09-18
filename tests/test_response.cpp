// HTTP 响应构造单元测试
#include "test_check.h"
#include <iostream>
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

    return test_check::finish("test_response", ok);
}
