// HTTP 响应构造单元测试
#include <cassert>
#include <iostream>
#include "response.h"
using namespace ai_gateway;

int main() {
    int ok = 0;

    auto r = make_ok_json(R"({"reply":"hello"})");
    assert(r.find("200 OK") != std::string::npos); ok++;
    assert(r.find("Content-Type: application/json") != std::string::npos); ok++;
    assert(r.find("Content-Length:") != std::string::npos); ok++;
    assert(r.find(R"({"reply":"hello"})") != std::string::npos); ok++;

    auto e400 = make_bad_request("{}");
    assert(e400.find("400 Bad Request") != std::string::npos); ok++;

    auto e503 = make_service_unavailable("{}");
    assert(e503.find("503 Service Unavailable") != std::string::npos); ok++;

    std::cout << "test_response: " << ok << "/6 passed\n";
    return 0;
}
