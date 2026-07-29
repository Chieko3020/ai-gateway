// Curl HTTP 客户端基类：消除 llm_client 和 embedding 的重复代码
#pragma once

#include <curl/curl.h>

#include <format>
#include <string>

namespace ai_gateway {

class CurlClient {
 public:
  CurlClient() : curl_(curl_easy_init()) {}
  ~CurlClient() {
    if (headers_) curl_slist_free_all(headers_);
    if (curl_) curl_easy_cleanup(curl_);
  }

  CurlClient(const CurlClient&) = delete;
  CurlClient& operator=(const CurlClient&) = delete;

  // 是否初始化成功
  explicit operator bool() const { return curl_ != nullptr; }

  // 设置目标 URL
  void set_url(const std::string& url) {
    curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
  }

  // 设置请求体（POST）
  void set_post_body(const std::string& body) {
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(body.size()));
  }

  // 设置超时（秒）
  void set_timeout(int seconds) {
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, static_cast<long>(seconds));
  }

  // 添加 Authorization Bearer 头
  // 以及通用 Content-Type: application/json
  void set_auth(const std::string& api_key) {
    add_header(std::format("Authorization: Bearer {}", api_key));
    add_header("Content-Type: application/json");
  }

  // 设置响应接收缓冲区
  void set_response_target(std::string* target) {
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, target);
  }

  // 执行请求，返回 HTTP 状态码；CURLE_OK 时 http_code 有效
  CURLcode perform(long* http_code) {
    CURLcode res = curl_easy_perform(curl_);
    if (res == CURLE_OK && http_code) {
      curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, http_code);
    }
    return res;
  }

 private:
  void add_header(const std::string& h) {
    headers_ = curl_slist_append(headers_, h.c_str());
  }

  static size_t write_callback(char* ptr, size_t size, size_t nmemb,
                               void* userdata) {
    if (!userdata) return 0;
    auto* body = static_cast<std::string*>(userdata);
    body->append(ptr, size * nmemb);
    return size * nmemb;
  }

  CURL* curl_ = nullptr;
  struct curl_slist* headers_ = nullptr;
};

}  // namespace ai_gateway
