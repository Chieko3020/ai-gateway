// Curl HTTP 客户端基类
// 把 libcurl 的 C API 包装成 RAII + 链式调用
#pragma once

#include <curl/curl.h>

#include <format>
#include <string>

namespace ai_gateway {

class CurlClient {
 public:
  CurlClient() : curl_(curl_easy_init()) {
    if (curl_) {
      // libcurl 要求多线程程序必须设置 NOSIGNAL：否则超时依赖 SIGALRM，
      // 信号可能投递到任意线程（报告 M6）
      curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L);
    }
  }
  ~CurlClient() {
    if (headers_) curl_slist_free_all(headers_);
    if (curl_) curl_easy_cleanup(curl_);
  }

  CurlClient(const CurlClient&) = delete;
  CurlClient& operator=(const CurlClient&) = delete;

  // 是否初始化成功
  explicit operator bool() const { return curl_ != nullptr; }

  // 设置目标 URL（内部持有副本，确保 c_str() 在 perform 前有效）
  void set_url(const std::string& url) {
    url_ = url;
    curl_easy_setopt(curl_, CURLOPT_URL, url_.c_str());
  }

  // 设置请求体（POST），内部持有副本，确保 c_str() 在 perform 前有效
  void set_post_body(const std::string& body) {
    body_ = body;
    curl_easy_setopt(curl_, CURLOPT_POST, 1L);
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body_.c_str());
    curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE,
                     static_cast<long>(body_.size()));
  }

  // 设置超时（秒）：总时限 + 连接阶段单独时限。
  // 只设 CURLOPT_TIMEOUT 时，连接阶段（TCP SYN 重传）仍可能长时间占用本线程
  void set_timeout(int seconds) {
    curl_easy_setopt(curl_, CURLOPT_TIMEOUT, static_cast<long>(seconds));
    curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, static_cast<long>(kConnectTimeoutSeconds));
  }

  // 添加 Authorization Bearer 头（日志中apikey脱敏为 ***）
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
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_);
    CURLcode res = curl_easy_perform(curl_);
    if (res == CURLE_OK && http_code) {
      curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, http_code);
    }
    return res;
  }

  // 裸句柄：仅供需要自带传输回调（CURLOPT_HEADERFUNCTION / WRITEFUNCTION）的
  // 流式调用方使用（backend/llm_client 的 call_llm_stream）。
  // 拿到句柄后仍需自己设置 CURLOPT_HTTPHEADER = headers()，否则鉴权头会丢
  CURL* raw() const { return curl_; }
  curl_slist* headers() const { return headers_; }

  // 与 perform() 同一条 URL/超时/鉴权配置，但不装 write 回调（由调用方自装），
  // 且无论 CURLE_OK 与否都回填 http_code（例如 write 回调主动 abort 时，
  // 上游真实状态码仍然可读，上层才能区分"上游报错"与"客户端断开"）
  CURLcode perform_raw(long* http_code) {
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_);
    CURLcode res = curl_easy_perform(curl_);
    if (http_code) {
      long code = 0;
      curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &code);
      *http_code = code;
    }
    return res;
  }

  // 重置句柄状态（保留底层连接），用于连接池复用
  void reset() {
    if (headers_) {
      curl_slist_free_all(headers_);
      headers_ = nullptr;
    }
    if (curl_) {
      curl_easy_reset(curl_);
    }
  }

 private:
  // 连接阶段上限：取总超时与 10s 的较小值，保证慢后端不占满线程
  static constexpr int kConnectTimeoutSeconds = 10;

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
  std::string url_;   // 持有 URL 副本，避免 c_str() 悬空
  std::string body_;  // 持有 POST body 副本，避免 c_str() 悬空
};

}  // namespace ai_gateway
