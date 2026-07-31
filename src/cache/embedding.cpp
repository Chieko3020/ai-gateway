// Embedding API 客户端实现
#include "cache/embedding.h"

#include <nlohmann/json.hpp>

#include <format>

#include "common/curl_client.h"
#include "common/logger.h"

using json = nlohmann::json;

namespace ai_gateway {

std::vector<float> get_embedding(const std::string& url,
                                  const std::string& api_key,
                                  const std::string& model,
                                  const std::string& text,
                                  int timeout_seconds) {
  std::string response_body;
  CurlClient curl;
  if (!curl) {
    LOG_ERROR("embedding: curl_easy_init failed");
    return {};
  }
  // 构造请求 JSON
  json req_body;
  req_body["model"] = model;
  req_body["input"] = text;
  std::string req_str = req_body.dump();

  curl.set_url(url);
  curl.set_post_body(req_str);
  curl.set_timeout(timeout_seconds);
  curl.set_auth(api_key);
  curl.set_response_target(&response_body);

  long http_code = 0;
  CURLcode res = curl.perform(&http_code);

  if (res != CURLE_OK) {
    LOG_WARN("embedding: curl error {}", static_cast<int>(res));
    return {};
  }
  if (http_code != 200) {
    LOG_WARN("embedding: HTTP {} {}", http_code,
             response_body.size() > 200
                 ? response_body.substr(0, 200)
                 : response_body);
    return {};
  }
  // 解析 JSON 响应 提取 embedding 向量
  try {
    auto resp = json::parse(response_body);
    auto& data = resp.at("data");
    if (data.empty()) return {};
    auto& emb = data[0].at("embedding");
    std::vector<float> vec;
    vec.reserve(emb.size());
    for (const auto& v : emb) vec.push_back(v.get<float>());
    LOG_DEBUG("embedding: got {} dims", vec.size());
    return vec;
  } catch (const std::exception& e) {
    LOG_ERROR("embedding: parse error: {}", e.what());
    return {};
  }
}

}  // namespace ai_gateway
