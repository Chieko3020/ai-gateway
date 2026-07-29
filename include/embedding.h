// Embedding API 客户端：将文本向量化（OpenAI 兼容格式）
#pragma once

#include <string>
#include <vector>

namespace ai_gateway {

// 调用 embedding API，返回浮点向量；失败返回空 vector
std::vector<float> get_embedding(const std::string& url,
                                  const std::string& api_key,
                                  const std::string& model,
                                  const std::string& text,
                                  int timeout_seconds = 30);

}  // namespace ai_gateway
