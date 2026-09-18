// HNSW 召回率基准：对比自研 HNSW 图索引与暴力余弦检索
//
// 方法：把数据集全部向量加入索引后逐条查询，测量
//   1) 自检索 top-1 成功率（查询自己，top-1 是否为自己）
//   2) top-k 与暴力搜索的交集召回率
//   3) 两种检索的平均耗时与加速比
//
// 用法（需在项目根目录执行，模型与数据集用相对路径）:
//   ./tests/recall_bench scripts/datasets/synthetic.jsonl model/model_int8.onnx model/vocab.txt

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "cache/hnsw_index.h"
#include "cache/onnx_embedding.h"

using json = nlohmann::json;
using namespace ai_gateway;

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  float dot = 0, na = 0, nb = 0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return (na > 0 && nb > 0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0f;
}

int main(int argc, char** argv) {
  const char* dataset = (argc > 1) ? argv[1] : "scripts/datasets/synthetic.jsonl";
  const char* model = (argc > 2) ? argv[2] : "model/model_int8.onnx";
  const char* vocab = (argc > 3) ? argv[3] : "model/vocab.txt";
  const int top_k = 3;

  std::vector<std::string> msgs;
  {
    std::ifstream f(dataset);
    if (!f) {
      std::cerr << "无法打开数据集: " << dataset << "\n";
      return 1;
    }
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      try {
        msgs.push_back(json::parse(line).value("q", ""));
      } catch (...) {
        msgs.push_back(line);
      }
    }
  }
  std::cout << "数据集: " << dataset << " (" << msgs.size() << " 条)\n";

  OnnxEmbedding emb(model, vocab, 512);
  if (!emb.ready()) {
    std::cerr << "ONNX 模型加载失败\n";
    return 1;
  }

  std::vector<std::vector<float>> vecs;
  vecs.reserve(msgs.size());
  auto t0 = std::chrono::steady_clock::now();
  for (auto& m : msgs) vecs.push_back(emb.encode(m));
  auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
  std::cout << "编码完成: " << enc_ms << "ms，平均 "
            << (msgs.empty() ? 0 : enc_ms / static_cast<long long>(msgs.size()))
            << "ms/条\n";

  HnswIndex index(HnswConfig{512, 16, 100, 50});
  for (size_t i = 0; i < vecs.size(); ++i)
    index.add(static_cast<int>(i), msgs[i], vecs[i]);
  std::cout << "索引构建完成: " << index.size() << " 向量\n";

  size_t self_top1 = 0, empty_results = 0;
  double recall_sum = 0;
  long long hnsw_us = 0, brute_us = 0;

  for (size_t i = 0; i < vecs.size(); ++i) {
    auto s0 = std::chrono::steady_clock::now();
    auto res = index.search(vecs[i], top_k);
    hnsw_us += std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - s0)
                   .count();

    auto b0 = std::chrono::steady_clock::now();
    std::vector<std::pair<float, size_t>> all;
    all.reserve(vecs.size());
    for (size_t j = 0; j < vecs.size(); ++j)
      all.emplace_back(cosine(vecs[i], vecs[j]), j);
    const size_t kk = std::min<size_t>(top_k, all.size());
    std::partial_sort(all.begin(), all.begin() + kk, all.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    brute_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - b0)
                    .count();

    if (res.empty()) {
      ++empty_results;
      continue;
    }
    if (res[0].key == msgs[i]) ++self_top1;

    size_t inter = 0;
    for (size_t a = 0; a < res.size() && a < kk; ++a)
      for (size_t b = 0; b < kk; ++b)
        if (res[a].key == msgs[all[b].second]) {
          ++inter;
          break;
        }
    recall_sum += static_cast<double>(inter) / static_cast<double>(kk);
  }

  const double n = vecs.empty() ? 1.0 : static_cast<double>(vecs.size());
  std::cout << "\n=== 结果 ===\n";
  std::cout << "查询数: " << vecs.size() << "\n";
  std::cout << "自检索 top-1 成功率: " << self_top1 << "/" << vecs.size() << " ("
            << (100.0 * self_top1 / n) << "%)\n";
  std::cout << "空结果查询数: " << empty_results << "\n";
  std::cout << "top-" << top_k << " 召回率(vs 暴力): " << (100.0 * recall_sum / n)
            << "%\n";
  std::cout << "HNSW 检索平均: " << (static_cast<double>(hnsw_us) / n) << "us\n";
  std::cout << "暴力检索平均: " << (static_cast<double>(brute_us) / n) << "us\n";
  if (hnsw_us > 0)
    std::cout << "加速比: " << (static_cast<double>(brute_us) / hnsw_us) << "x\n";
  return 0;
}
