// HNSW 召回率与性能基准：对比自研 HNSW 图索引与暴力余弦检索
//
// 两种模式：
//   1) 真实数据集模式：读 jsonl 数据集（{"q": "..."}），用 ONNX 模型编码后建索引
//      ./tests/recall_bench scripts/datasets/synthetic.jsonl model/model_int8.onnx model/vocab.txt
//   2) 合成向量模式：生成 N 个随机单位向量（可指定维度与查询数），用于测规模曲线
//      ./tests/recall_bench --synthetic 5000 --dim 512 --queries 200
//
// 测量：自检索 top-1 成功率、top-k 召回率（vs 抽样暴力检索）、两种检索延迟与加速比

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
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

// 抽样暴力检索的 top-k（返回索引）
static std::vector<size_t> bruteTopK(const std::vector<std::vector<float>>& vecs,
                                     const std::vector<float>& q, int k) {
  std::vector<std::pair<float, size_t>> all;
  all.reserve(vecs.size());
  for (size_t j = 0; j < vecs.size(); ++j) all.emplace_back(cosine(q, vecs[j]), j);
  const size_t kk = std::min<size_t>(static_cast<size_t>(k), all.size());
  std::partial_sort(all.begin(), all.begin() + kk, all.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<size_t> out;
  for (size_t i = 0; i < kk; ++i) out.push_back(all[i].second);
  return out;
}

int main(int argc, char** argv) {
  int top_k = 3;
  std::vector<std::string> msgs;
  std::vector<std::vector<float>> vecs;
  int dim = 512;
  std::string dataset_name;

  // ---- 解析参数 ----
  bool synthetic = false;
  size_t synth_n = 0;
  size_t queries = 200;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--synthetic" && i + 1 < argc) {
      synthetic = true;
      synth_n = std::stoul(argv[++i]);
    } else if (a == "--dim" && i + 1 < argc) {
      dim = std::stoi(argv[++i]);
    } else if (a == "--queries" && i + 1 < argc) {
      queries = std::stoul(argv[++i]);
    } else if (a == "--topk" && i + 1 < argc) {
      top_k = std::stoi(argv[++i]);
    } else {
      pos.push_back(a);
    }
  }

  if (synthetic) {
    dim = (dim > 0) ? dim : 512;
    std::mt19937 gen(20260918);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    vecs.reserve(synth_n);
    for (size_t i = 0; i < synth_n; ++i) {
      std::vector<float> v(dim);
      float norm = 0;
      for (int d = 0; d < dim; ++d) {
        v[d] = nd(gen);
        norm += v[d] * v[d];
      }
      norm = std::sqrt(norm) + 1e-12f;
      for (float& x : v) x /= norm;
      vecs.push_back(std::move(v));
    }
    dataset_name = "synthetic(" + std::to_string(synth_n) + " x " +
                   std::to_string(dim) + "d random unit vectors)";
    for (size_t i = 0; i < synth_n; ++i) msgs.push_back("v" + std::to_string(i));
  } else {
    const char* dataset = pos.size() > 0 ? pos[0].c_str() : "scripts/datasets/synthetic.jsonl";
    const char* model = pos.size() > 1 ? pos[1].c_str() : "model/model_int8.onnx";
    const char* vocab = pos.size() > 2 ? pos[2].c_str() : "model/vocab.txt";
    dataset_name = dataset;

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

    OnnxEmbedding emb(model, vocab, dim);
    if (!emb.ready()) {
      std::cerr << "ONNX 模型加载失败\n";
      return 1;
    }
    auto t0 = std::chrono::steady_clock::now();
    for (auto& m : msgs) vecs.push_back(emb.encode(m));
    auto enc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
    std::cout << "编码完成: " << enc_ms << "ms，平均 "
              << (msgs.empty() ? 0 : enc_ms / static_cast<long long>(msgs.size()))
              << "ms/条\n";
    if (!vecs.empty()) dim = static_cast<int>(vecs[0].size());
  }

  std::cout << "数据集: " << dataset_name << "（" << vecs.size() << " 向量）\n";

  // ---- 建索引 ----
  auto t1 = std::chrono::steady_clock::now();
  HnswIndex index(HnswConfig{dim, 16, 100, 50});
  for (size_t i = 0; i < vecs.size(); ++i)
    index.add(static_cast<int>(i), msgs[i], vecs[i]);
  auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t1)
                      .count();
  std::cout << "索引构建: " << index.size() << " 向量, " << build_ms << "ms\n";

  // ---- 查询（大规模时抽样做暴力对比）----
  const size_t total = vecs.size();
  const size_t qn = std::min(queries, total);
  std::vector<size_t> qidx;
  qidx.reserve(qn);
  for (size_t i = 0; i < qn; ++i)
    qidx.push_back(total <= queries ? i : (i * total / qn));  // 均匀抽样

  size_t self_top1 = 0, empty_results = 0;
  double recall_sum = 0;
  long long hnsw_us = 0, brute_us = 0;

  for (size_t qi : qidx) {
    auto s0 = std::chrono::steady_clock::now();
    auto res = index.search(vecs[qi], top_k);
    hnsw_us += std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - s0)
                   .count();

    auto b0 = std::chrono::steady_clock::now();
    auto bres = bruteTopK(vecs, vecs[qi], top_k);
    brute_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - b0)
                    .count();

    if (res.empty()) {
      ++empty_results;
      continue;
    }
    if (res[0].key == msgs[qi]) ++self_top1;

    size_t inter = 0;
    for (size_t a = 0; a < res.size() && a < bres.size(); ++a)
      for (size_t b = 0; b < bres.size(); ++b)
        if (static_cast<size_t>(res[a].id) == bres[b]) {
          ++inter;
          break;
        }
    recall_sum += static_cast<double>(inter) / static_cast<double>(std::min<size_t>(top_k, bres.size()));
  }

  const double n = qn ? static_cast<double>(qn) : 1.0;
  std::cout << "\n=== 结果 ===\n";
  std::cout << "查询数: " << qn << " / " << total << " 向量\n";
  std::cout << "自检索 top-1 成功率: " << self_top1 << "/" << qn << " ("
            << (100.0 * self_top1 / n) << "%)\n";
  std::cout << "空结果查询数: " << empty_results << "\n";
  std::cout << "top-" << top_k << " 召回率(vs 暴力): " << (100.0 * recall_sum / n) << "%\n";
  std::cout << "HNSW 检索平均: " << (static_cast<double>(hnsw_us) / n) << "us\n";
  std::cout << "暴力检索平均: " << (static_cast<double>(brute_us) / n) << "us\n";
  if (hnsw_us > 0)
    std::cout << "加速比: " << (static_cast<double>(brute_us) / hnsw_us) << "x\n";
  return 0;
}
