// embedding 指纹版本化回归（本轮新增）
//
// 背景：落盘缓存只存向量，不记录"它由哪个模型/分词器/维度产生"。换模型或改分词
// （本项目刚补过 WordPiece 规则）之后新旧向量不在同一空间，余弦相似度失去意义，
// 且**不会报错**——只会静默返回错误答案。本用例把三种情形固定成断言：
//
//   1. 指纹计算本身：模型/词表内容任意一个字节不同 -> 指纹不同；维度/分词器标识
//      参与指纹；文件读不到时指纹不可用（而不是"算出一个能匹配上的空指纹"）
//   2. 指纹一致：向量原样加载，索引可重建
//   3. 指纹不一致：**丢弃向量、保留文本**，并标记 mismatch（绝不把旧向量当新向量）
//   4. 旧格式文件（没有指纹字段）：能正常读入并判定为"无指纹"，向量丢弃、不崩溃
//   5. 落盘文件里真的写了指纹（不是只改了内存）
//
// 判别力说明：
//   - 把 load() 里的 mismatch 分支去掉（无脑加载 embedding）：第 3/4 段的
//     "向量被丢弃"断言失败（这正是"静默串答案"的复现路径）
//   - 把 set_fingerprint 去掉：第 5 段落盘文件里找不到 fingerprint 字段
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cache/embedding_fingerprint.h"
#include "cache/lru_store.h"
#include "test_check.h"

using namespace ai_gateway;

namespace {

std::string tmp_path(const char* name) {
  return std::string("/tmp/ai_gateway_fp_test_") + name;
}

void write_file(const std::string& path, const std::string& content) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  ofs << content;
}

// 写一份"旧格式"落盘文件（没有 __meta__ 指纹条目）
void write_legacy_file(const std::string& path) {
  nlohmann::json arr = nlohmann::json::array();
  nlohmann::json e1;
  e1["key"] = "msg:1";
  e1["value"] = "legacy-reply";
  e1["src"] = "ns1:question";
  e1["embedding"] = std::vector<float>{1.0f, 2.0f, 3.0f};
  e1["ctime"] = 1700000000;
  arr.push_back(e1);
  nlohmann::json e2;
  e2["key"] = "msg:2";
  e2["value"] = "legacy-text-only";
  e2["src"] = "ns1:q2";
  e2["ctime"] = 1700000001;
  arr.push_back(e2);
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  ofs << arr.dump();
}

}  // namespace

int main() {
  int ok = 0;

  // ── 1. 指纹计算 ─────────────────────────────────────────────────────
  {
    const std::string m1 = tmp_path("model_a.bin");
    const std::string m2 = tmp_path("model_b.bin");
    const std::string v1 = tmp_path("vocab_a.txt");
    const std::string v2 = tmp_path("vocab_b.txt");
    // 内容故意只差最后一个字节（采样式哈希会漏掉这种差异）
    std::string big(100000, 'x');
    write_file(m1, big + "A");
    write_file(m2, big + "B");
    write_file(v1, "[PAD]\n[UNK]\n你\n");
    write_file(v2, "[PAD]\n[UNK]\n我\n");

    auto fp_a = make_embedding_fingerprint(m1, v1, 512, "tok@v1");
    auto fp_b = make_embedding_fingerprint(m2, v1, 512, "tok@v1");  // 只换模型
    auto fp_c = make_embedding_fingerprint(m1, v2, 512, "tok@v1");  // 只换词表
    auto fp_d = make_embedding_fingerprint(m1, v1, 768, "tok@v1");  // 只换维度
    auto fp_e = make_embedding_fingerprint(m1, v1, 512, "tok@v2");  // 只换分词器标识

    CHECK(fp_a.valid());
    ok++;
    CHECK(fp_a.to_string() != fp_b.to_string());  // 模型内容变了
    ok++;
    CHECK(fp_a.to_string() != fp_c.to_string());  // 词表变了
    ok++;
    CHECK(fp_a.to_string() != fp_d.to_string());  // 维度变了
    ok++;
    CHECK(fp_a.to_string() != fp_e.to_string());  // 分词器实现变了
    ok++;
    // 确定性：同样输入两次算出同一个指纹
    auto fp_a2 = make_embedding_fingerprint(m1, v1, 512, "tok@v1");
    CHECK(fp_a.to_string() == fp_a2.to_string());
    ok++;
    // 指纹字符串格式包含四个成分（缺一不可）
    CHECK(fp_a.to_string().find("model=") != std::string::npos);
    ok++;
    CHECK(fp_a.to_string().find("vocab=") != std::string::npos);
    ok++;
    CHECK(fp_a.to_string().find("dim=512") != std::string::npos);
    ok++;
    CHECK(fp_a.to_string().find("tok=tok@v1") != std::string::npos);
    ok++;

    // 文件不存在：指纹不可用（不能"算出空串并匹配上"）
    auto fp_missing =
        make_embedding_fingerprint(tmp_path("no_such_model"), v1, 512, "tok@v1");
    CHECK(!fp_missing.valid());
    ok++;
    CHECK(hash_file_fnv1a64(tmp_path("no_such_model")).empty());
    ok++;
    CHECK(!hash_file_fnv1a64(m1).empty());
    ok++;

    std::remove(m1.c_str());
    std::remove(m2.c_str());
    std::remove(v1.c_str());
    std::remove(v2.c_str());
  }

  // ── 5 + 2. 落盘写指纹；指纹一致时向量原样加载 ────────────────────────
  const std::string fp_file = tmp_path("store.json");
  const std::string fp = "model=aaaa;vocab=bbbb;dim=512;tok=bert@v2";
  {
    std::remove(fp_file.c_str());
    LruStore w(10, 0);
    w.set_fingerprint(fp);
    w.put_with_embedding("msg:1", "reply-1", {1.0f, 2.0f, 3.0f});
    w.set_source("msg:1", "ns1:q1");
    w.put("msg:2", "reply-2");
    w.set_source("msg:2", "ns1:q2");
    CHECK(w.save(fp_file));
    ok++;

    // 文件里确实写了指纹（而不是只在内存里）
    {
      std::ifstream ifs(fp_file);
      auto arr = nlohmann::json::parse(ifs);
      bool found = false;
      for (auto& e : arr)
        if (e.value("key", std::string{}) == "__meta__" &&
            e.value("fp", std::string{}) == fp)
          found = true;
      CHECK(found);
      ok++;
      // 元数据条目不能被当成缓存内容计数
      LruStore probe(10, 0);
      probe.set_fingerprint(fp);
      CHECK(probe.load(fp_file));
      ok++;
      CHECK(probe.size() == 2);
      ok++;
    }

    // 指纹一致：向量保留
    LruStore r(10, 0);
    r.set_fingerprint(fp);
    CHECK(r.load(fp_file));
    ok++;
    CHECK(!r.fingerprint_mismatch());
    ok++;
    CHECK(r.loaded_file_had_fingerprint());
    ok++;
    CHECK(r.loaded_fingerprint() == fp);
    ok++;
    CHECK(r.dropped_vectors() == 0);
    ok++;
    CHECK(r.get_embedding("msg:1").size() == 3);
    ok++;
    CHECK(r.get("msg:1") == "reply-1");
    ok++;
    // 文本与 source 跨加载保留（降级精确匹配仍可用）
    CHECK(r.get_exact("ns1:q1") == "reply-1");
    ok++;
  }

  // ── 3. 指纹不一致：丢弃向量、保留文本 ───────────────────────────────
  {
    LruStore r(10, 0);
    r.set_fingerprint("model=zzzz;vocab=yyyy;dim=512;tok=bert@v2");  // 换了模型
    CHECK(r.load(fp_file));
    ok++;
    CHECK(r.fingerprint_mismatch());
    ok++;
    CHECK(r.dropped_vectors() == 1);  // 只有 msg:1 带向量
    ok++;
    // 关键断言：旧向量绝不能被当成新向量用
    CHECK(r.get_embedding("msg:1").empty());
    ok++;
    // 文本仍然保留：精确匹配（降级路径）照常可用
    CHECK(r.size() == 2);
    ok++;
    CHECK(r.get("msg:1") == "reply-1");
    ok++;
    CHECK(r.get_exact("ns1:q1") == "reply-1");
    ok++;
    // 加载后按新模型重建索引：向量为空 => 索引为空
    size_t with_vec = 0;
    r.for_each_embedding([&](const std::string&, const std::vector<float>&) {
      ++with_vec;
    });
    CHECK(with_vec == 0);
    ok++;
    // 维度变化也走同一条路径
    LruStore r2(10, 0);
    r2.set_fingerprint("model=aaaa;vocab=bbbb;dim=1024;tok=bert@v2");
    CHECK(r2.load(fp_file));
    ok++;
    CHECK(r2.fingerprint_mismatch());
    ok++;
    CHECK(r2.get_embedding("msg:1").empty());
    ok++;
  }

  // ── 4. 旧格式文件（无指纹）：能读、明确标记、向量丢弃 ────────────────
  {
    const std::string legacy = tmp_path("legacy.json");
    write_legacy_file(legacy);

    // 4.1 调用方期望指纹（有模型）：旧文件没有指纹 -> 判定不一致，丢向量留文本
    LruStore r(10, 0);
    r.set_fingerprint(fp);
    CHECK(r.load(legacy));
    ok++;
    CHECK(r.loaded_file_had_fingerprint() == false);
    ok++;
    CHECK(r.fingerprint_mismatch());
    ok++;
    CHECK(r.dropped_vectors() == 1);
    ok++;
    CHECK(r.size() == 2);
    ok++;
    CHECK(r.get("msg:1") == "legacy-reply");
    ok++;
    CHECK(r.get_embedding("msg:1").empty());
    ok++;
    CHECK(r.get_exact("ns1:question") == "legacy-reply");
    ok++;

    // 4.2 调用方没有指纹（无模型降级模式）：不做校验，原样加载
    //     （此时 embed 不可用，向量本来也不会被用来检索）
    LruStore r2(10, 0);
    CHECK(r2.load(legacy));
    ok++;
    CHECK(!r2.fingerprint_mismatch());
    ok++;
    CHECK(r2.get_embedding("msg:1").size() == 3);
    ok++;
    CHECK(r2.loaded_file_had_fingerprint() == false);
    ok++;
    // size() 不受"文件里没有 value 的元数据条目"影响
    CHECK(r2.size() == 2);
    ok++;

    // 4.3 文件损坏：load 失败返回 false（调用方据此告警），不能崩
    const std::string broken = tmp_path("broken.json");
    write_file(broken, "{\"not\":\"an array\"");
    LruStore r3(10, 0);
    r3.set_fingerprint(fp);
    CHECK(!r3.load(broken));
    ok++;

    std::remove(legacy.c_str());
    std::remove(broken.c_str());
  }

  // ── 追加：指纹不一致后重新落盘，文件里应换成新指纹 ───────────────────
  {
    LruStore r(10, 0);
    const std::string new_fp = "model=zzzz;vocab=yyyy;dim=512;tok=bert@v2";
    r.set_fingerprint(new_fp);
    CHECK(r.load(fp_file));
    ok++;
    CHECK(r.dropped_vectors() == 1);
    ok++;
    CHECK(r.save(fp_file));
    ok++;
    {
      std::ifstream ifs(fp_file);
      auto arr = nlohmann::json::parse(ifs);
      std::string stored;
      for (auto& e : arr)
        if (e.value("key", std::string{}) == "__meta__")
          stored = e.value("fp", std::string{});
      CHECK(stored == new_fp);
      ok++;
    }
    // 再用新指纹加载：这次一致，但向量已在上一轮被丢掉（不会被"复活"）
    LruStore r2(10, 0);
    r2.set_fingerprint(new_fp);
    CHECK(r2.load(fp_file));
    ok++;
    CHECK(!r2.fingerprint_mismatch());
    ok++;
    CHECK(r2.get_embedding("msg:1").empty());
    ok++;
    CHECK(r2.get("msg:1") == "reply-1");
    ok++;
    std::remove(fp_file.c_str());
  }

  return test_check::finish("test_embedding_fingerprint", ok);
}
