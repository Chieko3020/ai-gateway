// LRU Store 单元测试
#include "test_check.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <thread>
#include "cache/lru_store.h"
using namespace ai_gateway;
using namespace std::chrono_literals;

int main() {
    LruStore s(3, 1);  // 3条, 1s TTL
    int ok = 0;

    s.put("a", "1"); s.put("b", "2"); s.put("c", "3"); s.put("d", "4");
    CHECK(!s.get("a").has_value()); ok++;  // a evicted
    CHECK(s.get("d") == "4"); ok++;

    s.put("x", "val");
    CHECK(s.get("x") == "val");
    // 忙等待 TTL 过期，避免固定 sleep 在负载高时不够
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (s.get("x").has_value()) {
        if (std::chrono::steady_clock::now() > deadline) break;
        std::this_thread::sleep_for(100ms);
    }
    CHECK(!s.get("x").has_value()); ok++;  // TTL expired

    CHECK(s.hit_count() >= 1); ok++;
    CHECK(s.evict_count() >= 1); ok++;

    // purge_expired：主动清理过期条目，且未过期时不误删
    {
        LruStore p(10, 1);  // 10条, 1s TTL
        p.put("k1", "v1");
        p.put("k2", "v2");
        CHECK(p.size() == 2);
        CHECK(p.purge_expired() == 0); ok++;  // 未过期：不清理

        auto dl = std::chrono::steady_clock::now() + 5s;
        while (std::chrono::steady_clock::now() < dl && p.purge_expired() == 0) {
            std::this_thread::sleep_for(200ms);
        }
        CHECK(p.size() == 0); ok++;             // 过期后全部清理
        CHECK(p.expired_count() >= 2); ok++;    // 计入过期统计
    }

    // purge_expired：ttl=0（永不过期）时不清任何条目
    {
        LruStore q(10, 0);
        q.put("k", "v");
        CHECK(q.purge_expired() == 0); ok++;
        CHECK(q.size() == 1); ok++;
    }

    // save/load：原子替换 + src 字段 + 向后兼容旧文件（报告 M2/M4/M9）
    {
        const std::string path = "/tmp/ai_gateway_lru_test.json";
        std::remove(path.c_str());

        LruStore w(10, 0);
        w.put_with_embedding("msg:1", "reply-1", {1.0f, 2.0f, 3.0f});
        w.set_source("msg:1", "ns1:q1");
        w.put("msg:2", "reply-2");
        w.set_source("msg:2", "q2");
        CHECK(w.save(path)); ok++;

        // 原子写：目标文件存在、临时文件已消失
        CHECK(std::ifstream(path).good()); ok++;
        CHECK(!std::ifstream(path + ".tmp").good()); ok++;

        LruStore r(10, 0);
        CHECK(r.load(path)); ok++;
        CHECK(r.size() == 2); ok++;
        CHECK(r.get("msg:1") == "reply-1"); ok++;
        CHECK(r.get_embedding("msg:1").size() == 3); ok++;
        // src 字段跨进程保留：降级精确匹配在重启后仍然可用
        CHECK(r.get_exact("ns1:q1") == "reply-1"); ok++;
        CHECK(r.get_exact("q2") == "reply-2"); ok++;
        CHECK(!r.get_exact("q-unknown").has_value()); ok++;

        // 原始 SSE 字节（put_full）跨进程保留：流式命中要靠它回放。
        // 判别力：save/load 里漏掉 sse 字段 → 重启后流式命中退化为"无 SSE 载荷"，
        // 只能回源（功能静默失效，不留任何错误）
        {
            const std::string spath = path + ".sse";
            LruStore sw(10, 0);
            const std::string sse =
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
                "data: [DONE]\n\n";
            sw.put_full("msg:9", "hi", sse, {0.5f, 0.5f});
            sw.set_source("msg:9", "q9");
            CHECK(sw.save(spath)); ok++;

            LruStore sr(10, 0);
            CHECK(sr.load(spath)); ok++;
            CHECK(sr.get("msg:9") == "hi"); ok++;
            auto back = sr.sse_of("msg:9");
            CHECK(back.has_value()); ok++;
            CHECK(back.value() == sse); ok++;  // 逐字节相同（回放必须原样）
            CHECK(sr.get_embedding("msg:9").size() == 2); ok++;

            // 普通条目（没有 SSE 字节）读出来是 nullopt，而不是空串
            CHECK(!sr.sse_of("msg:1").has_value()); ok++;
            // 不存在的键同样是 nullopt
            CHECK(!sr.sse_of("msg:404").has_value()); ok++;
            // sse_of 是只读：**不改变命中计数**（命中判定已由 try_hit 记过账，
            // 这里再记一次会把一次命中记成两次）
            const size_t hits_before = sr.hit_count();
            CHECK(!sr.sse_of("msg:9").has_value() == false); ok++;
            CHECK(sr.hit_count() == hits_before); ok++;
            std::remove(spath.c_str());
        }

        // 旧落盘文件（没有 src 字段）必须仍能读进来
        {
            const std::string legacy = path + ".legacy";
            std::ofstream ofs(legacy);
            ofs << R"([{"key":"msg:7","value":"legacy-reply","ctime":)"
                << (std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count())
                << R"(}])";
            ofs.close();
            LruStore lr(10, 0);
            CHECK(lr.load(legacy)); ok++;
            CHECK(lr.get("msg:7") == "legacy-reply"); ok++;  // 缺 src 不影响基本读
            CHECK(lr.get_exact("msg:7") == "legacy-reply"); ok++;
            std::remove(legacy.c_str());
        }

        // 损坏文件：load 返回 false（而不是静默当成空缓存）
        {
            const std::string broken = path + ".broken";
            std::ofstream ofs(broken);
            ofs << "[{\"key\":\"msg:8\"";
            ofs.close();
            LruStore br(10, 0);
            CHECK(!br.load(broken)); ok++;
            std::remove(broken.c_str());
        }

        // 父目录不存在时主动补建（否则调用方以为已落盘，实际一份都没写）
        {
            const std::string nested = "/tmp/ai-gateway-nested-dir/x.json";
            std::remove(nested.c_str());
            std::remove("/tmp/ai-gateway-nested-dir/.keep");
            LruStore f(4, 0);
            f.put("k", "v");
            CHECK(f.save(nested)); ok++;
            CHECK(std::ifstream(nested).good()); ok++;
            std::remove(nested.c_str());
            std::remove("/tmp/ai-gateway-nested-dir");
        }
        std::remove(path.c_str());
    }

    // load 必须按 max_entries 裁剪（报告 M2）：
    //  旧实现逐条 push_back、全程不看 max_entries_，而插入路径是"淘汰 1 条 + 插 1 条"
    //  ⇒ size() 恒定不变，超限状态**永不收敛**（实测 max=5 时装入 20 条、再写 3 条
    //  仍是 20 条）。这是 MemoryMax=150M 那个 OOM 风险点的放大器：配置层以为限住了，
    //  实际没有，超限条目还会被 for_each_embedding 全部建进 HNSW 索引。
    //
    // 判别力：去掉 load() 末尾的裁剪循环（或把 while 改回 if），下面
    // "(2) 裁剪到上限" 与 "(3) 插入后收敛" 两组断言失败。
    //
    // 这里**不**断言"留下的具体是哪几条"：save() 的顺序是 [MRU…LRU]、这里逐条
    // push_back，因此内存链表天然也是 [MRU…LRU]（头部最近使用、尾部最久未用），
    // 裁剪必然丢尾部。"留下的就是最近使用的那些"由 LruStore 的顺序约定保证，
    // 不是这条用例的断言点（本轮曾在 load 里加过一行 reverse() 并声称"修正倒置"，
    // 那会让裁剪丢最新条目；该错误由下面 (2) 的 size 断言之外的一条
    // "最近访问过的条目仍在"断言当场抓到）
    {
        const std::string path = "/tmp/ai-gateway-lru-trim.json";
        std::remove(path.c_str());
        {
            LruStore w(100, 0);
            for (int i = 0; i < 20; ++i)
                w.put("msg:" + std::to_string(i), "v" + std::to_string(i));
            CHECK(w.size() == 20); ok++;
            // 让 msg:19 成为"最近使用"（后面断言它必须活过裁剪）
            CHECK(w.get("msg:19").has_value()); ok++;
            CHECK(w.save(path)); ok++;
        }
        // (1) 上限足够大：整份文件原样加载（不得丢条目）
        {
            LruStore r(100, 0);
            CHECK(r.load(path)); ok++;
            CHECK(r.size() == 20); ok++;
        }
        // (2) 上限小于文件条目数：必须裁剪到上限，且最近使用的那条必须在
        {
            LruStore r(5, 0);
            CHECK(r.load(path)); ok++;
            CHECK(r.size() == 5); ok++;              // ★ 不裁剪时这里是 20
            CHECK(r.get("msg:19").has_value()); ok++;  // 最近使用 → 保留
        }
        // (3) 插入路径防御：一次写入就把超限压回上限
        //     （旧实现"淘汰 1 条 + 插 1 条"，超限状态下 size 恒定不变）
        {
            LruStore r(5, 0);
            CHECK(r.load(path)); ok++;
            for (int i = 100; i < 105; ++i) r.put("new:" + std::to_string(i), "v");
            CHECK(r.size() <= 5); ok++;
            CHECK(r.get("new:104").has_value()); ok++;
        }
        // (4) 上限为 0 = 不限制：不得误裁
        {
            LruStore r(0, 0);
            CHECK(r.load(path)); ok++;
            CHECK(r.size() == 20); ok++;
        }
        std::remove(path.c_str());
    }

    return test_check::finish("test_lru_store", ok);
}
