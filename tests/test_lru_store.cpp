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

    return test_check::finish("test_lru_store", ok);
}
