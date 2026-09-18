// LRU Store 单元测试
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include "cache/lru_store.h"
using namespace ai_gateway;
using namespace std::chrono_literals;

int main() {
    LruStore s(3, 1);  // 3条, 1s TTL
    int ok = 0;

    s.put("a", "1"); s.put("b", "2"); s.put("c", "3"); s.put("d", "4");
    assert(!s.get("a").has_value()); ok++;  // a evicted
    assert(s.get("d") == "4"); ok++;

    s.put("x", "val");
    assert(s.get("x") == "val");
    // 忙等待 TTL 过期，避免固定 sleep 在负载高时不够
    auto deadline = std::chrono::steady_clock::now() + 5s;
    while (s.get("x").has_value()) {
        if (std::chrono::steady_clock::now() > deadline) break;
        std::this_thread::sleep_for(100ms);
    }
    assert(!s.get("x").has_value()); ok++;  // TTL expired

    assert(s.hit_count() >= 1); ok++;
    assert(s.evict_count() >= 1); ok++;

    // purge_expired：主动清理过期条目，且未过期时不误删
    {
        LruStore p(10, 1);  // 10条, 1s TTL
        p.put("k1", "v1");
        p.put("k2", "v2");
        assert(p.size() == 2);
        assert(p.purge_expired() == 0); ok++;  // 未过期：不清理

        auto dl = std::chrono::steady_clock::now() + 5s;
        while (std::chrono::steady_clock::now() < dl && p.purge_expired() == 0) {
            std::this_thread::sleep_for(200ms);
        }
        assert(p.size() == 0); ok++;             // 过期后全部清理
        assert(p.expired_count() >= 2); ok++;    // 计入过期统计
    }

    // purge_expired：ttl=0（永不过期）时不清任何条目
    {
        LruStore q(10, 0);
        q.put("k", "v");
        assert(q.purge_expired() == 0); ok++;
        assert(q.size() == 1); ok++;
    }

    std::cout << "test_lru_store: " << ok << "/10 passed\n";
    return 0;
}
