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

    std::cout << "test_lru_store: " << ok << "/5 passed\n";
    return 0;
}
