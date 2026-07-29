// VectorIndex 单元测试
#include <cassert>
#include <iostream>
#include "cache/vector_index.h"
using namespace ai_gateway;

int main() {
    VectorIndex idx;
    int ok = 0;

    idx.add(1, "a", {1, 0, 0});
    idx.add(2, "b", {0, 1, 0});
    idx.add(3, "c", {0, 0, 1});

    auto r = idx.search({1, 0.1f, 0}, 2);
    assert(r.size() == 2); ok++;
    assert(r[0].key == "a"); ok++;
    assert(r[0].similarity > r[1].similarity); ok++;

    idx.remove(1);
    auto r2 = idx.search({1, 0, 0}, 1);
    assert(r2[0].key != "a"); ok++;

    std::cout << "test_vector_index: " << ok << "/4 passed\n";
    return 0;
}
