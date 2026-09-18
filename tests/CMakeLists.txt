cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(nlohmann_json REQUIRED)

# 断言工具 test_check.h 的 CHECK 宏不依赖 NDEBUG：
# Release 构建定义了 NDEBUG，用 <cassert> 的 assert 会让断言被整体剔除，
# 测试于是"全绿"但一条断言都没执行（报告 H7）
add_library(ai_gateway_test_check INTERFACE)
target_include_directories(ai_gateway_test_check INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Module 1: server
add_executable(test_filter test_filter.cpp)
target_link_libraries(test_filter ai_gateway_server ai_gateway_common ai_gateway_test_check)

add_executable(test_request test_request.cpp)
target_link_libraries(test_request ai_gateway_server ai_gateway_common ai_gateway_test_check)

add_executable(test_response test_response.cpp)
target_link_libraries(test_response ai_gateway_server ai_gateway_common ai_gateway_test_check)

add_executable(test_router test_router.cpp)
target_link_libraries(test_router ai_gateway_server ai_gateway_common ai_gateway_test_check)

# Module 2: cache
add_executable(test_lru_store test_lru_store.cpp)
target_link_libraries(test_lru_store ai_gateway_cache ai_gateway_common ai_gateway_test_check)

# 缓存协调器：purge/rebuild 后的键一致性与并发检索（报告 C3/M8/C1 回归）
add_executable(test_cache_engine test_cache_engine.cpp)
target_link_libraries(test_cache_engine ai_gateway_cache ai_gateway_common
    nlohmann_json::nlohmann_json ai_gateway_test_check)

# HNSW 召回率基准：默认手动运行；--check 时把关键指标变成断言（见 ctest 的 hnsw_recall）
add_executable(recall_bench recall_bench.cpp)
target_link_libraries(recall_bench ai_gateway_cache ai_gateway_common
    nlohmann_json::nlohmann_json ai_gateway_test_check)

# Common
add_executable(test_config test_config.cpp)
target_link_libraries(test_config ai_gateway_common nlohmann_json::nlohmann_json
    ai_gateway_test_check)
# 注入源码根目录：测试可在任意工作目录下运行（相对路径失效时回退）
target_compile_definitions(test_config PRIVATE
    GATEWAY_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# 请求合并：取消/超时/顶替语义（报告 C2 回归）
add_executable(test_singleflight test_singleflight.cpp)
target_link_libraries(test_singleflight ai_gateway_common ai_gateway_test_check)

# HTTP 服务的连接生命周期（报告 H5 半关闭/空闲超时/连接上限、M12 drain）
add_executable(test_http_server test_http_server.cpp)
target_link_libraries(test_http_server ai_gateway_server ai_gateway_common
    ai_gateway_test_check)
target_compile_definitions(test_http_server PRIVATE GATEWAY_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# Module 4: stats
add_executable(test_stats test_stats.cpp)
target_link_libraries(test_stats ai_gateway_stats ai_gateway_common ai_gateway_test_check)

# 日志轮转（报告 8.7 第 4 条）：切分/保留份数/行边界/关闭开关
add_executable(test_log_rotation test_log_rotation.cpp)
target_link_libraries(test_log_rotation ai_gateway_common ai_gateway_test_check)

add_test(NAME filter COMMAND test_filter)
add_test(NAME request COMMAND test_request)
add_test(NAME response COMMAND test_response)
add_test(NAME router COMMAND test_router)
add_test(NAME lru_store COMMAND test_lru_store)
add_test(NAME stats COMMAND test_stats)
add_test(NAME config COMMAND test_config)
add_test(NAME singleflight COMMAND test_singleflight)
add_test(NAME cache_engine COMMAND test_cache_engine)
add_test(NAME http_server COMMAND test_http_server)
add_test(NAME log_rotation COMMAND test_log_rotation)

# HNSW 自检索召回率断言（合成向量 + 固定种子，确定性）
add_test(NAME hnsw_recall
    COMMAND recall_bench --synthetic 800 --dim 512 --queries 200 --check)

# test_config 依赖相对路径 config/gateway.example.json：把工作目录钉在源码根目录，
# 否则从构建目录运行时会加载失败（Debug 下曾直接断言 abort）
set_tests_properties(config PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(hnsw_recall PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
