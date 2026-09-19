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

# 请求合并：取消/超时/顶替语义（报告 C2 回归）+ 实体一致性否决（第六轮）
# 需要 ai_gateway_cache：singleflight 复用 cache/entity_tokens 的 extract_entity_tokens
# / entity_mismatch（合并路径与缓存命中路径必须同一套实体规则）
add_executable(test_singleflight test_singleflight.cpp)
target_link_libraries(test_singleflight ai_gateway_cache ai_gateway_common
    ai_gateway_test_check)

# HTTP 服务的连接生命周期（报告 H5 半关闭/空闲超时/连接上限、M12 drain）
add_executable(test_http_server test_http_server.cpp)
target_link_libraries(test_http_server ai_gateway_server ai_gateway_common
    ai_gateway_test_check)
target_compile_definitions(test_http_server PRIVATE GATEWAY_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# SSE 流式透传（本轮新增）：事件边界过滤 + chunked 分帧 + 端到端逐块到达
add_executable(test_sse_stream test_sse_stream.cpp)
target_link_libraries(test_sse_stream ai_gateway_server ai_gateway_common
    ai_gateway_test_check)

# keep-alive：同一连接多请求 / Connection: close 遵守 / 端口复用观测
add_executable(test_keep_alive test_keep_alive.cpp)
target_link_libraries(test_keep_alive ai_gateway_server ai_gateway_common
    ai_gateway_test_check)

# embedding 指纹版本化：指纹计算 + 落盘/加载比对 + 旧格式向后兼容
add_executable(test_embedding_fingerprint test_embedding_fingerprint.cpp)
target_link_libraries(test_embedding_fingerprint ai_gateway_cache
    ai_gateway_common nlohmann_json::nlohmann_json ai_gateway_test_check)

# Module 4: stats
add_executable(test_stats test_stats.cpp)
target_link_libraries(test_stats ai_gateway_stats ai_gateway_common ai_gateway_test_check)

# 实体一致性否决（本轮新增）：数字/大写缩略语/混合标识符的提取与不对称判定。
# 依赖实测反例（继续下一题 ↔ 继续12题、什么是DMA ↔ 什么是DNS）
add_executable(test_entity_tokens test_entity_tokens.cpp)
target_link_libraries(test_entity_tokens ai_gateway_cache ai_gateway_common
    ai_gateway_test_check)

# SSE usage 解析（本轮新增）：流式 token 统计，任意 chunk 切分 + 有界性 + 不误判
add_executable(test_sse_usage test_sse_usage.cpp)
target_link_libraries(test_sse_usage ai_gateway_server ai_gateway_common
    ai_gateway_test_check)

# 流式写死线语义（本轮新增）：流式用空闲死线（长流不被切断）、缓冲式保留总死线
add_executable(test_stream_timeout test_stream_timeout.cpp)
target_link_libraries(test_stream_timeout ai_gateway_server ai_gateway_common
    ai_gateway_test_check)

# 原始 SSE 字节 → 回答文本（流式回填缓存的解析路径）：null content / reasoning /
# [DONE] / CRLF / 多行 data / 坏事件 / 断流 / 非流式形态
add_executable(test_sse_capture test_sse_capture.cpp)
target_link_libraries(test_sse_capture ai_gateway_server ai_gateway_common
    ai_gateway_test_check)

# 日志轮转（报告 8.7 第 4 条）：切分/保留份数/行边界/关闭开关
add_executable(test_log_rotation test_log_rotation.cpp)
target_link_libraries(test_log_rotation ai_gateway_common ai_gateway_test_check)

# /metrics 端点（报告 8.7 第 4 条）：Prometheus 文本 + GET 路由不破坏 POST
add_executable(test_metrics test_metrics.cpp)
target_link_libraries(test_metrics ai_gateway_server ai_gateway_stats
    ai_gateway_common ai_gateway_test_check)

# 分词器保真度（报告 8.7 第 5 条）：与 HuggingFace BERT WordPiece 的金标准 id 对照
add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries(test_tokenizer ai_gateway_cache ai_gateway_common
    ai_gateway_test_check)
target_compile_definitions(test_tokenizer PRIVATE GATEWAY_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# 生产请求管道（对抗性审查 T2）：直接链接 src/gateway/pipeline.cpp —— 与
# ai-gateway 可执行文件同一份对象代码，覆盖 H2（合并 ns 隔离）/M1（SSE 中止归因）/
# M3（keep_alive 决策真的作用到连接）三处此前零覆盖的缺陷
add_executable(test_pipeline test_pipeline.cpp)
target_link_libraries(test_pipeline ai_gateway_pipeline ai_gateway_test_check)

add_test(NAME filter COMMAND test_filter)
add_test(NAME sse_capture COMMAND test_sse_capture)
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
add_test(NAME metrics COMMAND test_metrics)
add_test(NAME tokenizer COMMAND test_tokenizer)
add_test(NAME sse_stream COMMAND test_sse_stream)
add_test(NAME keep_alive COMMAND test_keep_alive)
add_test(NAME embedding_fingerprint COMMAND test_embedding_fingerprint)
add_test(NAME entity_tokens COMMAND test_entity_tokens)
add_test(NAME sse_usage COMMAND test_sse_usage)
add_test(NAME stream_timeout COMMAND test_stream_timeout)
add_test(NAME pipeline COMMAND test_pipeline)

# 端到端集成验证（本轮新增）：真实 ai-gateway 二进制 + 真实 TCP + 真实 SIGTERM。
# 覆盖单测覆盖不到的部分：H1（main 是否真的接线 drain，要发 SIGTERM 才看得出来）、
# H2（真 curl 并发的跨 system 合并）、M3（真连接上的 Connection/复用行为）、
# M1（真实 SSE 静默流的日志归因）。脚本自己挑随机端口、自己清理进程（只按 PID）。
add_test(NAME integration_pipeline
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/integration_pipeline_test.sh
            $<TARGET_FILE:ai-gateway>)

# HNSW 自检索召回率断言（合成向量 + 固定种子，确定性）
add_test(NAME hnsw_recall
    COMMAND recall_bench --synthetic 800 --dim 512 --queries 200 --check)

# test_config 依赖相对路径 config/gateway.example.json：把工作目录钉在源码根目录，
# 否则从构建目录运行时会加载失败（Debug 下曾直接断言 abort）
set_tests_properties(config PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
# test_tokenizer 需要 model/vocab.txt（相对源码根目录）
set_tests_properties(tokenizer PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(hnsw_recall PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
