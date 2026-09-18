# Module libraries — include/ai-gateway/ 方案
set(INCLUDE_ROOT ${CMAKE_SOURCE_DIR}/include)

# Common (logger, config, types, curl_client, log_file)
add_library(ai_gateway_common STATIC
    common/config.cpp
    common/log_file.cpp
)
target_include_directories(ai_gateway_common PUBLIC ${INCLUDE_ROOT})
target_link_libraries(ai_gateway_common PUBLIC nlohmann_json::nlohmann_json)

# Server (http_server, request, response, filter, router, connection_handler, metrics)
add_library(ai_gateway_server STATIC
    server/http_server.cpp
    server/request.cpp
    server/response.cpp
    server/filter.cpp
    server/router.cpp
    server/connection_handler.cpp
    server/metrics.cpp
    server/sse_usage.cpp
)
target_include_directories(ai_gateway_server PUBLIC ${INCLUDE_ROOT})
# metrics.cpp 渲染 Stats 快照（ai_gateway_stats 在下方定义，CMake 在生成期解析）
target_link_libraries(ai_gateway_server PUBLIC ai_gateway_common ai_gateway_stats)

# Backend (llm_client)
add_library(ai_gateway_backend STATIC
    backend/llm_client.cpp
)
target_include_directories(ai_gateway_backend PUBLIC ${INCLUDE_ROOT})
target_link_libraries(ai_gateway_backend PUBLIC ai_gateway_common CURL::libcurl)

# Cache (lru_store, cache_engine, onnx_embedding, entity_tokens)
add_library(ai_gateway_cache STATIC
    cache/lru_store.cpp
    cache/cache_engine.cpp
    cache/onnx_embedding.cpp
    cache/embedding_fingerprint.cpp
    cache/entity_tokens.cpp
)
target_include_directories(ai_gateway_cache PUBLIC ${INCLUDE_ROOT})
target_link_libraries(ai_gateway_cache PUBLIC ai_gateway_common CURL::libcurl nlohmann_json::nlohmann_json
    onnxruntime)

# Stats
add_library(ai_gateway_stats STATIC
    stats/stats.cpp
)
target_include_directories(ai_gateway_stats PUBLIC ${INCLUDE_ROOT})
target_link_libraries(ai_gateway_stats PUBLIC ai_gateway_common)

# Main executable
add_executable(ai-gateway main.cpp)
target_link_libraries(ai-gateway PRIVATE
    ai_gateway_common
    ai_gateway_server
    ai_gateway_backend
    ai_gateway_cache
    ai_gateway_stats
)
