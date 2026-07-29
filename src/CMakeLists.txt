# Module libraries

# Common (logger, config, types)
add_library(ai_gateway_common STATIC
    common/config.cpp
)
target_include_directories(ai_gateway_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ai_gateway_common PUBLIC nlohmann_json::nlohmann_json)

# Server (http_server, request, response, filter)
add_library(ai_gateway_server STATIC
    server/http_server.cpp
    server/request.cpp
    server/response.cpp
    server/filter.cpp
    server/router.cpp
    server/connection_handler.cpp
)
target_include_directories(ai_gateway_server PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ai_gateway_server PUBLIC ai_gateway_common)

# Backend (llm_client)
add_library(ai_gateway_backend STATIC
    backend/llm_client.cpp
)
target_include_directories(ai_gateway_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ai_gateway_backend PUBLIC ai_gateway_common CURL::libcurl)

# Cache (lru_store, embedding, vector_index, cache_engine)
add_library(ai_gateway_cache STATIC
    cache/lru_store.cpp
    cache/embedding.cpp
    cache/vector_index.cpp
    cache/cache_engine.cpp
)
target_include_directories(ai_gateway_cache PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ai_gateway_cache PUBLIC ai_gateway_common CURL::libcurl nlohmann_json::nlohmann_json)

# Stats
add_library(ai_gateway_stats STATIC
    stats/stats.cpp
)
target_include_directories(ai_gateway_stats PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
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
