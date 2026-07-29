cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(nlohmann_json REQUIRED)

# Module 1: server
add_executable(test_filter test_filter.cpp)
target_link_libraries(test_filter ai_gateway_server ai_gateway_common)

add_executable(test_request test_request.cpp)
target_link_libraries(test_request ai_gateway_server ai_gateway_common)

add_executable(test_response test_response.cpp)
target_link_libraries(test_response ai_gateway_server ai_gateway_common)

add_executable(test_router test_router.cpp)
target_link_libraries(test_router ai_gateway_server ai_gateway_common)

# Module 2: cache
add_executable(test_lru_store test_lru_store.cpp)
target_link_libraries(test_lru_store ai_gateway_cache ai_gateway_common)

add_executable(test_vector_index test_vector_index.cpp)
target_link_libraries(test_vector_index ai_gateway_cache ai_gateway_common)

# Module 4: stats
add_executable(test_stats test_stats.cpp)
target_link_libraries(test_stats ai_gateway_stats ai_gateway_common)

# Common
add_executable(test_config test_config.cpp)
target_link_libraries(test_config ai_gateway_common nlohmann_json::nlohmann_json)

enable_testing()
add_test(NAME filter COMMAND test_filter)
add_test(NAME request COMMAND test_request)
add_test(NAME response COMMAND test_response)
add_test(NAME router COMMAND test_router)
add_test(NAME lru_store COMMAND test_lru_store)
add_test(NAME vector_index COMMAND test_vector_index)
add_test(NAME stats COMMAND test_stats)
add_test(NAME config COMMAND test_config)
