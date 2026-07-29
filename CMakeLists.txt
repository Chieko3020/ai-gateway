cmake_minimum_required(VERSION 3.16)
project(ai-gateway VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 确保 find_package 能找到 nlohmann_json
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

# Enable AVX2 if available (for vector_index SIMD dot product)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
  add_compile_options(-mavx2 -mfma)
  message(STATUS "AVX2+FMA enabled for SIMD dot product")
endif()

add_subdirectory(src)
add_subdirectory(tests EXCLUDE_FROM_ALL)
