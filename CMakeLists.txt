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

option(BUILD_TESTS "Build the test suite (ctest targets)" ON)
# 必须写在顶层：enable_testing() 只在子目录里调用时，构建根目录的 CTestTestfile.cmake
# 不会生成，在根目录跑 ctest 会报 "No tests were found!!!"（报告 L1）
if(BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(src)

if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
