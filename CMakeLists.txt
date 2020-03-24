
cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(quant_conv)

set(ENV{HTTP_PROXY} "http://proxy.cse.cuhk.edu.hk:8000")
SET(ENV{HTTPS_PROXY} "http://proxy.cse.cuhk.edu.hk:8000")
set(CMAKE_CXX_STANDARD 11)


ENABLE_TESTING()
# Build Google Test.

MESSAGE(STATUS "Start to build GoogleTest")
CONFIGURE_FILE(cmake/DownloadGoogleTest.cmake "${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt")
EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download")
SET(GOOGLETEST_SOURCE_DIR "${CMAKE_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")

# Build Google Benchmark.
MESSAGE(STATUS "Start to build GoogleBenchmark")
CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_SOURCE_DIR}/googlebenchmark" CACHE STRING "Google Benchmark source directory")

SET(FLYCONV_SRCS
        detect_arm.cpp
        convolution_quant.cpp)


# --- [build google test
IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    ADD_SUBDIRECTORY(
            "${GOOGLETEST_SOURCE_DIR}"
            "${CMAKE_BINARY_DIR}/googletest")
ENDIF()

IF(NOT TARGET benchmark)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
            "${GOOGLEBENCHMARK_SOURCE_DIR}"
            "${CMAKE_BINARY_DIR}/googlebenchmark")
ENDIF()

add_library(flyconv SHARED ${FLYCONV_SRCS})

add_executable(bench-conv benchmark.cpp convolution_quant.cpp detect_arm.cpp)
SET_TARGET_PROPERTIES(bench-conv PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO)
target_link_libraries(bench-conv PRIVATE benchmark)


add_executable(test-conv convolution-test.cpp)

SET_TARGET_PROPERTIES(test-conv PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO)

target_link_libraries(test-conv PRIVATE gtest gtest_main)
add_test(test-conv test-conv)