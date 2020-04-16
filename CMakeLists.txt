
CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)

project(quant_conv C CXX ASM)

#set(ENV{HTTP_PROXY} "http://proxy.cse.cuhk.edu.hk:8000")
#SET(ENV{HTTPS_PROXY} "http://proxy.cse.cuhk.edu.hk:8000")
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "-fopenmp")


ENABLE_TESTING()
# Build Google Test.

#MESSAGE(STATUS "start to build pthreadpool")
# copy the Download* file to the binary directory.
#CONFIGURE_FILE(cmake/DownloadPThreadPool.cmake "${CMAKE_BINARY_DIR}/pthreadpool-download/CMakeLists.txt")
# Download the pthreadpool.git to pthreadpool-download.
#EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
#        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")
# Build the pthreadpool
#EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
#        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/pthreadpool-download")

#SET(PTHREADPOOL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/pthreadpool" CACHE STRING "PthreadPoool Source directory")


MESSAGE(STATUS  "Start to build GoogleTest")
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

SET(QUANTCONV_SRCS
        convolution.cpp indirection.cpp
        convolution_quant.cpp
        kernel_aarch64_neon.cpp
        kernel_pack.cpp
        block_map.cpp
        pair.cpp
        pack_kernel.cpp
        pack.cpp)

# --- [build thread pool]

#IF(NOT TARGET pthreadpool)
#    ADD_SUBDIRECTORY("${PTHREADPOOL_SOURCE_DIR}"
#             "${CMAKE_BINARY_DIR}/pthreadpool")
#ENDIF()
# --- [build google test
IF(NOT TARGET gtest)
    SET(gtest_force_shared_crt ON CACHE BOOL  "" FORCE)
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

add_library(quantconv SHARED ${QUANTCONV_SRCS})

add_executable(bench-conv benchmark.cpp)
SET_TARGET_PROPERTIES(bench-conv PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO)
target_link_libraries(bench-conv PRIVATE benchmark quantconv)


add_executable(test-conv convolution-test.cpp)

SET_TARGET_PROPERTIES(test-conv PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO)

target_link_libraries(test-conv PRIVATE gtest gtest_main quantconv)
add_test(test-conv test-conv)