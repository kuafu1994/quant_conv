
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12 FATAL_ERROR)

PROJECT(googlebenchmark-download NONE)

INCLUDE(ExternalProject)
ExternalProject_Add(googlebenchmark
        URL https://github.com/google/benchmark/archive/v1.4.1.zip
        URL_HASH SHA256=61ae07eb5d4a0b02753419eb17a82b7d322786bb36ab62bd3df331a4d47c00a7
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/googlebenchmark"
        BINARY_DIR "${CMAKE_BINARY_DIR}/googlebenchmark"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        TEST_COMMAND ""
        )

