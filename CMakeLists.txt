cmake_minimum_required(VERSION 3.14)

project(CppMachineLearning LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(CML INTERFACE)
add_library(CML::CML ALIAS CML)

target_include_directories(CML INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(CML INTERFACE cxx_std_20)

option(CML_BUILD_TESTS "Build the test suite" ON)
option(CML_BUILD_EXAMPLES "Build the example programs" ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(CML_BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()
    if(CML_BUILD_TESTS)
        add_subdirectory(examples)
    endif()
endif()