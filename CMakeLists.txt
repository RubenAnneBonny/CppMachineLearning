cmake_minimum_required(VERSION 3.14)

project(CppMachineLearning LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

add_library(CML INTERFACE)
add_library(CML::CML ALIAS CML)

target_include_directories(CML INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(CML INTERFACE cxx_std_20)

add_library(cml_warnings INTERFACE)
target_compile_options(cml_warnings INTERFACE
    $<$<CXX_COMPILER_ID:MSVC>:/W4;/WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wconversion;-Wsign-conversion;-Werror>
)

option(CML_BUILD_TESTS "Build the test suite" ON)
option(CML_BUILD_EXAMPLES "Build the example programs" ON)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(CML_BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()
    if(CML_BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()

    include(CMakePackageConfigHelpers)

    install(TARGETS CML EXPORT CMLTargets)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(EXPORT CMLTargets
            FILE CMLTargets.cmake
            NAMESPACE CML::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CML)
    
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CMLConfig.cmake.in
        ${CMAKE_BINARY_DIR}/CMLConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CML)
    
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMLConfig.cmake
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CML)
endif()