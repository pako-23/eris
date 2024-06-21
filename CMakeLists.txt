cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(eris
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "A federated learning framework implementing the eris algorithm")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CMAKE_EXPORT_COMPILE_COMMANDS "Export the compilation commands" OFF)
option(ERIS_BUILD_TESTS "Build test programs" OFF)
option(ERIS_ENABLE_COVERAGE "Enable code coverage" OFF)
option(ERIS_ENABLE_MEMCHECK "Enable memory checks" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(no-insource-builds)

add_compile_options(-Wall)
add_compile_options(-Werror)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
  add_compile_options(-g3)

  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(ERIS_BUILD_TESTS ON)
  set(ERIS_ENABLE_COVERAGE ON)
  set(ERIS_ENABLE_MEMCHECK ON)
endif()

set(ERIS_BUILD_TESTS $<OR:${ERIS_BUILD_TESTS},${ERIS_ENABLE_COVERAGE},${ERIS_ENABLE_MEMCHECK}>)

include(coverage)
include(doxygen)

include(FetchContent)

add_subdirectory(src)

if (ERIS_BUILD_TESTS)
  include(testing)
  add_subdirectory(test)
endif()

Doxygen(src docs)
