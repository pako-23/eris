cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(eris
  VERSION 0.0.1
  LANGUAGES CXX
  DESCRIPTION "A federated learning framework implementing the eris algorithm")


option(CMAKE_EXPORT_COMPILE_COMMANDS "Export the compilation commands" OFF)
option(eris_BUILD_TESTS "Build test programs" OFF)
option(eris_ENABLE_COVERAGE "Enable code coverage" OFF)

option(eris_ENABLE_MEMCHECK "Enable memory checks" OFF)

option(eris_ENABLE_TSAN "Enable ThreadSanitizer build" OFF)
option(eris_ENABLE_ASAN "Enable AddressSanitizer build" OFF)
option(eris_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer build" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(no-insource-builds)

add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_17)
target_compile_options(compiler_flags INTERFACE -Wall -Werror)


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(compiler_flags INTERFACE -Wno-stringop-overflow)
endif()


if(eris_ENABLE_MEMCHECK AND (eris_ENABLE_ASAN OR eris_ENABLE_TSAN OR eris_ENABLE_UBSAN))
  message(FATAL_ERROR
    "\n"
    "The options eris_ENABLE_MEMCHECK and eris_ENABLE_ASAN, "
    "eris_ENABLE_TSAN, eris_ENABLE_UBSAN "
    "cannot be enabled at the same time.\n")
elseif(eris_ENABLE_TSAN AND eris_ENABLE_ASAN)
  message(FATAL_ERROR
    "\n"
    "The options eris_ENABLE_TSAN and eris_ENABLE_ASAN "
    "cannot be enabled at the same time.\n")
elseif(eris_ENABLE_TSAN AND (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
  message(WARNING
    "\n"
    "The ThreadSanitizer is supported only for the clang compiler suite.\n"
    "Disabling ThreadSanitizer.\n")
  set(CMAKE_BUILD_TYPE "Debug")
  set(eris_ENABLE_TSAN OFF)
elseif(eris_ENABLE_ASAN AND (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
  message(WARNING
    "\n"
    "The AddressSanitizer is supported only for the clang compiler suite.\n"
    "Disabling AdderessSanitizer.\n")
  set(CMAKE_BUILD_TYPE "Debug")
  set(eris_ENABLE_ASAN OFF)
elseif(eris_ENABLE_UBSAN AND (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
  message(WARNING
    "\n"
    "The UndefinedBehaviorSanitizer is supported only for the clang compiler suite.\n"
    "Disabling UndefinedBehaviorSanitizer.\n")
  set(CMAKE_BUILD_TYPE "Debug")
  set(eris_ENABLE_UBSAN OFF)
endif()


if(eris_ENABLE_TSAN)
  set(CMAKE_BUILD_TYPE "Debug")
  target_compile_options(compiler_flags INTERFACE
    -fsanitize=thread
    -fno-omit-frame-pointer)
  target_link_options(compiler_flags INTERFACE
    -fsanitize=thread
    -fsanitize-address-use-after-scope)
endif()


if(eris_ENABLE_ASAN)
  set(CMAKE_BUILD_TYPE "Debug")
  target_compile_options(compiler_flags INTERFACE
    -fsanitize=address
    -fsanitize-address-use-after-scope
    -fno-omit-frame-pointer)
  target_link_options(compiler_flags INTERFACE
    -fsanitize=address
    -fsanitize-address-use-after-scope)
endif()


if(eris_ENABLE_UBSAN)
  set(CMAKE_BUILD_TYPE "Debug")
  target_compile_options(compiler_flags INTERFACE
    -fsanitize=undefined
    -fsanitize=nullability
    -fsanitize=vptr
    -fno-omit-frame-pointer)
  target_link_options(compiler_flags INTERFACE
    -fsanitize=undefined
    -fsanitize=nullability
    -fsanitize=vptr
    -fno-omit-frame-pointer)
endif()


if(CMAKE_BUILD_TYPE STREQUAL Debug)
  target_compile_options(compiler_flags INTERFACE -g3 -O0)

  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
  set(eris_ENABLE_COVERAGE ON)
  set(eris_BUILD_TESTS ON)
elseif(CMAKE_BUILD_TYPE STREQUAL Release)
  target_compile_options(compiler_flags INTERFACE -O3)
endif()

if(eris_ENABLE_COVERAGE)
  set(eris_BUILD_TESTS ON)
endif()

if(eris_ENABLE_COVERAGE AND (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")))
  set(eris_ENABLE_COVERAGE OFF)
  message(WARNING
    "\n"
    "Coverage is supported only for the GNU compiler suite.\n"
    "Disabling coverage.\n")
endif()


include(coverage)
include(doxygen)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

add_subdirectory(src)

if(eris_BUILD_TESTS)
  include(testing)
  add_subdirectory(test)
endif()

Doxygen(src docs)
