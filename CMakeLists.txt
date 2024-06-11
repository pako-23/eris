cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(eris VERSION 0.0.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(no-insource-builds)

include(FetchContent)
  
set(FETCHCONTENT_QUIET OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

include(testing)
add_subdirectory(test)
