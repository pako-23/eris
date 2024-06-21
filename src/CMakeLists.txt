add_library(erisfl INTERFACE)

target_include_directories(erisfl INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(util)

add_subdirectory(proto)

add_subdirectory(algorithms)
