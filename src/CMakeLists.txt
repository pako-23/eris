add_library(erisfl INTERFACE)

target_include_directories(erisfl INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(erisfl INTERFACE compiler_flags)

FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG        v4.3.5)
set(BUILD_TESTS OFF CACHE INTERNAL "")
set(BUILD_SHARED OFF CACHE INTERNAL "")
set(BUILD_STATIC ON CACHE INTERNAL "")
FetchContent_MakeAvailable(libzmq)

add_subdirectory(proto)
add_subdirectory(util)
add_subdirectory(algorithms)
add_subdirectory(pybindings)
