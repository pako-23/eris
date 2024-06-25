include(logging)
include(pybind11)

find_package(gRPC REQUIRED)

 FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG        v4.3.5)
set(BUILD_TESTS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(libzmq)

add_library(algorithms_obj OBJECT
  eris/builder.cc
  eris/coordinator.cc
  eris/aggregator.cc)
target_include_directories(algorithms_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  ${LOGGING_INCLUDE_DIR}
  ${Python_INCLUDE_DIRS}
  ${libzmq_SOURCE_DIR}/include
  ${pybind11_INCLUDE_DIR})
set_target_properties(algorithms_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(algorithms_obj PUBLIC
  util_static proto_obj erisfl libzmq)

add_library(algorithms_static STATIC)
target_link_libraries(algorithms_static algorithms_obj)

CleanCoverage(algorithms_static)
