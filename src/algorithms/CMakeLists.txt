include(logging)
include(pybind11)
include(coverage)

add_library(algorithms_obj OBJECT
  eris/builder.cc)
target_include_directories(algorithms_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${pybind11_INCLUDE_DIR}
  ${Python_INCLUDE_DIRS}
  ${GRPC_STUBS_INCLUDE_DIR}
  ${LOGGING_INCLUDE_DIR})
set_target_properties(algorithms_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(algorithms_obj
  PUBLIC util_static proto_obj)

add_library(algorithms_static STATIC)
target_link_libraries(algorithms_static algorithms_obj)

CleanCoverage(algorithms_static)
