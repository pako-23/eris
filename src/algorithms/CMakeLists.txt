include(logging)

add_library(algorithms_obj OBJECT
  eris/aggregation_strategy.cc
  eris/config.cc
  eris/split.cc
  include/algorithms/eris/aggregator.h
  include/algorithms/eris/aggregation_strategy.h
  include/algorithms/eris/client.h
  include/algorithms/eris/coordinator.h
  include/algorithms/eris/service.h)
target_include_directories(algorithms_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  ${spdlog_SOURCE_DIR}/include
  ${libzmq_SOURCE_DIR}/include)
set_target_properties(algorithms_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(algorithms_obj PRIVATE
  compiler_flags
  spdlog::spdlog
  libzmq-static)
target_link_libraries(algorithms_obj PUBLIC
  erisfl
  proto_obj
  util_static)

add_library(algorithms_static STATIC)
target_link_libraries(algorithms_static algorithms_obj)
set_target_properties(algorithms_static PROPERTIES OUTPUT_NAME algorithms)

CleanCoverage(algorithms_static)
