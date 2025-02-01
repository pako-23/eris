include(protobuf)

add_library(proto_obj OBJECT
  algorithms/eris/aggregator.proto
  algorithms/eris/common.proto
  algorithms/eris/coordinator.proto)

target_link_libraries(proto_obj PUBLIC
  protobuf::libprotobuf)

target_compile_features(proto_obj PRIVATE cxx_std_17)

target_include_directories(proto_obj PUBLIC
  ${Protobuf_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(proto_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

compile_proto_files(proto_obj)
