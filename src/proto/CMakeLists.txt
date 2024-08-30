include(grpc)

add_library(proto_obj OBJECT
  algorithms/eris/aggregator.proto
  algorithms/eris/client.proto
  algorithms/eris/common.proto
  algorithms/eris/coordinator.proto)

target_link_libraries(proto_obj PUBLIC
  ${_PROTOBUF_LIB}
  #gRPC::grpc
  ${_GRPC_GRPCPP})

target_include_directories(proto_obj PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(proto_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

compile_proto_files(proto_obj)
