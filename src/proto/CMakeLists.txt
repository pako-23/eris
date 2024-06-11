include(FindProtobuf)
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)
find_package(Threads REQUIRED)

add_library(proto_obj OBJECT
  algorithms/eris/aggregator.proto
  algorithms/eris/client.proto
  algorithms/eris/common.proto
  algorithms/eris/coordinator.proto)

target_link_libraries(proto_obj PUBLIC
  protobuf::libprotobuf
  gRPC::grpc
  gRPC::grpc++)
target_include_directories(proto_obj PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR})


protobuf_generate(TARGET proto_obj LANGUAGE cpp)
protobuf_generate(TARGET proto_obj
  LANGUAGE grpc
  PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc)
set_target_properties(proto_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
