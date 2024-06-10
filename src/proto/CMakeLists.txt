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


get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)

protobuf_generate(TARGET proto_obj LANGUAGE cpp)
protobuf_generate(TARGET proto_obj
  LANGUAGE grpc
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")
