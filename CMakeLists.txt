cmake_minimum_required(VERSION 3.28)
project(eris VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(ABSL_ENABLE_INSTALL ON)

add_library(eris_algo eris/eris/coordinator.cc eris/eris/client.cc)
set_target_properties(eris_algo PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(extern/pybind11)
pybind11_add_module(eris eris/pybindings.cc)
target_link_libraries(eris PRIVATE eris_algo)

add_subdirectory(extern/grpc EXCLUDE_FROM_ALL)

set(_REFLECTION grpc++_reflection)
set(_GRPC_GRPCPP grpc++)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

foreach(proto_file coordinator)
  get_filename_component(proto "eris/proto/${proto_file}.proto" ABSOLUTE)
  get_filename_component(proto_path "${proto}" PATH)

  set(proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/${proto_file}.pb.cc")
  set(proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${proto_file}.pb.h")
  set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${proto_file}.grpc.pb.cc")
  set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${proto_file}.grpc.pb.h")

  add_custom_command(
    OUTPUT "${proto_srcs}" "${proto_hdrs}" "${grpc_srcs}" "${grpc_hdrs}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
         -I "${proto_path}"
         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
         "${proto}"
    DEPENDS "${proto}")

  add_library(grpc_proto
    "${proto_srcs}"
    "${proto_hdrs}"
    "${grpc_srcs}"
    "${grpc_hdrs}")
  target_link_libraries(grpc_proto
    "${_REFLECTION}"
    "${_GRPC_GRPCPP}"
    "${_PROTOBUF_LIBPROTOBUF}")
endforeach()
set_target_properties(grpc_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(eris_algo
  grpc_proto
  "${_REFLECTION}"
  "${_GRPC_GRPCPP}"
  "${_PROTOBUF_LIBPROTOBUF}")
