cmake_minimum_required(VERSION 3.27)
project(eris VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(ABSL_ENABLE_INSTALL ON)
set(h_files_dir eris/include)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs")
include_directories(eris/include)
include_directories("${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs")

add_subdirectory(extern/pybind11)
pybind11_add_module(eris eris/pybindings.cc)

add_library(algorithms eris/algorithms/eris/coordinator.cc eris/algorithms/eris/client.cc)
set_target_properties(algorithms PROPERTIES POSITION_INDEPENDENT_CODE ON)

## Add gRPC libraries
set(_REFLECTION grpc++_reflection)
set(_GRPC_GRPCPP grpc++)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

add_subdirectory(extern/grpc EXCLUDE_FROM_ALL)

foreach(proto_file eris/coordinator)
  get_filename_component(proto "eris/proto/${proto_file}.proto" ABSOLUTE)
  get_filename_component(proto_path "${proto}" PATH)

  set(proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs/algorithms/${proto_file}.pb.h")
  set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs/algorithms/${proto_file}.grpc.pb.h")
  set(proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs/algorithms/${proto_file}.pb.cc")
  set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs/algorithms/${proto_file}.grpc.pb.cc")

  get_filename_component(proto_out_dir "${proto_hdrs}" PATH)
  add_custom_target(create_proto_out_dir 
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${proto_out_dir}")
  
  add_custom_command(
    OUTPUT "${proto_hdrs}" "${grpc_hdrs}" "${proto_srcs}" "${grpc_srcs}"
    COMMAND ${_PROTOBUF_PROTOC}
    ARGS --grpc_out "${proto_out_dir}"
         --cpp_out "${proto_out_dir}"
         -I "${proto_path}"
         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
         "${proto}"
    DEPENDS "${proto}" create_proto_out_dir)

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

## Add zmq libraries
set(ZeroMQ_DIR "${CMAKE_CURRENT_BINARY_DIR}/extern/libzmq")
add_subdirectory(extern/libzmq EXCLUDE_FROM_ALL)
include_directories(extern/libzmq)
include_directories(extern/cppzmq)

## Add boost log library


target_link_libraries(algorithms
  grpc_proto
  "${_REFLECTION}"
  "${_GRPC_GRPCPP}"
  "${_PROTOBUF_LIBPROTOBUF}"
  libzmq)


target_link_libraries(eris PRIVATE algorithms)
