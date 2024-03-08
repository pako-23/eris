cmake_minimum_required(VERSION 3.27)
project(eris
  VERSION 0.0.1
  LANGUAGES C CXX)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# pybind11 configurations
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1)
FetchContent_MakeAvailable(pybind11)

# gRPC configurations
set(ABSL_ENABLE_INSTALL ON)
find_package(Threads REQUIRED)
FetchContent_Declare(
  grpc
  GIT_REPOSITORY https://github.com/grpc/grpc.git
  GIT_TAG        v1.62.0)
FetchContent_MakeAvailable(grpc)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

# Add zmq libraries
FetchContent_Declare(
  libzmq
  GIT_REPOSITORY https://github.com/zeromq/libzmq.git
  GIT_TAG        v4.3.5)
FetchContent_MakeAvailable(libzmq)
FetchContent_Declare(
  cppzmq
  GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
  GIT_TAG        v4.10.0)
FetchContent_MakeAvailable(cppzmq)

# Add logging library
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.13.0)
FetchContent_MakeAvailable(spdlog)

# Building gRPC stubs
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs")

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


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_library(util eris/util/util.cc eris/include/util/util.h)
set_target_properties(util PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(util PRIVATE eris/include)


add_library(algorithms
  eris/include/algorithms/eris/coordinator.h
  eris/include/algorithms/eris/client.h
  eris/algorithms/eris/coordinator.cc
  eris/algorithms/eris/client.cc)
set_target_properties(algorithms PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(algorithms
  PRIVATE eris/include
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs"
  PRIVATE "${spdlog_SOURCE_DIR}/include"
  PRIVATE "${libzmq_SOURCE_DIR}"
  PRIVATE "${cppzmq_SOURCE_DIR}")
target_link_libraries(algorithms
  grpc_proto
  "${_REFLECTION}"
  "${_GRPC_GRPCPP}"
  "${_PROTOBUF_LIBPROTOBUF}"
  util
  libzmq)


pybind11_add_module(eris eris/pybindings.cc)
target_include_directories(eris
  PRIVATE eris/include
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/grpc_stubs"
  PRIVATE "${spdlog_SOURCE_DIR}/include"
  PRIVATE "${libzmq_SOURCE_DIR}"
  PRIVATE "${cppzmq_SOURCE_DIR}")
target_link_libraries(eris PRIVATE algorithms)
