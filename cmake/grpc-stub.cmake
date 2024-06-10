
find_package(Threads REQUIRED)

set(PROTOC_INSTALLED ON)

if (PROTOC_INSTALLED)
  find_package(Protobuf CONFIG REQUIRED)
else ()
  FetchContent_Declare(
    absl
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp
    GIT_TAG 20230802.1
    OVERRIDE_FIND_PACKAGE)

  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_BUILD_TESTING OFF)
  set(ABSL_ENABLE_INSTALL ON)

  FetchContent_MakeAvailable(absl)
  FetchContent_Declare(
    grpc
    GIT_REPOSITORY https://github.com/grpc/grpc.git
    GIT_TAG        v1.62.0)
  set(gRPC_BUILD_TESTS OFF)
  set(RE2_BUILD_TESTING OFF)

  FetchContent_MakeAvailable(grpc)
  set(_PROTOBUF_LIBPROTOBUF libprotobuf)
  set(_REFLECTION grpc++_reflection)
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
  set(_GRPC_GRPCPP grpc++)
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif ()



# FetchContent_Declare( protobuf GIT_REPOSITORY
#   https://github.com/protocolbuffers/protobuf GIT_TAG v25.3
#   OVERRIDE_FIND_PACKAGE)

# set(protobuf_BUILD_TESTS OFF) set(protobuf_INSTALL ON)
# set(protobuf_ABSL_PROVIDER package)

# FetchContent_MakeAvailable(protobuf)
# FetchContent_Declare(
#   grpc
#   GIT_REPOSITORY https://github.com/grpc/grpc.git
#   GIT_TAG        v1.62.0)
# set(gRPC_BUILD_TESTS OFF)
# set(RE2_BUILD_TESTING OFF)
# # set(protobuf_BUILD_PROTOBUF_BINARIES OFF)
# FetchContent_MakeAvailable(grpc)

# set(_PROTOBUF_LIBPROTOBUF libprotobuf)
# set(_REFLECTION grpc++_reflection)
# set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
# set(_GRPC_GRPCPP grpc++)
# set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

set(GRPC_STUBS_INCLUDE_DIR "${CMAKE_BINARY_DIR}/grpc_stubs")

file(MAKE_DIRECTORY ${GRPC_STUBS_INCLUDE_DIR})

function(AddgRPCStub stub_name)
  set(grpc_proto_dependencies "")
  
  math(EXPR last "${ARGC} - 1")
  foreach(n RANGE 1 ${last})
    get_filename_component(proto "proto/${ARGV${n}}.proto" ABSOLUTE)
    get_filename_component(proto_path "${proto}" PATH)


    set(proto_hdrs "${GRPC_STUBS_INCLUDE_DIR}/${ARGV${n}}.pb.h")
    set(grpc_hdrs "${GRPC_STUBS_INCLUDE_DIR}/${ARGV${n}}.grpc.pb.h")
    set(proto_srcs "${GRPC_STUBS_INCLUDE_DIR}/${ARGV${n}}.pb.cc")
    set(grpc_srcs "${GRPC_STUBS_INCLUDE_DIR}/${ARGV${n}}.grpc.pb.cc")

    get_filename_component(proto_out_dir "${proto_hdrs}" PATH)

    string(REGEX REPLACE "[^A-Za-z0-9]" "_" create_proto_out_dir "${proto_out_dir}_out_dir")
    if (NOT TARGET "${create_proto_out_dir}")
      add_custom_target("${create_proto_out_dir}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_out_dir}")
    endif()

    add_custom_command(
      OUTPUT "${proto_hdrs}" "${grpc_hdrs}" "${proto_srcs}" "${grpc_srcs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${proto_out_dir}"
           --cpp_out "${proto_out_dir}"
           -I "${proto_path}"
           --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
           "${proto}"
      DEPENDS "${proto}" "${create_proto_out_dir}")
   
    list(APPEND grpc_proto_dependencies "${proto_hdrs}" "${grpc_hdrs}" "${proto_srcs}" "${grpc_srcs}")  
  endforeach()

  add_library(${stub_name}_obj OBJECT "${grpc_proto_dependencies}")
  target_link_libraries(${stub_name}_obj PUBLIC
    "${_REFLECTION}"
    "${_GRPC_GRPCPP}"
    "${_PROTOBUF_LIBPROTOBUF}")
  target_include_directories(${stub_name}_obj PUBLIC
    "${CMAKE_CURRENT_BINARY_DIR}")
  set_target_properties(${stub_name}_obj PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

  add_library(${stub_name}_static STATIC)
  target_link_libraries(${stub_name}_static ${stub_name}_obj)
endfunction()
