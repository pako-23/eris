#set(ABSL_PROPAGATE_CXX_STD ON)
#set(ABSL_ENABLE_INSTALL ON)
#find_package(ZLIB REQUIRED)


set(ABSL_ENABLE_INSTALL ON)
set(ABSL_BUILD_TESTING OFF)
set(ABSL_PROPAGATE_CXX_STD ON)
# FetchContent_Declare(
#   absl
#   GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
#   GIT_TAG        origin/master
#   OVERRIDE_FIND_PACKAGE
# )
# FetchContent_MakeAvailable(absl)


# FetchContent_Declare(
#         protobuf
#         GIT_REPOSITORY https://github.com/google/protobuf.git
#         GIT_TAG        v28.0
#         GIT_PROGRESS   TRUE
#         GIT_SHALLOW    TRUE
#         USES_TERMINAL_DOWNLOAD TRUE
#         GIT_SUBMODULES_RECURSE FALSE
#         GIT_SUBMODULES "third_party/abseil-cpp")
# set(protobuf_BUILD_TESTS OFF)
# set(protobuf_BUILD_CONFORMANCE OFF)
# set(protobuf_BUILD_EXAMPLES OFF)
# set(protobuf_BUILD_PROTOC_BINARIES ON)
# set(protobuf_DISABLE_RTTI ON)
# set(protobuf_MSVC_STATIC_RUNTIME ON)
# set(protobuf_WITH_ZLIB ON CACHE BOOL "" FORCE)
# set(protobuf_INSTALL OFF)
# set(utf8_range_ENABLE_INSTALL OFF)
# FetchContent_MakeAvailable(protobuf)
#set(PROTOBUF_ROOT_DIR "${protobuf_SOURCE_DIR}")


FetchContent_Declare(
        grpc
        GIT_REPOSITORY https://github.com/grpc/grpc.git
        GIT_TAG        v1.60.0
        GIT_PROGRESS   TRUE
        GIT_SHALLOW    TRUE
        USES_TERMINAL_DOWNLOAD TRUE
        GIT_SUBMODULES_RECURSE FALSE
        GIT_SUBMODULES
            "third_party/cares"
            "third_party/boringssl-with-bazel"
            "third_party/re2"
            "third_party/abseil-cpp"
	    "third_party/protobuf")
set(gRPC_ABSL_BUILD_TESTING OFF)
set(gRPC_BUILD_TESTS OFF)
set(gRPC_BUILD_CODEGEN ON)
set(gRPC_BUILD_GRPC_CPP_PLUGIN ON)
set(gRPC_BUILD_CSHARP_EXT OFF)
set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF)
set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PYTHON_PLUGIN OFF)
set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF)
set(gRPC_BENCHMARK_PROVIDER "none" CACHE STRING "" FORCE)
#set(gRPC_ZLIB_PROVIDER "package" CACHE STRING "" FORCE)
FetchContent_MakeAvailable(grpc)


set(_PROTOC_BIN $<TARGET_FILE:protoc>)
set(_PROTOBUF_LIB libprotobuf)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

function(compile_proto_files target)
  get_target_property(_source_list ${target} SOURCES)

  set(_generate_extensions .pb.h .pb.cc .grpc.pb.h .grpc.pb.cc)
  set(_generated_srcs_all)
  foreach(_file ${_source_list})
    get_filename_component(_abs_file ${_file} ABSOLUTE)
    get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
    get_filename_component(_basename ${_file} NAME_WLE)
    file(RELATIVE_PATH _rel_dir ${CMAKE_CURRENT_SOURCE_DIR} ${_abs_dir})

    set(_generated_srcs)
    foreach(_ext ${_generate_extensions})
      list(APPEND _generated_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_rel_dir}/${_basename}${_ext}")
    endforeach()

    foreach(_src ${_generated_srcs})
      message(STATUS "Generating file: ${_src}")
    endforeach()

    list(APPEND _generated_srcs_all ${_generated_srcs})
      
    add_custom_command(
       OUTPUT ${_generated_srcs}
       COMMAND ${_PROTOC_BIN}
       ARGS --grpc_out ${CMAKE_CURRENT_BINARY_DIR}
            --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
            --proto_path ${CMAKE_CURRENT_SOURCE_DIR}
            --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}  ${_abs_file}
       DEPENDS ${_abs_file} ${_PROTOC_BIN}
       COMMENT "Running C++ protocol buffer compiler on ${_file}"
       VERBATIM)
   endforeach()

   set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
   target_sources(${target} PRIVATE ${_generated_srcs_all})
endfunction()
