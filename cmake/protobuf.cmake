
FetchContent_Declare(protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        v28.2)
set(ABSL_BUILD_TESTING OFF)
set(ABSL_PROPAGATE_CXX_STD ON)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_INSTALL OFF)
set(protobuf_BUILD_SHARED_LIBS ON)
FetchContent_MakeAvailable(protobuf)

set(protobuf_INCLUDE_DIR ${protobuf_SOURCE_DIR}/src)

function(compile_proto_files target)
  get_target_property(_source_list ${target} SOURCES)

  set(_generate_extensions .pb.h .pb.cc)
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

    list(APPEND _generated_srcs_all ${_generated_srcs})
      
    add_custom_command(
       OUTPUT ${_generated_srcs}
       COMMAND $<TARGET_FILE:protoc>
       ARGS --proto_path ${CMAKE_CURRENT_SOURCE_DIR}
            --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
	    ${_abs_file}
       DEPENDS ${_abs_file} $<TARGET_FILE:protoc>
       COMMENT "Running C++ protocol buffer compiler on ${_file}"
       VERBATIM)
   endforeach()

   set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
   target_sources(${target} PRIVATE ${_generated_srcs_all})
endfunction()
