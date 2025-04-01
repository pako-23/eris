add_library(util_obj OBJECT
  include/util/networking.h
  networking.cc)
target_include_directories(util_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${libzmq_SOURCE_DIR}/include)
set_target_properties(util_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(util_obj
  PRIVATE compiler_flags libzmq-static)

add_library(util_static STATIC)
target_link_libraries(util_static util_obj)
set_target_properties(util_static PROPERTIES OUTPUT_NAME util)

CleanCoverage(util_static)
