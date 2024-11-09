add_library(util_obj OBJECT
  include/util/networking.h
  networking.cc)
target_include_directories(util_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${libzmq_SOURCE_DIR}/include)
set_target_properties(util_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(util_obj
  PRIVATE compiler_flags libzmq)

add_library(util_shared SHARED)
target_link_libraries(util_shared util_obj)
add_library(util_static STATIC)
target_link_libraries(util_static util_obj)

CleanCoverage(util_static)
