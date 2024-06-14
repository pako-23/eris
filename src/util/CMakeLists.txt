add_library(util_obj OBJECT networking.cc)
target_include_directories(util_obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(util_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)


add_library(util_shared SHARED)
target_link_libraries(util_shared util_obj)
add_library(util_static STATIC)
target_link_libraries(util_static util_obj)

CleanCoverage(util_static)
