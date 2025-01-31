include(pybind11)

pybind11_add_module(eris bindings.cc)

target_link_libraries(eris PRIVATE
  algorithms_static
  compiler_flags
  proto_obj)

set_target_properties(eris PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION ON
  CXX_VISIBILITY_PRESET internal
  VISIBILITY_INLINES_HIDDEN ON)

if(APPLE)
  set_target_properties(${MODULE} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()
