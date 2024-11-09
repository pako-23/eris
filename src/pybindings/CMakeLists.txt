include(pybind11)

pybind11_add_module(eris bindings.cc)

target_link_libraries(eris PRIVATE
  algorithms_static
  compiler_flags
  proto_obj)
