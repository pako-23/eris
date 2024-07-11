include(pybind11)

pybind11_add_module(eris bindings.cc)

target_link_libraries(eris PRIVATE
  erisfl algorithms_obj proto_obj)
