EnableCoverage(algorithms_obj)

add_executable(eris_builder_test
  eris/service_builder_test.cc
  eris/coordinator_builder_test.cc)
target_link_libraries(eris_builder_test PRIVATE
  proto_obj
  algorithms_static)
AddTests(eris_builder_test)
