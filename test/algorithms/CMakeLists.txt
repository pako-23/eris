EnableCoverage(algorithms_obj)

add_executable(eris_builder_test
  eris/coordinator_builder_test.cc)
target_link_libraries(eris_builder_test PRIVATE
  proto_obj algorithms_static)
AddTests(eris_builder_test)


add_executable(eris_coordinator_test
  eris/coordinator_test.cc)
target_link_libraries(eris_coordinator_test PRIVATE
  proto_obj algorithms_static)
AddTests(eris_coordinator_test)
