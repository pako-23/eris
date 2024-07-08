EnableCoverage(algorithms_obj)

add_executable(eris_builder_test
  eris/aggregator_builder_test.cc
  eris/coordinator_builder_test.cc)
target_link_libraries(eris_builder_test PRIVATE
  proto_obj algorithms_static)
AddTests(eris_builder_test)


add_executable(eris_coordinator_test
  eris/coordinator_test.cc)
target_link_libraries(eris_coordinator_test PRIVATE
  proto_obj algorithms_static)
AddTests(eris_coordinator_test)


add_executable(eris_aggregator_test
  eris/aggregator_test.cc)
target_link_libraries(eris_aggregator_test PRIVATE
  proto_obj algorithms_static)
AddTests(eris_aggregator_test)


add_executable(eris_split_test
  eris/split_test.cc)
target_link_libraries(eris_split_test PRIVATE
  proto_obj algorithms_static)
AddTests(eris_split_test)


add_executable(eris_client_test
  eris/client_join_test.cc
  eris/client_test.cc
  eris/client_receive_weights_test.cc
  eris/client_weight_submit_test.cc)
target_link_libraries(eris_client_test PRIVATE
  proto_obj algorithms_static)
target_include_directories(eris_client_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
AddTests(eris_client_test)
