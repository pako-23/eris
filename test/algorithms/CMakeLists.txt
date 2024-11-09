EnableCoverage(algorithms_obj)

add_executable(eris_config_test
  eris/service_config_test.cc
  eris/coordinator_config_test.cc
  eris/test_runner.cc)
target_link_libraries(eris_config_test PRIVATE
  algorithms_static
  proto_obj
  compiler_flags)
AddTests(eris_config_test)


add_executable(eris_coordinator_test
  eris/coordinator_test.cc
  mock_zmq_socket.cc)
target_link_libraries(eris_coordinator_test PRIVATE
  algorithms_static
  proto_obj
  compiler_flags)
target_include_directories(eris_coordinator_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
AddTests(eris_coordinator_test)


add_executable(eris_aggregator_test
  eris/aggregator_test.cc
  mock_zmq_socket.cc)
target_link_libraries(eris_aggregator_test PRIVATE
  algorithms_static
  proto_obj
  compiler_flags)
target_include_directories(eris_aggregator_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
AddTests(eris_aggregator_test)


add_executable(eris_split_test
  eris/split_test.cc)
target_link_libraries(eris_split_test PRIVATE
  algorithms_static
  proto_obj
  compiler_flags)
AddTests(eris_split_test)


add_executable(eris_client_test
  eris/client_join_test.cc
  eris/client_init_test.cc
  eris/client_train_test.cc
  eris/client_receive_weights_test.cc
  eris/client_weight_submit_test.cc
  eris/test_runner.cc
  mock_zmq_socket.cc)
target_link_libraries(eris_client_test PRIVATE
  algorithms_static
  proto_obj
  compiler_flags)
target_include_directories(eris_client_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
AddTests(eris_client_test)
