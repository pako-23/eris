EnableCoverage(util_obj)

add_executable(networking_test
  valid_ipv4_test.cc
  valid_zmq_endpoint_test.cc)
target_link_libraries(networking_test PRIVATE
  compiler_flags
  util_static)
AddTests(networking_test)
