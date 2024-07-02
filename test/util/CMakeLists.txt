EnableCoverage(util_obj)

add_executable(networking_test valid_aggregator_publish_test.cc
  valid_aggregator_submit_test.cc
  valid_ipv4_test.cc)
target_link_libraries(networking_test PRIVATE util_static)
AddTests(networking_test)
