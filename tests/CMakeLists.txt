find_program(GCOV_PATH gcov)

if (NOT GCOV_PATH)
  message(FATAL_ERROR "gcov not found! Aborting...")
endif()


target_compile_options(compiler_flags INTERFACE -g -O0 -fprofile-arcs -ftest-coverage -lgcov)


FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0)
FetchContent_MakeAvailable(googletest)

add_executable(dummy
  dummy.cc)

target_link_libraries(
  dummy
  GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(dummy)
