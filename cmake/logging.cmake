FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG        v1.13.0)
set(SPDLOG_BUILD_TESTS OFF)
set(SPDLOG_BUILD_TESTS_HO OFF)
FetchContent_MakeAvailable(spdlog)

set(LOGGING_INCLUDE_DIR ${spdlog_SOURCE_DIR}/include)
