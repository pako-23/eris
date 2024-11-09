if (eris_ENABLE_COVERAGE)
  find_program(GCOV_PATH gcov REQUIRED)
  find_program(LCOV_PATH lcov REQUIRED)
  find_program(GENHTML_PATH genhtml REQUIRED)
endif()

function(EnableCoverage target)
  if (eris_ENABLE_COVERAGE)
    target_compile_options(${target} PRIVATE
      --coverage -fno-inline -fprofile-update=atomic)
    target_link_libraries(${target} PRIVATE gcov)
  endif()
endfunction()

function(CleanCoverage target)
  add_custom_command(TARGET ${target} PRE_BUILD COMMAND
                     find ${CMAKE_BINARY_DIR} -type f
                     -name '*.gcda' -delete)
endfunction()

if (eris_ENABLE_COVERAGE)
  if (NOT TARGET coverage)
    add_custom_target(coverage
      COMMAND ${LCOV_PATH} -d . --zerocounters
      COMMAND ${CMAKE_MAKE_PROGRAM} test
      COMMAND ${LCOV_PATH} -d . --capture -o coverage.info
      COMMAND ${LCOV_PATH} -r coverage.info '/usr/include/*'
                           -o filtered.info
                           --ignore-errors unused version
      COMMAND ${GENHTML_PATH} -o coverage
                              filtered.info --legend
      COMMAND rm -rf coverage.info filtered.info
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
endif()

function(AddCoverage target)
  if (eris_ENABLE_COVERAGE)
    add_custom_target(coverage-${target}
      COMMAND ${LCOV_PATH} -d . --zerocounters
      COMMAND $<TARGET_FILE:${target}>
      COMMAND ${LCOV_PATH} -d . --capture -o coverage.info
      COMMAND ${LCOV_PATH} -r coverage.info '/usr/include/*'
                           -o filtered.info
                           --ignore-errors unused version
      COMMAND ${GENHTML_PATH} -o coverage-${target}
                              filtered.info --legend
      COMMAND rm -rf coverage.info filtered.info
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endfunction()
