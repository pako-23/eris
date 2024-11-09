if (eris_ENABLE_MEMCHECK)
  FetchContent_Declare(memcheck-cover
    GIT_REPOSITORY https://github.com/Farigh/memcheck-cover.git
    GIT_TAG        release-1.2)
  FetchContent_MakeAvailable(memcheck-cover)

  set(MEMCHECK_REPORTER ${memcheck-cover_SOURCE_DIR}/bin)
endif()


if (eris_ENABLE_MEMCHECK AND (NOT TARGET memcheck))
  set(MEMCHECK_REPORT ${CMAKE_BINARY_DIR}/memcheck)
  file(MAKE_DIRECTORY ${MEMCHECK_REPORT})
  add_custom_target(memcheck
    COMMAND ${MEMCHECK_REPORTER}/generate_html_report.sh
      -i ${MEMCHECK_REPORT}
      -o ${MEMCHECK_REPORT}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()


function(AddMemcheck target)
  if (eris_ENABLE_MEMCHECK)
    add_custom_target(memcheck-${target}
      COMMAND ${MEMCHECK_REPORTER}/memcheck_runner.sh
        -o ${MEMCHECK_REPORT}/${target} -- $<TARGET_FILE:${target}>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    add_dependencies(memcheck memcheck-${target})
  endif()
endfunction()
