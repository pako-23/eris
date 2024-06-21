if (ERIS_ENABLE_MEMCHECK)
  FetchContent_Declare(memcheck-cover
    GIT_REPOSITORY https://github.com/Farigh/memcheck-cover.git
    GIT_TAG        release-1.2)
  FetchContent_MakeAvailable(memcheck-cover)

  set(REPORT_PATH "${CMAKE_BINARY_DIR}/memcheck")
  set(MEMCHECK_PATH ${memcheck-cover_SOURCE_DIR}/bin)

  if (NOT TARGET memcheck)
    add_custom_target(memcheck
      COMMAND ${MEMCHECK_PATH}/generate_html_report.sh
              -i ${REPORT_PATH}
              -o ${REPORT_PATH}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif()
endif()


function(AddMemcheck target)
  if (ERIS_ENABLE_MEMCHECK)
    add_custom_target(memcheck-${target}
      COMMAND ${MEMCHECK_PATH}/memcheck_runner.sh -o
              "${REPORT_PATH}/${target}"
	      -i "${CMAKE_SOURCE_DIR}/valgrind.supp"
              -- $<TARGET_FILE:${target}>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    add_dependencies(memcheck memcheck-${target})
  endif()
endfunction()
