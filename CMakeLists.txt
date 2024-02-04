cmake_minimum_required(VERSION 3.4...3.18)
project(eris)

add_library(coordinator STATIC eris/coordinator.cc)
set_target_properties(coordinator PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(client STATIC eris/client.cc)
set_target_properties(client PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(extern/pybind11)
pybind11_add_module(eris eris/pybindings.cc)
target_link_libraries(eris PRIVATE coordinator client)
