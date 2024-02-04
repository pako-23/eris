cmake_minimum_required(VERSION 3.4...3.18)
project(eris)

add_library(eris_algo STATIC eris/eris/coordinator.cc eris/eris/client.cc)
set_target_properties(eris_algo PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(extern/pybind11)
pybind11_add_module(eris eris/pybindings.cc)
target_link_libraries(eris PRIVATE eris_algo)
