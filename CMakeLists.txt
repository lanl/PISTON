cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(piston)

find_package(CUDA)

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(examples)
add_subdirectory(demos)
