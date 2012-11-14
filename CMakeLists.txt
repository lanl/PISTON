cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(piston)

set (BUILD_DEMO true CACHE BOOL "Build demo (uses VTK)")
set (USE_CUDA true CACHE BOOL "Use CUDA")
set (ENABLE_INTEROP ON CACHE BOOL "Use interop")
set (USE_DISTRIBUTED false CACHE BOOL "Build distributed mem demos")

if (USE_CUDA)
  find_package(CUDA)
  include_directories(${CMAKE_SOURCE_DIR})
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
else ()
  set (THRUST_DIR "" CACHE PATH "Thrust directory")
  include_directories(${CMAKE_SOURCE_DIR})
  include_directories(${THRUST_DIR})
endif ()

add_subdirectory(examples)

if (BUILD_DEMO)
  add_subdirectory(demos)
  add_subdirectory(tutorial)
endif ()

add_subdirectory(benchmark)

