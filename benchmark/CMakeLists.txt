find_package(VTK)
include(${VTK_USE_FILE})

cuda_add_executable(cayley_timing_GPU cayley_timing.cu)
target_link_libraries(cayley_timing_GPU pthread)

add_executable(cayley_timing_OMP cayley_timing.cpp)
set_target_properties(cayley_timing_OMP PROPERTIES COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
target_link_libraries(cayley_timing_OMP pthread gomp)

cuda_add_executable(marching_cube marching_cube.cu)
target_link_libraries(marching_cube vtkImaging pthread)

cuda_add_executable(marching_tetra marching_tetra.cu)
target_link_libraries(marching_tetra vtkImaging pthread)

add_executable(vtkTetra vtkTetra.cpp)
target_link_libraries(vtkTetra vtkImaging vtkGraphics)
