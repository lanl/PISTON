find_package(OpenGL)
find_package(GLUT)

find_package(VTK)
include(${VTK_USE_FILE})

find_package(Qt4)
set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/examples)
include_directories(${CUDA_SDK_ROOT_DIR}/common/inc/)

set (DATA_DIR "" CACHE PATH "Data directory")

cuda_add_executable(marching_cube marching_cube.cu OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(marching_cube vtkIO pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(tangle tangle.cu)
target_link_libraries(tangle pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(tangle_omp tangle.cu
                OPTIONS -Xcompiler -fopenmp OPTIONS -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP)
target_link_libraries(tangle_omp pthread gomp GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(cutplane cutplane.cu)
target_link_libraries(cutplane vtkIO pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(threshold threshold.cu)
target_link_libraries(threshold pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(eddy eddy.cu OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(eddy vtkIO GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})