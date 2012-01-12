find_package(OpenGL)
find_package(GLUT)

find_package(VTK)
include(${VTK_USE_FILE})
include_directories(${CUDA_SDK_ROOT_DIR}/C/common/inc/)

set (DATA_DIR "" CACHE PATH "Data directory")

cuda_add_executable(marching_cube marching_cube.cu OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(marching_cube vtkIO pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(tangle tangle.cu)
target_link_libraries(tangle vtkIO pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(cutplane cutplane.cu)
target_link_libraries(cutplane vtkIO pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(threshold threshold.cu)
target_link_libraries(threshold pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(threshold_timing threshold_timing.cu)
target_link_libraries(threshold_timing vtkIO pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(eddy eddy.cu OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(eddy vtkIO GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})