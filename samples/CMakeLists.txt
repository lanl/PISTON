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
include_directories(${CMAKE_SOURCE_DIR}/demos)

set (DATA_DIR "" CACHE PATH "Data directory")
set (INIT_BUFFER_SIZE 10000 CACHE INT "Initial VBO Size")

if (ENABLE_INTEROP)
  set(GPU_FLAGS "-DUSE_INTEROP")
endif ()

set (USING_OSX true CACHE BOOL "Using OSX")

if (USING_OSX)
  set (OSX_FW_DIR "" CACHE PATH "OSX framework directory")
  include_directories(${OSX_FW_DIR})
endif ()


if (USE_CUDA)
  cuda_add_executable(mysimpleGPU my_simple_opengl.cu ${CMAKE_SOURCE_DIR}/piston/util/quaternion.cpp OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" "-DBUFFER_SIZE=${INIT_BUFFER_SIZE}" ${GPU_FLAGS})
  target_link_libraries(mysimpleGPU pthread vtkIO vtkRendering vtkFiltering vtkCommon vtkGraphics GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

#  cuda_add_executable(operatorTesterGPU operatorTester.cu logic2d.cu ${CMAKE_SOURCE_DIR}/piston/util/quaternion.cpp OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" "-DBUFFER_SIZE=${INIT_BUFFER_SIZE}" ${GPU_FLAGS})
#  target_link_libraries(operatorTesterGPU  pthread vtkIO vtkRendering vtkFiltering vtkCommon vtkGraphics GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})  
endif ()

#add_executable(operatorTesterOMP operatorTester.cu ${CMAKE_SOURCE_DIR}/piston/util/quaternion.cpp)
#set_target_properties(operatorTesterOMP PROPERTIES COMPILE_FLAGS "-fopenmp -DDATA_DIRECTORY=${DATA_DIR} -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
#target_link_libraries(operatorTesterOMP pthread vtkIO GLEW gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})



