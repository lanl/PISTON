find_package(OpenGL)
find_package(GLUT)

find_package(Qt4)
set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/examples)

if (ENABLE_INTEROP)
  set(GPU_FLAGS "-DUSE_INTEROP")
endif ()


set(ex_HEADERS glwindow.h)
set(ex_FORMS example.ui)
set(ex_SOURCES example.cpp quaternion.cpp)
QT4_WRAP_CPP(ex_HEADERS_MOC ${ex_HEADERS})
QT4_WRAP_UI(ex_FORMS_HEADERS ${ex_FORMS})


if (USE_CUDA)
  cuda_add_executable(tangleGPU glwindow.cu ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DTANGLE" ${GPU_FLAGS})
  target_link_libraries(tangleGPU GLEW pthread ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

  cuda_add_executable(cutplaneGPU glwindow.cu ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DCUTPLANE" ${GPU_FLAGS})
  target_link_libraries(cutplaneGPU GLEW pthread ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

  cuda_add_executable(thresholdGPU glwindow.cu ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS ${GPU_FLAGS})
  target_link_libraries(thresholdGPU GLEW pthread ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})
endif ()


add_executable(tangleOMP glwindow.cpp ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS})
set_target_properties(tangleOMP PROPERTIES COMPILE_FLAGS "-fopenmp -DTANGLE -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
target_link_libraries(tangleOMP GLEW pthread gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

add_executable(cutplaneOMP glwindow.cpp ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS})
set_target_properties(cutplaneOMP PROPERTIES COMPILE_FLAGS "-fopenmp -DCUTPLANE -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
target_link_libraries(cutplaneOMP GLEW pthread gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

add_executable(thresholdOMP glwindow.cpp ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS})
set_target_properties(thresholdOMP PROPERTIES COMPILE_FLAGS "-fopenmp -DTHRESHOLD -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
target_link_libraries(thresholdOMP GLEW pthread gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})


add_executable(mandelbrot mandelbrot.cpp)
target_link_libraries(mandelbrot pthread)

#set(mc_SOURCES example.cpp marching_cube.cu quaternion.cpp)
#cuda_add_executable(marching_cube ${mc_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
#target_link_libraries(marching_cube GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

#set(eddy_SOURCES example.cpp eddy.cu quaternion.cpp)
#cuda_add_executable(eddy ${eddy_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
#target_link_libraries(eddy GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

