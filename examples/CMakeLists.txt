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


set(ex_HEADERS glwindow.h)
set(ex_FORMS example.ui)
set(ex_SOURCES example.cpp glwindow.cu quaternion.cpp)
QT4_WRAP_CPP(ex_HEADERS_MOC ${ex_HEADERS})
QT4_WRAP_UI(ex_FORMS_HEADERS ${ex_FORMS})


cuda_add_executable(tangleGPU ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DTANGLE" "-DUSE_INTEROP")
target_link_libraries(tangleGPU GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

cuda_add_executable(cutplaneGPU ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DCUTPLANE" "-DUSE_INTEROP")
target_link_libraries(cutplaneGPU GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

cuda_add_executable(thresholdGPU ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DTHRESHOLD" )
target_link_libraries(thresholdGPU GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})


cuda_add_executable(tangleOMP ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS -Xcompiler -fopenmp OPTIONS "-DTANGLE" -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP)
target_link_libraries(tangleOMP GLEW pthread vtkIO gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

cuda_add_executable(cutplaneOMP ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS -Xcompiler -fopenmp OPTIONS "-DCUTPLANE" -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP)
target_link_libraries(cutplaneOMP GLEW pthread vtkIO gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

cuda_add_executable(thresholdOMP ${ex_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS -Xcompiler -fopenmp OPTIONS "-DTHRESHOLD" -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP)
target_link_libraries(thresholdOMP GLEW pthread vtkIO gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})



set(mc_SOURCES example.cpp marching_cube.cu quaternion.cpp)
cuda_add_executable(marching_cube ${mc_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(marching_cube GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

set(eddy_SOURCES example.cpp eddy.cu quaternion.cpp)
cuda_add_executable(eddy ${eddy_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(eddy GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

