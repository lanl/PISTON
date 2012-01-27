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
QT4_WRAP_CPP(ex_HEADERS_MOC ${ex_HEADERS})
QT4_WRAP_UI(ex_FORMS_HEADERS ${ex_FORMS})

set(mc_SOURCES example.cpp marching_cube.cu quaternion.cpp)
cuda_add_executable(marching_cube ${mc_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(marching_cube GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

set(tangle_SOURCES example.cpp tangle.cu quaternion.cpp)
cuda_add_executable(tangle ${tangle_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS})
target_link_libraries(tangle GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

set(cp_SOURCES example.cpp cutplane.cu quaternion.cpp)
cuda_add_executable(cutplane ${cp_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(cutplane GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

set(threshold_SOURCES example.cpp threshold.cu quaternion.cpp)
cuda_add_executable(threshold ${threshold_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS})
target_link_libraries(threshold GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

set(eddy_SOURCES example.cpp eddy.cu quaternion.cpp)
cuda_add_executable(eddy ${eddy_SOURCES} ${ex_HEADERS_MOC} ${ex_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(eddy GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

