find_package(OpenGL)
find_package(GLUT)
find_package(VTK)
include(${VTK_USE_FILE})
find_package(Qt4)
set(QT_USE_QTOPENGL TRUE)
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/demos)
include_directories(${CUDA_SDK_ROOT_DIR}/C/common/inc/)

set (DATA_DIR "" CACHE PATH "Data directory")


cuda_add_executable(contourGPU contour.cu isorender.cu quaternion.cpp OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DUSE_INTEROP)
target_link_libraries(contourGPU pthread vtkIO GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(contourOMP contour.cu isorender.cu quaternion.cpp OPTIONS -Xcompiler -fopenmp OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP)
target_link_libraries(contourOMP pthread vtkIO GLEW gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})


set(swdemo_SOURCES main.cpp glwindow.cpp isorender.cu quaternion.cpp)
set(swdemo_HEADERS glwindow.h)
set(swdemo_FORMS swdemo.ui)
QT4_WRAP_CPP(swdemo_HEADERS_MOC ${swdemo_HEADERS})
QT4_WRAP_UI(swdemo_FORMS_HEADERS ${swdemo_FORMS})

cuda_add_executable(swdemoGPU ${swdemo_SOURCES} ${swdemo_HEADERS_MOC} ${swdemo_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DUSE_INTEROP)
target_link_libraries(swdemoGPU GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

cuda_add_executable(swdemoOMP ${swdemo_SOURCES} ${swdemo_HEADERS_MOC} ${swdemo_FORMS_HEADERS} 
	OPTIONS -Xcompiler -fopenmp OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP)
target_link_libraries(swdemoOMP GLEW pthread vtkIO gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})


set(demo_SOURCES demo.cpp glwindow.cpp isorender.cu quaternion.cpp)
set(demo_HEADERS glwindow.h)
set(demo_FORMS demo.ui)
QT4_WRAP_CPP(demo_HEADERS_MOC ${demo_HEADERS})
QT4_WRAP_UI(demo_FORMS_HEADERS ${demo_FORMS})

cuda_add_executable(demoGPU ${demo_SOURCES} ${demo_HEADERS_MOC} ${demo_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DUSE_INTEROP)
target_link_libraries(demoGPU GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})

cuda_add_executable(demoOMP ${demo_SOURCES} ${demo_HEADERS_MOC} ${demo_FORMS_HEADERS} 
	OPTIONS -Xcompiler -fopenmp OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP)
target_link_libraries(demoOMP GLEW pthread vtkIO gomp ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})


set(bigdemo_SOURCES bigdemo.cpp glwindow.cpp isorender.cu quaternion.cpp)
set(bigdemo_HEADERS glwindow.h)
set(bigdemo_FORMS bigdemo.ui)
QT4_WRAP_CPP(bigdemo_HEADERS_MOC ${bigdemo_HEADERS})
QT4_WRAP_UI(bigdemo_FORMS_HEADERS ${bigdemo_FORMS})

cuda_add_executable(bigdemoGPU ${bigdemo_SOURCES} ${bigdemo_HEADERS_MOC} ${bigdemo_FORMS_HEADERS} OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DUSE_INTEROP)
target_link_libraries(bigdemoGPU GLEW pthread vtkIO ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${QT_LIBRARIES})


add_executable(contourVTK contourVTK.cpp)
set(CMAKE_INSTALL_RPATH ${VTK_DIR}/bin)
set_source_files_properties(contourVTK.cpp PROPERTIES COMPILE_FLAGS "-DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(contourVTK vtkRendering vtkFiltering vtkIO vtkCommon vtkGraphics)

add_executable(contourPAR contourPAR.cpp)
set_source_files_properties(contourPAR.cpp PROPERTIES COMPILE_FLAGS "-fopenmp -DDATA_DIRECTORY=${DATA_DIR}")
target_link_libraries(contourPAR vtkRendering vtkFiltering vtkIO vtkCommon vtkGraphics vtkParallel)


cuda_add_executable(glyphGPU glyph.cu glyphrender.cu quaternion.cpp OPTIONS "-DDATA_DIRECTORY=${DATA_DIR}" -DUSE_INTEROP)
target_link_libraries(glyphGPU pthread vtkIO vtkRendering vtkFiltering vtkCommon vtkGraphics GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
