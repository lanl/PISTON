find_package(OpenGL)
find_package(GLUT)
#find_package(GL
cuda_add_executable(scalar_field scalar_field.cu)
target_link_libraries(scalar_field pthread)

#cuda_add_executable(extraction extraction.cu)
#target_link_libraries(extraction pthread)

cuda_add_executable(threshold threshold.cu)
target_link_libraries(threshold pthread GLEW ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})

cuda_add_executable(simple_opengl simple_opengl.cu)
target_link_libraries(simple_opengl pthread)