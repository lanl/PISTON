cuda_add_executable(scalar_field scalar_field.cu)
target_link_libraries(scalar_field pthread)

#cuda_add_executable(extraction extraction.cu)
#target_link_libraries(extraction pthread)

cuda_add_executable(threshold threshold.cu)
target_link_libraries(threshold pthread)