cmake_minimum_required(VERSION 2.8)

set (THRUST_DIR "" CACHE PATH "Thrust directory")
include_directories(${THRUST_DIR})

add_executable(vec_add vec_add.cpp)
set_target_properties(vec_add PROPERTIES
    COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -std=c++0x")
target_link_libraries(vec_add pthread gomp)

add_executable(quadratic quadratic.cpp)
set_target_properties(quadratic PROPERTIES
    COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -std=c++0x")
target_link_libraries(quadratic pthread gomp)

add_executable(integral integral.cpp)
set_target_properties(integral PROPERTIES
    COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -std=c++0x")
target_link_libraries(integral pthread gomp)

add_executable(integral2 integral2.cpp)
set_target_properties(integral2 PROPERTIES
    COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -std=c++0x")
target_link_libraries(integral2 pthread gomp)

add_executable(fibonacci fibonacci.cpp)
set_target_properties(fibonacci PROPERTIES
    COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -std=c++0x")
target_link_libraries(fibonacci pthread gomp)

add_executable(primes primes.cpp)
set_target_properties(primes PROPERTIES
    COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP -std=c++0x")
target_link_libraries(primes pthread gomp)



