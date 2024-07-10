cmake_minimum_required(VERSION 3.3)

project (norch)

execute_process(
  COMMAND python3 -m pybind11 --include 
  OUTPUT_VARIABLE PYBIND11_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall ${PYBIND11_INCLUDE}")
set(CMAKE_BUILD_TYPE Release)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../needle)

add_library(${PROJECT_NAME} SHARED src/main.cpp)
add_library(ndarray_backend_cpu SHARED src/ndarray_backend_cpu.cc)
# add_library(ndarray_backend_cuda SHARED src/ndarray_backend_cuda.cu)
