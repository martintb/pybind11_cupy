cmake_minimum_required(VERSION 3.6)
project(pybind_cuda)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    )

link_directories(
    /opt/tbm/cuda/10.2/lib64/
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(demo SHARED
  demo.cu)

target_link_libraries(demo
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(demo PROPERTIES PREFIX "")
