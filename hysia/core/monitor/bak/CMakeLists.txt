cmake_minimum_required(VERSION 3.5)
project(monitor)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 11)
set(CUDA_INCLUDE /usr/local/cuda/include)
include_directories(${CUDA_INCLUDE})
include_directories(/usr/local/cuda/samples/common/inc)
link_directories(/usr/local/cuda/lib64 /usr/local/lib)

add_executable(monitor
	Monitor.cpp
	GpuDevice.cpp
	Resource.cpp
	TaskResource.cpp)

target_link_libraries(monitor
	-lcuda
	-lcudart
	-lpcre)

