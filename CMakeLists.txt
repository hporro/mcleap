cmake_minimum_required (VERSION 3.11 FATAL_ERROR)
project (
	CUDADelaunator
	DESCRIPTION "GPU based 2D Delaunay Triangulation"
	LANGUAGES CUDA CXX
)

# Cuda configuration
SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CUDA_ARCHITECTURES 50 60 70 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 20)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(playground playground.cu "Device_Triangulation.h" "Helpers_Triangulation.h" "Common_Triangulation.h")
target_include_directories(playground PUBLIC lib/glm)