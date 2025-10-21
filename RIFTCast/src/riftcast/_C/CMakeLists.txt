cmake_minimum_required(VERSION 3.27)

project(RIFTCast_lib LANGUAGES C CXX CUDA VERSION 0.1.0)


set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native")
string(APPEND CMAKE_CUDA_FLAGS " -use_fast_math")
find_package(CUDAToolkit REQUIRED)

#charonload
add_subdirectory(external/charonload)

# Add target library
set(target_name RIFTCast_lib)
charonload_add_torch_library(${target_name} STATIC)

# Collect all source files in this directory
file(GLOB_RECURSE source "include/**.h*" "src/**.c*")

# atcg_lib
set(ATCG_HEADLESS ${RIFT_HEADLESS})
set(ATCG_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/atcg_lib)

# torchhull
set(TORCHHULL_BUILD_BINDINGS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/torchhull/src/torchhull/_C)

# maskcompression
set(MASKCOMPRESSION_BUILD_BINDINGS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/maskcompression/src/maskcompression/_C)

# zstd
set(ZSTD_BUILD_STATIC ON)
set(ZSTD_BUILD_SHARED OFF)
add_subdirectory(external/zstd/build/cmake)

# HTTPRequest
add_library(HTTPRequest INTERFACE)
target_include_directories(HTTPRequest INTERFACE "external/HTTPRequest")
    
target_sources(${target_name} PRIVATE ${source})
target_link_libraries(${target_name} PUBLIC atcg_lib 
                                            torchhull::torchhull_cpp 
                                            CUDA::nvjpeg 
                                            maskcompression::maskcompression_cpp 
                                            libzstd_static 
                                            charonload::torch_python 
                                            Python::Python
                                            HTTPRequest)

target_include_directories(${target_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                                                 ${PROJECT_SOURCE_DIR}/external/json)

if(RIFTCAST_LIB_BUILD_BINDINGS)
    charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)
                                         
    target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/riftcast.cpp)
                                         
    target_link_libraries(${TORCH_EXTENSION_NAME} PRIVATE RIFTCast_lib)
endif()
