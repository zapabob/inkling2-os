cmake_minimum_required(VERSION 3.28)
project(inkling2_os_engine_linux VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120a)

find_package(CUDAToolkit 13.5 REQUIRED)

# Reuses core .strata / tiered cache / sm_120a CUDA 13.5 kernels with Linux WDDM-free OS runtime
add_library(inkling2_linux_os STATIC
    src/linux_os_runtime.cpp
)

target_include_directories(inkling2_linux_os PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

target_link_libraries(inkling2_linux_os PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    uring
    zstd
    crypto
    pthread
)

target_compile_options(inkling2_linux_os PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -O3 -march=znver2 -mtune=znver2>
)
