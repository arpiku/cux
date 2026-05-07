cmake_minimum_required(VERSION 3.18)

# nvcc + GCC 16/libstdc++ needs host-side flags during compiler identification
# so CUDA parsing doesn't choke on C++20 library features like char8_t/requires.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-std=c++17 -Xcompiler=-fno-char8_t" CACHE STRING "CUDA compiler flags" FORCE)

project(cux LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architectures: Hopper (sm_90a) and Blackwell (sm_120)
set(CMAKE_CUDA_ARCHITECTURES 90a 120)

# Explicit flags for build types
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0 -lineinfo")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(cux
    main.cu
)

target_include_directories(cux PRIVATE
    .
)

find_package(CUDAToolkit REQUIRED)
# For CPU Parallel fun
find_package(OpenMP REQUIRED)

# CUDA + libstdc++ 16 trips over builtin traits while parsing STL headers.
# Disable those traits for CUDA compilation so nvcc can consume the headers.
target_compile_definitions(cux PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:_GLIBCXX_DO_NOT_USE_BUILTIN_TRAITS>
)

target_compile_options(cux PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
)

target_link_libraries(cux PRIVATE
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
)
