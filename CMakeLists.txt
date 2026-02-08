cmake_minimum_required(VERSION 3.18)
project(TransformerCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- optional CUDA for GPU acceleration ----
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # VS 2026 (MSVC 19.50) isn't officially supported by CUDA 12.8 yet ;force it
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA found (${CMAKE_CUDA_COMPILER_VERSION}) - GPU acceleration enabled")
else()
    message(STATUS "CUDA not found - CPU-only mode")
endif()

# optional OpenMP for parallel CPU matmul ;fallback when CUDA not used
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel CPU matrix ops enabled")
else()
    message(STATUS "OpenMP not found - single-threaded CPU mode")
endif()

include_directories(include)

# ---- shared source files (everything except main entry points) ----
set(CORE_SOURCES
    src/Matrix.cpp
    src/tokenizer/encode.cpp
    src/tokenizer/decode.cpp
    src/tokenizer/buildVocab.cpp
    src/tokenizer/saveLoad.cpp
    src/layers/Embedding.cpp
    src/layers/PositionalEncoding.cpp
    src/layers/LayerNorm.cpp
    src/layers/FeedForward.cpp
    src/attention/MultiHeadAttention.cpp
    src/attention/computeAttention.cpp
    src/transformer/TransformerBlock.cpp
    src/transformer/Transformer.cpp
    src/transformer/CrossEntropyLoss.cpp
    src/transformer/AdamOptimizer.cpp
)

# ---- CUDA sources ----
if(CMAKE_CUDA_COMPILER)
    set(CUDA_SOURCES src/cuda_kernels.cu)
    # set CUDA architectures ;sm_120 for RTX 5090 (Blackwell), plus older for compat
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;120")
endif()

# ---- training binary ----
add_executable(train
    src/train.cpp
    ${CORE_SOURCES}
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:${CUDA_SOURCES}>
)

# ---- inference binary ----
add_executable(inference
    src/inference.cpp
    ${CORE_SOURCES}
    $<$<BOOL:${CMAKE_CUDA_COMPILER}>:${CUDA_SOURCES}>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(train PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(inference PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(train PUBLIC cudart)
    target_link_libraries(inference PUBLIC cudart)
endif()
