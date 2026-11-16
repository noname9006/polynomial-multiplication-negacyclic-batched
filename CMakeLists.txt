# GENERATED for polymul/negacyclic-batched@1.0.0. Do not edit — `--update` rewrites it.
#
# A CUDA build: the CUDA language is enabled and the answer is compiled by nvcc.
# It differs from the plain C++ CMakeLists in three ways — the CUDA language,
# the device-LTO flags, and linking NVIDIA cuPQC when the challenge image
# provides it. Everything else is the same, and the platform builds it with the
# same `cmake -B build && cmake --build build`.
cmake_minimum_required(VERSION 3.24)
project(fherma_solution LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Release unless told otherwise: an answer measured at -O0 is not a measurement
# of the answer.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The GPU to build for. The platform's own runner page names the card:
# fherma-gpu-rtx6000 carries an NVIDIA RTX PRO 6000 Blackwell Server Edition,
# compute capability 12.0 — not the Ada board this file was generated against.
# sm_89 SASS would not load there at all, and the self-check would quietly keep
# the host path, so the default is the card that actually runs it. Override with
# -DFHERMA_CUDA_ARCH=sm_89 for an Ada machine. CMAKE_CUDA_ARCHITECTURES is
# left off deliberately: it emits -gencode, which nvcc refuses alongside the
# -dlto that cuPQC's device-LTO objects need, so the architecture is passed as
# -arch= instead.
set(FHERMA_CUDA_ARCH sm_120 CACHE STRING "nvcc -arch value")

add_executable(solution main.cpp solve.cu)
target_include_directories(solution PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(solution PROPERTIES
  CUDA_ARCHITECTURES OFF
  CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(solution PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-arch=${FHERMA_CUDA_ARCH};-dlto>)
target_link_options(solution PRIVATE
  $<DEVICE_LINK:-arch=${FHERMA_CUDA_ARCH};-dlto>)

# NVIDIA cuPQC, when the challenge image provides it (CUPQC_SDK_DIR is set in
# fherma/cupqc): its device BigInt library and headers. BigInt is linked because
# the wide-integer kernels are what a CUDA answer here usually reaches for; add
# cupqc-ntt, cupqc-hash or cupqc-pk the same way if you use them.
if(DEFINED ENV{CUPQC_SDK_DIR})
  target_include_directories(solution PRIVATE
    $ENV{CUPQC_SDK_DIR}/include
    $ENV{CUPQC_SDK_DIR}/include/cupqc)
  target_link_libraries(solution PRIVATE
    $ENV{CUPQC_SDK_DIR}/lib/libcupqc-bigint.a)
endif()
