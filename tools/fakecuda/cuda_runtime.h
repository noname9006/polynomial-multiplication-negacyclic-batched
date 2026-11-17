// A declaration-only stand-in for <cuda_runtime.h>, used by the host check that
// compiles solve.cu with an ordinary C++ compiler (see tools/hostcheck.ps1).
//
// Nothing here is executed: the check is -fsyntax-only, and the interesting
// part is that every call in solve.cu is type-checked against signatures that
// match the real runtime (dim3, void**, size_t, the memcpy kinds, the stream
// handle), which is where a typo would otherwise cost a platform round trip.
#pragma once

#include <cstddef>

typedef int cudaError_t;
const cudaError_t cudaSuccess = 0;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
};

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

struct CUevent_st;
typedef CUevent_st* cudaEvent_t;

struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
};

cudaError_t cudaMalloc(void** devPtr, std::size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind,
                            cudaStream_t stream = nullptr);
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
const char* cudaGetErrorString(cudaError_t error);
