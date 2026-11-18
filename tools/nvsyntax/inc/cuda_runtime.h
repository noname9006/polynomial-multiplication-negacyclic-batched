#pragma once
#include <cstddef>
typedef int cudaError_t;
const cudaError_t cudaSuccess = 0, cudaErrorInvalidValue = 1;
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize, cudaFuncAttributePreferredSharedMemoryCarveout, cudaFuncAttributeNonPortableClusterSizeAllowed };
struct CUstream_st; typedef CUstream_st* cudaStream_t;
struct CUevent_st; typedef CUevent_st* cudaEvent_t;
struct cudaDeviceProp { int major, minor; std::size_t sharedMemPerBlock, sharedMemPerBlockOptin; int multiProcessorCount; char name[256]; };
#define CUDA_ANY(name) template<class... A> inline cudaError_t name(A...) { return 0; }
CUDA_ANY(cudaMalloc) CUDA_ANY(cudaFree) CUDA_ANY(cudaMemcpy) CUDA_ANY(cudaMemcpyAsync) CUDA_ANY(cudaMemsetAsync) CUDA_ANY(cudaMemset)
CUDA_ANY(cudaStreamCreate) CUDA_ANY(cudaStreamDestroy) CUDA_ANY(cudaStreamSynchronize) CUDA_ANY(cudaGetDeviceCount)
CUDA_ANY(cudaSetDevice) CUDA_ANY(cudaGetDevice) CUDA_ANY(cudaGetDeviceProperties) CUDA_ANY(cudaPeekAtLastError)
CUDA_ANY(cudaDeviceSynchronize) CUDA_ANY(cudaEventCreate) CUDA_ANY(cudaEventDestroy) CUDA_ANY(cudaEventRecord)
CUDA_ANY(cudaEventElapsedTime) CUDA_ANY(cudaMemcpyToSymbol) CUDA_ANY(cudaFuncSetAttribute) CUDA_ANY(cudaMallocHost)
CUDA_ANY(cudaFreeHost) CUDA_ANY(cudaHostRegister) CUDA_ANY(cudaStreamWaitEvent) CUDA_ANY(cudaEventSynchronize)
CUDA_ANY(cudaDeviceGetAttribute) CUDA_ANY(cudaOccupancyMaxActiveBlocksPerMultiprocessor) CUDA_ANY(cudaEventCreateWithFlags)
inline cudaError_t cudaGetLastError() { return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return ""; }
struct dim3 { unsigned x, y, z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };

const unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2;
CUDA_ANY(cudaStreamCreateWithFlags)
CUDA_ANY(cudaMemcpyFromSymbol)
enum cudaLaunchAttributeID { cudaLaunchAttributeClusterDimension };
struct cudaLaunchAttributeValue { struct { unsigned x, y, z; } clusterDim; };
struct cudaLaunchAttribute { cudaLaunchAttributeID id; cudaLaunchAttributeValue val; };
struct cudaLaunchConfig_t { dim3 gridDim, blockDim; std::size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute* attrs; unsigned numAttrs; };
CUDA_ANY(cudaLaunchKernelEx) CUDA_ANY(cudaOccupancyMaxActiveClusters)
