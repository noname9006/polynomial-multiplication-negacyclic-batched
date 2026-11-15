// A stand-in for the CUDA execution model, so the device arithmetic in
// polymul_gpu.cuh can be compiled and checked by an ordinary host compiler.
//
// It is deliberately the weakest possible model: one block, one thread. A
// kernel body then runs once, in order, with threadIdx == 0 and blockDim == 1,
// which is the semantics those kernels have anyway — every thread owns whole
// elements and the only cross-thread dependency is the barrier between NTT
// stages, which a single thread passes through by simply finishing the stage.
//
// What this cannot check is the parallel decomposition itself: whether block b
// and thread t are given the right elements. That is what the untimed self-test
// in fherma_init checks, on the real device, against the host path.
#pragma once

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __restrict__
#define __constant__
#define __forceinline__ inline

struct ShimIndex {
    unsigned x, y, z;
};
struct ShimSize {
    unsigned x, y, z;
};

inline ShimIndex threadIdx{0, 0, 0};
inline ShimIndex blockIdx{0, 0, 0};
inline ShimSize blockDim{1, 1, 1};
inline ShimSize gridDim{1, 1, 1};

inline void __syncthreads() {}

inline unsigned long long __umul64hi(unsigned long long a, unsigned long long b) {
    return static_cast<unsigned long long>((static_cast<unsigned __int128>(a) * b) >> 64);
}
