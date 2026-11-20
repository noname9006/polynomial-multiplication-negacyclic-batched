p = 'tc.cuh'
s = open(p, encoding='utf-8').read()

anchor = '''__device__ __forceinline__ void mma_u8(int (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3,
                                       u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
'''
assert s.count(anchor) == 1
kernel = r'''
#ifndef P6_PROBE
#define P6_PROBE 1
#endif
#if P6_PROBE
// P6: sparse radix-16 twiddle products via mma_u8.  8 groups x 8 butterflies,
// 4 byte planes; rows=group, cols=butterfly i, k=i*4+j (byte j of the input).
// B is diagonal in (k,col).  Timing-only: writes accumulator sums to global.
__global__ void p6_sparse_probe(u32* __restrict__ out, const u32* __restrict__ vin,
                                const u32* __restrict__ btab, unsigned reps) {
    __shared__ u32 vs[8][8];
    const unsigned tid = threadIdx.x;
    const unsigned g = tid >> 2, t = tid & 3;
    if (g < 8) {
        vs[g][t]     = vin[g * 8 + t];
        vs[g][4 + t] = vin[g * 8 + 4 + t];
    }
    __syncthreads();
    if (g >= 8) return;                               // full mask before mma
    const u32 a0 = vs[g][t], a2 = vs[g][4 + t];
    const u32 z = 0u;
    int acc[4];
#pragma unroll 1
    for (unsigned it = 0; it < reps; ++it) {
        acc[0] = acc[1] = acc[2] = acc[3] = 0;
#pragma unroll 1
        for (unsigned sp = 0; sp < 4; ++sp) {
            const u32 bb0 = (g == t)     ? btab[t * 4 + sp] : z;
            const u32 bb1 = (g == 4 + t) ? btab[(4 + t) * 4 + sp] : z;
            mma_u8(acc, a0, z, a2, z, bb0, bb1);
        }
    }
    const unsigned idx = blockIdx.x * 256 + tid;
    out[idx] = u32(acc[0]) + u32(acc[1]) + u32(acc[2]) + u32(acc[3]);
}
#endif
'''
s = s.replace(anchor, anchor + kernel)
open(p, 'w', encoding='utf-8').write(s)
print('P6 kernel added')
