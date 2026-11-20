p = 'tc.cuh'
s = open(p, encoding='utf-8').read()

# flag near output section (before first #if that uses it we insert at top)
anchor = '// OC43: triple-buffer'
if anchor not in s:
    # master does not have OC43; insert before TC30 flag instead
    anchor = '#ifndef TC30_CPA\n#define TC30_CPA 1\n#endif\n'
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + '''// P4 probe (H2 bold): measure raw m16n8k32 u8 IMMA throughput at full NTT volume.
#ifndef P4_PROBE
#define P4_PROBE 1
#endif
''')
else:
    raise SystemExit('unexpected branch')

# probe kernel before mma_u8 definition (needs mma_u8) => insert right after mma_u8
kanchor = '''__device__ __forceinline__ void mma_u8(int (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3,
                                       u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
'''
assert s.count(kanchor) == 1
probe = kanchor + '''
#if P4_PROBE
// P4: warp-uniform loop of mma_u8; writes accumulators so the work cannot be
// optimised away.  Measures achievable tensor-core throughput, not NTT maths.
__global__ void p4_mma_probe(u32* __restrict__ out, const uint2* __restrict__ frag,
                             unsigned fwords, unsigned reps) {
    int acc[4] = {0, 0, 0, 0};
    const unsigned lane = threadIdx.x & 31;
#pragma unroll 1
    for (unsigned i = 0; i < reps; ++i) {
        const uint2 b = frag[(i * 32u + lane) % fwords];
        mma_u8(acc, 0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u, b.x, b.y);
    }
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = u32(acc[0]) + u32(acc[1]) + u32(acc[2]) + u32(acc[3]);
}
#endif
'''
s = s.replace(kanchor, probe)
open(p, 'w', encoding='utf-8').write(s)
print('tc.cuh patched: P4 kernel')
