// crt_mma.cuh — EXACT tensor-core CRT reconstruction for the batched 31-bit RNS
// polynomial multiplication (drop-in replacement for gpu31::reconstruct_kernel).
//
// Header-only. Create only this file; nothing else is modified. Integrate by
// #include "crt_mma.cuh" after gpu31.cuh (the CrtArgs type is duck-typed, so a
// gpu31::CrtArgs can be passed directly — see run/run_s8/run_simple below).
//
// ---------------------------------------------------------------------------
// WHY THIS FILE EXISTS / THE SIGN TRAP (read this before changing anything)
// ---------------------------------------------------------------------------
// The reference CRT computes, for every limb i and coefficient c,
//
//        X_i(c) = sum_{j=0}^{K} terms[j](c) * weights[i*pitch + j]        (u32*u32)
//
// as an EXACT big integer (split into a low/high u64 pair) and then reduces it
// mod q with the existing carry_and_reduce/reduce_sum logic.  To put this on
// int8 tensor cores we split each u32 operand into its four bytes and use
// mma.sync.aligned.m16n8k32 (... .s32.<atype>.<btype>.s32) to form the 16
// partial sums  S[c1][c2] = sum_j byte_c1(w_ij) * byte_c2(t_j), then recombine
// them with shifts 2^(8*(c1+c2)).
//
// *** The naive "chunk[c] = int8_t((v >> (8*c)) & 0xFF)" trick is WRONG. ***
// Let s_c = int8_t(byte_c) and e_c = bit7(byte_c) in {0,1}.  Then
//     byte_c = s_c + 256*e_c   (mod 2^32 the 256*e_3 term vanishes, but the
//                              256*e_0/1/2 terms do NOT).
// Consequently  sum_c s_c 2^(8c) = v - 256*sum_c e_c 2^(8c) != v (mod 2^32),
// and the raw s8 MMA result is S(v)*S(w), not v*w.  Concrete counterexample:
//     v = 0x00000080, w = 0x00000001.
//     signed bytes: v -> {-128,0,0,0}, w -> {1,0,0,0}; MMA gives -128 (=
//     0xFFFFFF80), but the true low-32 product is +128.  So s8 MMA with raw
//     bytes can NOT be byte-for-byte identical to the reference.
//
// We therefore provide TWO exact tensor-core paths and one CUDA-core path:
//
//   * run()      — TENSOR CORE, `.u8.u8.s32`.  Bytes are fed as UNSIGNED
//                  (PTX `u8`), so each byte-pair product is exactly the u32
//                  byte product. This is the fast, recommended path. It uses
//                  the very same m16n8k32 hardware instruction as s8 (only the
//                  type qualifier differs) — 16 byte-pair passes, K=pitch.
//
//   * run_s8()   — TENSOR CORE, `.s8.s8.s32` (the literal instruction requested
//                  by the task), made EXACT with a 4-way compensation. Each
//                  u32 byte u is written as u = (u & 0x7F) + 128*(u >> 7); both
//                  pieces are in [0,127] and hence are valid non-negative s8
//                  values.  Expanding the product of two such decompositions
//                  needs four MMA accumulations per (c1,c2) byte-pair (weights
//                  1, 128, 128, 16384) instead of one, so this path is ~4x the
//                  MMA work of run(). Use it only if you specifically want the
//                  s8 encoding; run() is faster and equally exact.
//
//   * run_simple() — CUDA-core reference sum_column (copied verbatim from
//                  gpu31.cuh).  Use for A/B testing / debugging.
//
// All three paths produce byte-for-byte identical output to reconstruct_kernel.
//
// ---------------------------------------------------------------------------
// EXACTNESS OF THE BIG-INTEGER REASSEMBLY
// ---------------------------------------------------------------------------
// The MMA gives S[c1][c2] = sum_j byte_c1(w_ij)*byte_c2(t_j), each < 2^23.
// We rebuild X_i = sum_{c1,c2} S[c1][c2] << (8*(c1+c2)) in a 128-bit accumulator
// (X_i < 2^70 for K<=118).  We then feed carry_and_reduce the pair
//     low[i]  = u32(X_i)          (bits 0..31)
//     high[i] = u64(X_i >> 32)    (bits 32.., < 2^38)
// which satisfies low[i] + high[i]*2^32 == X_i exactly.  carry_and_reduce is
// linear in (low,high), so the assembled big integer sum_i X_i*2^(32*i) is
// identical to the reference's (whose low/high are the per-product splits).
// Using low=u32(X) (not the full low 64 bits) also keeps the u64 carry in
// carry_and_reduce from overflowing.  Final reduction is the unmodified helper
// copied from gpu31.cuh, so canonicalization is bit-identical.
//
// ---------------------------------------------------------------------------
// ASSUMPTIONS / RISKS (nothing could be compiled locally)
// ---------------------------------------------------------------------------
//  - sm_80+ (m16n8k32 integer MMA); target here is sm_120.  PTX verified against
//    PTX ISA 9.5 sec. 9.7.16.5.10 (fragment layouts) and 9.7.16.5.14 (types).
//  - __uint128_t is supported in device code by nvcc (used only to assemble X_i).
//  - Weights in device memory are stored as 4 consecutive bytes per u32 in
//    little-endian order (exactly the host u32 bit pattern), so the kernel can
//    read them back as u32 on the little-endian GPU. cudaMalloc is 256B aligned.
//  - The device weight table holds L*pitch u32; the K+1'st column is the
//    pre-built correction column. We recompute the correction with FP64 anyway
//    (see step 2) and simply ignore the stored correction column's value (we
//    still allocate/read it so layouts line up). This matches the reference.
//  - M_COEF=32 coefficients per block, 128 threads (4 warps), one 16x8 MMA tile
//    per warp; L is capped at CRT_MMA_MAXL=128 and pitch at CRT_MMA_MAXPITCH=512.
//  - The mma asm uses "+r" accumulators and "=r"-style fragment regs; all 32
//    lanes of a warp execute identical, uniform loops (mma .aligned satisfied).
//  - Correction uses double exactly as the reference (round(sum term/p) ); error
//    is far inside the 1/4 margin for K<=1024.
// ===========================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstddef>

#include <cuda_runtime.h>          // shim provides the error enum in host builds
#ifdef __CUDACC__
#include <mma.h>                   // included per spec; the MMA itself is inline PTX
#endif

#include "fherma.h"

namespace crt_mma {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__)
#define CRTMMA_FHD __host__ __device__
#else
#define CRTMMA_FHD
#endif

// Compile-time limits (raise if a future point needs more; the host side checks).
static constexpr unsigned CRT_MMA_MAXL     = 128;   // max limbs
static constexpr unsigned CRT_MMA_MAXPITCH = 512;   // max K+1 columns
static constexpr unsigned M_COEF           = 32;    // coefficients per block
static constexpr unsigned MMA_WARPS        = 4;     // 4 warps * 8 coefs = 32
static constexpr unsigned MMA_THREADS      = MMA_WARPS * 32;

// ---------------------------------------------------------------------------
// Device-side mirror of the CRT structs (layout-identical to gpu31's).
// ---------------------------------------------------------------------------
struct PrimeInfo {
    u32 p, inv, inv_shoup;
    double reciprocal;
};

struct CrtArgs {
    unsigned n, limbs, bits, count;
    u32 gap;
    const PrimeInfo* primes;
    const u32* weights;
    const u32* q;
};

// ===========================================================================
// 1. Helpers copied verbatim from gpu31.cuh's fherma_crt section
//    (Stride is kept for fidelity; all uses here instantiate Stride=1).
// ===========================================================================
CRTMMA_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template <unsigned Stride>
CRTMMA_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}

template <unsigned Stride>
CRTMMA_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
                                      unsigned limbs, unsigned shift) {
    for (unsigned j = size; j-- > 0;) {
        const u32 b = shifted_word<Stride>(q, limbs, j, shift);
        if (a[j * Stride] < b) return false;
        if (a[j * Stride] > b) break;
    }
    u64 borrow = 0;
    for (unsigned j = 0; j < size; ++j) {
        const u64 b = u64(shifted_word<Stride>(q, limbs, j, shift)) + borrow;
        const u32 x = a[j * Stride];
        a[j * Stride] = x - u32(b);
        borrow = u64(x) < b;
    }
    return true;
}

template <unsigned Stride>
CRTMMA_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}

template <unsigned Stride>
CRTMMA_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
    const unsigned size = limbs + 2;
    if (gap) {
        for (unsigned iteration = 0; iteration < 5; ++iteration) {
            const u32 high0 = read_window<Stride>(a, size, bits);
            const u32 high1 = read_window<Stride>(a, size, bits + 32);
            if (!(high0 | high1)) {
                subtract_if_ge<Stride>(a, size, q, limbs, 0);
                return;
            }
            unsigned cut = bits / 32;
            const unsigned off = bits % 32;
            if (off) a[cut++ * Stride] &= (u32(1) << off) - 1;
            for (unsigned j = cut; j < size; ++j) a[j * Stride] = 0;
            u64 carry = u64(high0) * gap + a[0];
            a[0] = u32(carry);
            carry >>= 32;
            carry += u64(high1) * gap + a[Stride];
            a[Stride] = u32(carry);
            carry >>= 32;
            for (unsigned j = 2; carry && j < size; ++j) {
                carry += a[j * Stride];
                a[j * Stride] = u32(carry);
                carry >>= 32;
            }
        }
    }
    for (unsigned s = 43; s-- > 0;)
        if (bits + s <= size * 32) subtract_if_ge<Stride>(a, size, q, limbs, s);
}

template <unsigned Stride>
CRTMMA_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes, unsigned count) {
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}

template <unsigned Stride>
CRTMMA_FHD inline void sum_column(const u32* terms, const u32* basis_column,
                                  unsigned pitch, u64& low_sum, u64& high_sum) {
    u64 lo = 0, hi = 0;
    for (unsigned j = 0; j < pitch; ++j) {
        const u64 product = u64(terms[j * Stride]) * basis_column[j];
        lo += u32(product);
        hi += product >> 32;
    }
    low_sum = lo;
    high_sum = hi;
}

template <unsigned Stride>
CRTMMA_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
                                        unsigned l, unsigned bits, u32 gap, const u32* q) {
    u64 carry = 0, prior_high = 0;
    for (unsigned i = 0; i < l; ++i) {
        carry += low[i * Stride] + prior_high;
        digits[i * Stride] = u32(carry);
        carry >>= 32;
        prior_high = high[i * Stride];
    }
    carry += prior_high;
    digits[l * Stride] = u32(carry);
    digits[(l + 1) * Stride] = u32(carry >> 32);
    reduce_sum<Stride>(digits, l, bits, gap, q);
}

// ===========================================================================
// 2. CUDA-core reference kernel (run_simple path) — byte-identical to gpu31.
// ===========================================================================
#if defined(__CUDACC__)
__global__ void crt_simple_kernel(CrtArgs plan, const u32* residues,
                                  const u32* weights, u32* out) {
    const unsigned coef = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = blockIdx.y;
    if (coef >= plan.n) return;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;

    u32 terms[CRT_MMA_MAXPITCH];
    u64 low[CRT_MMA_MAXL], high[CRT_MMA_MAXL];
    u32 digits[CRT_MMA_MAXL + 2];

    const u64 rbase = u64(pair) * k * plan.n;
    for (unsigned j = 0; j < k; ++j)
        terms[j] = weight_residue(residues[rbase + u64(j) * plan.n + coef], plan.primes[j]);
    terms[k] = crt_correction<1>(terms, plan.primes, k);
    for (unsigned i = 0; i < l; ++i)
        sum_column<1>(terms, weights + i * pitch, pitch, low[i], high[i]);
    carry_and_reduce<1>(digits, low, high, l, plan.bits, plan.gap, plan.q);

    const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
    for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i];
}

// ===========================================================================
// 3. Tensor-core kernel (run / run_s8 paths).
// ===========================================================================

// ---- MMA fragment helpers ---------------------------------------------------
// PTX ISA 9.5, sec 9.7.16.5.10, m16n8k32 with .s8/.u8 operands.
//   A (16x32, row): each thread holds a[0..3] (4 packed u32 = 16 bytes).
//     groupID = lane>>2, tig = lane%4.
//     a[0]: row=groupID,      K=tig*4 + {0..3}
//     a[1]: row=groupID+8,    K=tig*4 + {0..3}
//     a[2]: row=groupID,      K=tig*4 + {0..3} + 16
//     a[3]: row=groupID+8,    K=tig*4 + {0..3} + 16
//   B (32x8, col): each thread holds b[0..1] (2 packed u32 = 8 bytes).
//     b[0]: K=tig*4+{0..3},      N=groupID
//     b[1]: K=tig*4+{0..3}+16,   N=groupID
//   C/D (16x8, s32): each thread holds c[0..3].
//     c[0]: row=groupID,   N=tig*2
//     c[1]: row=groupID,   N=tig*2+1
//     c[2]: row=groupID+8, N=tig*2
//     c[3]: row=groupID+8, N=tig*2+1
// Packing: byte b (0..3) of a u32 register is element index (... + b), i.e. the
// lowest-address byte is element 0.

// EB selects how a u32 source word contributes one int8 MMA element:
//   0 : the full byte (0..255)            -> used with .u8.u8.s32 (exact unsigned)
//   1 : (byte & 0x7F)  in [0,127]          -> low part of the s8 compensation
//   2 : bit 7 of the byte, in {0,1}        -> high part of the s8 compensation
template <int EB>
__device__ __forceinline__ u32 mma_byte(u32 v, unsigned byteIdx) {
    if (EB == 0) return (v >> (8u * byteIdx)) & 0xFFu;
    if (EB == 1) return (v >> (8u * byteIdx)) & 0x7Fu;
    return (v >> (8u * byteIdx + 7u)) & 1u;
}

template <int EB>
__device__ __forceinline__ void make_a(u32 a[4], const u32* sw, unsigned pitch,
                                       unsigned row0, unsigned kb, unsigned lane,
                                       unsigned cb, unsigned L) {
    const unsigned base = lane >> 2, tig = lane & 3;
    const unsigned r0 = row0 + base, r1 = row0 + base + 8;
#pragma unroll
    for (unsigned b = 0; b < 4; ++b) {
        const unsigned k0 = kb * 32 + tig * 4 + b;
        const unsigned k1 = kb * 32 + 16 + tig * 4 + b;
        const u32 v0 = (r0 < L && k0 < pitch) ? sw[r0 * pitch + k0] : 0u;
        const u32 v1 = (r1 < L && k0 < pitch) ? sw[r1 * pitch + k0] : 0u;
        const u32 v2 = (r0 < L && k1 < pitch) ? sw[r0 * pitch + k1] : 0u;
        const u32 v3 = (r1 < L && k1 < pitch) ? sw[r1 * pitch + k1] : 0u;
        a[0] |= mma_byte<EB>(v0, cb) << (8u * b);
        a[1] |= mma_byte<EB>(v1, cb) << (8u * b);
        a[2] |= mma_byte<EB>(v2, cb) << (8u * b);
        a[3] |= mma_byte<EB>(v3, cb) << (8u * b);
    }
}

template <int EB>
__device__ __forceinline__ void make_b(u32 b[2], const u32* st, unsigned pitch,
                                       unsigned m0, unsigned kb, unsigned lane,
                                       unsigned db) {
    const unsigned tig = lane & 3;
    const unsigned n = m0 + (lane >> 2);              // N index within the tile
#pragma unroll
    for (unsigned q = 0; q < 4; ++q) {
        const unsigned k0 = kb * 32 + tig * 4 + q;
        const unsigned k1 = kb * 32 + 16 + tig * 4 + q;
        // st has exactly `pitch` rows of M_COEF u32. Rows >= pitch are the
        // zero padding of the last k-block; reading past them would alias s_low.
        const u32 v0 = (k0 < pitch) ? st[k0 * M_COEF + n] : 0u;
        const u32 v1 = (k1 < pitch) ? st[k1 * M_COEF + n] : 0u;
        b[0] |= mma_byte<EB>(v0, db) << (8u * q);
        b[1] |= mma_byte<EB>(v1, db) << (8u * q);
    }
}

// One full dot product over K=pitch (nb = ceil(pitch/32) m16n8k32 instructions).
// USE_U8 selects the PTX type qualifier; EB_A/EB_B select operand extraction.
template <bool USE_U8, int EB_A, int EB_B>
__device__ __forceinline__ void mma_dot(int acc[4], const u32* sw, const u32* st,
                                        unsigned pitch, unsigned row0, unsigned m0,
                                        unsigned lane, unsigned cb, unsigned db,
                                        unsigned L) {
    const unsigned nb = (pitch + 31) / 32;
#pragma unroll
    for (unsigned kb = 0; kb < nb; ++kb) {
        u32 a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
        make_a<EB_A>(a, sw, pitch, row0, kb, lane, cb, L);
        make_b<EB_B>(b, st, pitch, m0, kb, lane, db);
        if constexpr (USE_U8) {
            // Unsigned bytes -> EXACT u32 byte products, s32 accumulate.
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        } else {
            // Signed bytes, but both operands are in [0,127] (see file header).
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
    }
}

__device__ __forceinline__ void add_X(__uint128_t X[4], const int acc[4],
                                      u64 scale, unsigned shift) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        // acc >= 0 for both MMA modes; treat as unsigned for the shift.
        const u64 v = u64(u32(acc[r])) * scale;
        X[r] += (__uint128_t)v << shift;
    }
}

// MODE 0 = u8 MMA (one pass per byte-pair); MODE 1 = s8 MMA with 4-way
// compensation (see header). Both produce the exact X_i.
template <int MODE>
__global__ void crt_mma_kernel(CrtArgs plan, const u32* residues,
                               const u32* d_weights, u32* out) {
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned pair = blockIdx.z;
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned tid = threadIdx.x;
    const unsigned m0 = warp * 8;                      // this warp's 8 coefficients

    extern __shared__ unsigned char smem[];
    u32* sw = reinterpret_cast<u32*>(smem);                       // L*pitch
    u32* st = sw + l * pitch;                                     // pitch*M_COEF
    // Round the u32 count up to an even number so the u64 arrays below are
    // 8-byte aligned for every (L,pitch), not just the official point.
    const unsigned before_u64 = pitch * (l + M_COEF);
    const unsigned aligned_u64 = (before_u64 + 1u) & ~1u;
    u64* s_low = reinterpret_cast<u64*>(st + aligned_u64 - l * pitch);
    u64* s_high = s_low + l * M_COEF;

    // ---- load the CRT weight table (int8 buffer read back as u32) ----------
    for (unsigned idx = tid; idx < l * pitch; idx += MMA_THREADS)
        sw[idx] = d_weights[idx];

    // ---- step 1: Shoup terms  t_j = r_j * (P/p_j)^-1 mod p_j ---------------
    // st[j][m] = terms[j] for each of this block's M_COEF coefficients. Flatten
    // the (j,m) space so every (j,m) is written by exactly one thread.
    const u64 rbase = u64(pair) * k * plan.n;
    for (unsigned idx = tid; idx < k * M_COEF; idx += MMA_THREADS) {
        const unsigned j = idx / M_COEF, m = idx % M_COEF;
        const unsigned coef = blockIdx.x * M_COEF + m;
        const u32 r = (coef < plan.n)
                    ? residues[rbase + u64(j) * plan.n + coef] : 0u;
        st[j * M_COEF + m] = weight_residue(r, plan.primes[j]);
    }
    __syncthreads();

    // ---- step 2: FP64 correction, identical to reference crt_correction ----
    if (tid < M_COEF) {
        const unsigned m = tid;
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
            sum += double(st[j * M_COEF + m]) * plan.primes[j].reciprocal;
        st[k * M_COEF + m] = u32(sum + 0.5);
    }
    __syncthreads();

    // ---- step 3: tensor-core GEMM -> exact X_i, then low/high split --------
    const unsigned nTiles = (l + 15) / 16;
    for (unsigned lt = 0; lt < nTiles; ++lt) {
        const unsigned row0 = lt * 16;
        __uint128_t X[4] = {0, 0, 0, 0};
        for (unsigned cb = 0; cb < 4; ++cb) {
            for (unsigned db = 0; db < 4; ++db) {
                const unsigned shift = 8u * (cb + db);
                if constexpr (MODE == 0) {
                    int acc[4] = {0, 0, 0, 0};
                    mma_dot<true, 0, 0>(acc, sw, st, pitch, row0, m0,
                                        lane, cb, db, l);
                    add_X(X, acc, 1u, shift);
                } else {
                    // u = (u & 0x7F) + 128*(u>>7) on both operands:
                    //   (lW)(lT)             scale 1
                    //   (lW)(hT), (hW)(lT)   scale 128
                    //   (hW)(hT)             scale 16384
                    int a[4] = {0, 0, 0, 0};
                    mma_dot<false, 1, 1>(a, sw, st, pitch, row0, m0, lane, cb, db, l);
                    add_X(X, a, 1u, shift);
                    int b[4] = {0, 0, 0, 0};
                    mma_dot<false, 1, 2>(b, sw, st, pitch, row0, m0, lane, cb, db, l);
                    add_X(X, b, 128u, shift);
                    int c[4] = {0, 0, 0, 0};
                    mma_dot<false, 2, 1>(c, sw, st, pitch, row0, m0, lane, cb, db, l);
                    add_X(X, c, 128u, shift);
                    int d[4] = {0, 0, 0, 0};
                    mma_dot<false, 2, 2>(d, sw, st, pitch, row0, m0, lane, cb, db, l);
                    add_X(X, d, 16384u, shift);
                }
            }
        }
        const unsigned base = lane >> 2, tig = lane & 3;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const unsigned limb = row0 + (r < 2 ? base : base + 8);
            const unsigned m = m0 + tig * 2 + (r & 1);
            if (limb < l) {
                s_low[limb * M_COEF + m] = u64(u32(X[r]));      // bits 0..31
                s_high[limb * M_COEF + m] = u64(X[r] >> 32);    // bits 32..
            }
        }
    }
    __syncthreads();

    // ---- step 4: reuse the reference carry_and_reduce / reduce_sum ---------
    if (tid < M_COEF) {
        const unsigned m = tid;
        const unsigned coef = blockIdx.x * M_COEF + m;
        u64 low[CRT_MMA_MAXL], high[CRT_MMA_MAXL];
        for (unsigned i = 0; i < l; ++i) {
            low[i] = s_low[i * M_COEF + m];
            high[i] = s_high[i * M_COEF + m];
        }
        u32 digits[CRT_MMA_MAXL + 2];
        carry_and_reduce<1>(digits, low, high, l, plan.bits, plan.gap, plan.q);
        if (coef < plan.n) {
            const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
            for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i];
        }
    }
}
#endif  // __CUDACC__

// ===========================================================================
// 4. Host-side plan: weights in int8 layout + launchers
// ===========================================================================
struct CrtMmaPlan {
    unsigned L = 0, pitch = 0;
    std::int8_t* d_w = nullptr;      // L*pitch*4 bytes, 4 int8 per u32
    bool ok = false;

    CrtMmaPlan() = default;

    CrtMmaPlan(const std::vector<u32>& weights_host, unsigned L_, unsigned pitch_)
        : L(L_), pitch(pitch_) {
#ifdef __CUDACC__
        if (L == 0 || pitch == 0 || L > CRT_MMA_MAXL || pitch > CRT_MMA_MAXPITCH)
            throw std::runtime_error("crt_mma: L/pitch outside compiled limits");
        if (weights_host.size() != std::size_t(L) * pitch)
            throw std::runtime_error("crt_mma: weights table has wrong size");
        // Convert each u32 to four signed bytes in little-endian order. The bit
        // pattern is preserved; on the (little-endian) device the kernel can
        // read this buffer straight back as u32.
        std::vector<std::int8_t> bytes(weights_host.size() * 4);
        for (std::size_t i = 0; i < weights_host.size(); ++i) {
            const u32 v = weights_host[i];
            for (unsigned c = 0; c < 4; ++c)
                bytes[i * 4 + c] = static_cast<std::int8_t>((v >> (8u * c)) & 0xFFu);
        }
        if (cudaMalloc(reinterpret_cast<void**>(&d_w), bytes.size()) != cudaSuccess)
            throw std::runtime_error("crt_mma: cudaMalloc failed");
        if (cudaMemcpy(d_w, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_w);
            d_w = nullptr;
            throw std::runtime_error("crt_mma: weight upload failed");
        }
        ok = true;
#else
        (void)weights_host;
        throw std::runtime_error("crt_mma: constructed without CUDA");
#endif
    }

    CrtMmaPlan(const CrtMmaPlan&) = delete;
    CrtMmaPlan& operator=(const CrtMmaPlan&) = delete;

    ~CrtMmaPlan() {
#ifdef __CUDACC__
        if (d_w) cudaFree(d_w);
#endif
        d_w = nullptr;
    }

    // Build the device argument block from any CrtArgs-compatible struct
    // (e.g. gpu31::CrtArgs). Duck typing avoids an include-order dependency.
    template <class Args>
    CrtArgs make_args(const Args& args) const {
        CrtArgs ca{};
        ca.n = args.n;
        ca.limbs = args.limbs;
        ca.bits = args.bits;
        ca.count = args.count;
        ca.gap = args.gap;
        ca.primes = reinterpret_cast<const PrimeInfo*>(args.primes);
        ca.weights = reinterpret_cast<const u32*>(d_w);   // used by run_simple
        ca.q = args.q;
        return ca;
    }

    template <class Args>
    bool check(const Args& args, u32 B, const u32* residues, const u32* out) const {
        return ok && B > 0 && residues && out && args.primes && args.q &&
               args.limbs > 0 && args.limbs <= CRT_MMA_MAXL &&
               args.count + 1 <= CRT_MMA_MAXPITCH &&
               args.count + 1 == pitch && args.limbs == L;
    }

    // ---- PRIMARY: exact u8 tensor-core MMA (fast, recommended) -------------
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const unsigned before_u64 = pitch * (L + M_COEF);
        const unsigned aligned_u64 = (before_u64 + 1u) & ~1u;
        const std::size_t smem =
            std::size_t(aligned_u64) * sizeof(u32) +       // sw + st (+pad)
            2u * std::size_t(L) * M_COEF * sizeof(u64);    // s_low + s_high
        crt_mma_kernel<0><<<dim3((ca.n + M_COEF - 1) / M_COEF, 1, B),
                            MMA_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;   // never called in non-CUDA builds
#endif
    }

    // ---- EXACT s8 tensor-core MMA with 4-way compensation -----------------
    template <class Args>
    cudaError_t run_s8(const u32* residues, u32* out, const Args& args, u32 B,
                       cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const unsigned before_u64 = pitch * (L + M_COEF);
        const unsigned aligned_u64 = (before_u64 + 1u) & ~1u;
        const std::size_t smem =
            std::size_t(aligned_u64) * sizeof(u32) +       // sw + st (+pad)
            2u * std::size_t(L) * M_COEF * sizeof(u64);    // s_low + s_high
        crt_mma_kernel<1><<<dim3((ca.n + M_COEF - 1) / M_COEF, 1, B),
                            MMA_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;   // never called in non-CUDA builds
#endif
    }

    // ---- CUDA-core reference (byte-identical to reconstruct_kernel) -------
    template <class Args>
    cudaError_t run_simple(const u32* residues, u32* out, const Args& args, u32 B,
                           cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_simple_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;   // never called in non-CUDA builds
#endif
    }
};

}  // namespace crt_mma

#undef CRTMMA_FHD
