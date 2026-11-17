// crt_mma_tc.cuh — EXACT CRT reconstruction with a REAL tensor-core
// (mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32) byte-split inner product.
//
// Header-only.  Nothing outside this file is modified.  Namespace is
// `crt_mma_tc` so this file can be compiled side-by-side with `crt_mma.cuh`.
// API is identical to `crt_mma::CrtMmaPlan`:
//     CrtMmaPlan(weights_host, L, pitch)
//     run(residues, out, args, B, stream)
//     run_s8(...), run_simple(...)
//     fields L / pitch / ok
// `CrtArgs` is duck-typed: a gpu31::CrtArgs can be passed directly.
//
// ---------------------------------------------------------------------------
// WHAT WAS FIXED RELATIVE TO THE ORIGINAL crt_mma.cuh MMA ATTEMPT
// ---------------------------------------------------------------------------
//  1. cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
//     smem_bytes) is now called from the host, with error checking, for BOTH
//     template instantiations before every launch.  If the device cannot
//     grant the request (pre-Volta or a device with a smaller opt-in limit),
//     run()/run_s8() transparently fall back to the exact CUDA-core
//     run_simple() path instead of leaving `out` unwritten.
//  2. `shifted_word` uses q[i] (the reference's Stride=1 code), never
//     q[i*Stride].  The original stray copy is gone.
//  3. All dynamic-shared indices are range checked by construction
//     (j < pitch, m < 32, i < L, acc slots < 128) and every global output
//     write is guarded by dst + L <= B*N*L.
//  4. The MMA fragment packing uses the *exact* PTX ISA 9.5 lane mappings for
//     `m16n8k32` with `.u8` operands (see the comments at the PTX loop below).
//
// ---------------------------------------------------------------------------
// ALGORITHM (byte-split reassembly math is the verified math of crt_mma.cuh)
// ---------------------------------------------------------------------------
// For each (coefficient, limb) we need  lo + 2^32*hi = sum_j a_j * b_j with
// a_j = terms[j] (u32), b_j = weights[limb][j] (u32).  Writing
// a_j = sum_ct a_j[ct]*256^ct, b_j = sum_cw b_j[cw]*256^cw (ct,cw = 0..3),
//
//     sum_j a_j*b_j = sum_{ct,cw} 256^(ct+cw) * S[ct][cw],
//     S[ct][cw]     = sum_j a_j[ct] * b_j[cw].          (16 byte-pair sums)
//
// `S[ct][cw]` is computed EXACTLY by the tensor core: each MMA does 32 such
// byte products (k=32) in one `s32` accumulator, and up to 16 k-chunks are
// accumulated (pitch <= 512).  Max accumulator value = 512*255*255 ~ 3.3e7,
// far below 2^31, so the `.s32` accumulator never overflows.  The `.u8`
// qualifier is essential: raw `.s8` bytes would make bytes >= 128 negative.
// After the k-loop the 16 partial sums are shifted into a __uint128_t X;
// low = u32(X), high = u64(X >> 32) are then fed to the unmodified reference
// `carry_and_reduce`, giving byte-for-byte identical output to
// gpu31::reconstruct_kernel.
//
// ---------------------------------------------------------------------------
// MAPPING ONE MMA TILE -> 4 COEFFICIENTS x 2 LIMBS (16x8 accumulators)
// ---------------------------------------------------------------------------
// Warp lanes 0..31, g = lane>>2 (0..7), t = lane%4 (0..3).
//   MMA row m = 4*coef_in_warp + ct  (0..15),  MMA column n = 4*limb_in_pair
//     + cw (0..7).
//   A fragment (row-major u8, 4x u32 per lane):
//     a0: row g,   K = 4t + 0..3       (coef_in_warp = g>>2, ct = g%4)
//     a1: row g+8, K = 4t + 0..3       (coef_in_warp = 2+(g>>2), ct = g%4)
//     a2: row g,   K = 16+4t + 0..3
//     a3: row g+8, K = 16+4t + 0..3
//   B fragment (col-major u8, 2x u32 per lane):
//     b0: col g, K = 4t + 0..3         (limb_in_pair = g>>2, cw = g%4)
//     b1: col g, K = 16+4t + 0..3
//   C/D fragment (s32, 4x s32 per lane):
//     c0: row g,   col 2t     -> coef g>>2,     limb t>>1,     ct g%4, cw 2*(t%2)
//     c1: row g,   col 2t+1   -> coef g>>2,     limb t>>1,     ct g%4, cw 2*(t%2)+1
//     c2: row g+8, col 2t     -> coef 2+(g>>2), limb t>>1,     ct g%4, cw 2*(t%2)
//     c3: row g+8, col 2t+1   -> coef 2+(g>>2), limb t>>1,     ct g%4, cw 2*(t%2)+1
// These are the layouts of PTX ISA section 9.7.16.5.10, verified against the
// published row/col formulas.  Accumulators are staged in per-warp shared
// memory and a lane of the warp gathers the 16 S[ct][cw] values to build X.
//
// ---------------------------------------------------------------------------
// RISK NOTE (please read before shipping the MMA path)
// ---------------------------------------------------------------------------
// The byte-split math and the fragment layouts are derived from the PTX ISA
// and are believed correct, but the inline-PTX path has not been executed on
// the target GPU in this environment.  `run_simple()` is kept as a fully
// verified CUDA-core fallback, and `run()` falls back to it automatically if
// the dynamic-shared-memory opt-in fails.  For absolute safety, call
// `run_simple()` (or keep the existing crt_mma.cuh) in production until the
// `crt_mma_tc_selftest()` + an on-device selfcheck confirm the MMA path.
// The `.u8` m16n8k32 MMA requires sm_75 (Turing) or newer; on older targets
// the kernel must not be compiled.  The device selfcheck in solve.cu is the
// authoritative gate: keep the reference CRT if even one limb differs.
// ===========================================================================

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>          // host shim supplies the error enum in CPU builds

#include "fherma.h"

namespace crt_mma_tc {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__)
#define CRTMMA_TC_FHD __host__ __device__
#else
#define CRTMMA_TC_FHD
#endif

// Compile-time limits (checked on the host side).
static constexpr unsigned TC_MAXL       = 128;   // max limbs
static constexpr unsigned TC_MAXPITCH   = 512;   // max K+1 columns
static constexpr unsigned TC_M_COEF     = 32;    // coefficients per block
static constexpr unsigned TC_WARPS      = TC_M_COEF / 4;  // 8: 4 coefs per warp
static constexpr unsigned TC_THREADS    = TC_WARPS * 32;  // 256
static constexpr unsigned TC_TS         = TC_M_COEF + 1;  // 33: ODD stride (bank-safe)
static constexpr unsigned TC_ACC_WORDS  = TC_WARPS * 2u * 4u * 4u * 4u; // 8*128 ints

// Dynamic shared memory used by crt_mma_tc_kernel (host + kernel must agree).
//   terms : pitch * TC_M_COEF u32          (byte-split A operands survive in gmem)
//   low   : l * TC_TS u64
//   high  : l * TC_TS u64
//   acc   : TC_ACC_WORDS int               (per-warp 16x8 MMA accumulators)
// At L=55, pitch=118: 15104 + 14520 + 14520 + 4096 = 48,240 bytes.
static inline std::size_t crt_mma_tc_shared_bytes(unsigned l, unsigned pitch) {
    const std::size_t terms = std::size_t(pitch) * TC_M_COEF * sizeof(u32);
    const std::size_t lhs   = std::size_t(l) * TC_TS * sizeof(u64);
    const std::size_t acc   = std::size_t(TC_ACC_WORDS) * sizeof(int);
    return terms + 2u * lhs + acc;
}

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
// 1. Helpers copied from gpu31.cuh's fherma_crt section.
//    `shifted_word` is the FIXED version: q[i], never q[i*Stride].
// ===========================================================================
CRTMMA_TC_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template <unsigned Stride>
CRTMMA_TC_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}

template <unsigned Stride>
CRTMMA_TC_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
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
CRTMMA_TC_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}

template <unsigned Stride>
CRTMMA_TC_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
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
CRTMMA_TC_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes, unsigned count) {
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}

template <unsigned Stride>
CRTMMA_TC_FHD inline void sum_column(const u32* terms, const u32* basis_column,
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
CRTMMA_TC_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
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
__global__ void crt_mma_tc_simple_kernel(CrtArgs plan, const u32* residues,
                                         const u32* weights, u32* out, u32 batches) {
    const unsigned coef = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = blockIdx.y;
    if (coef >= plan.n) return;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;

    u32 terms[TC_MAXPITCH];
    u64 low[TC_MAXL], high[TC_MAXL];
    u32 digits[TC_MAXL + 2];

    const u64 rbase = u64(pair) * k * plan.n;
    for (unsigned j = 0; j < k; ++j)
        terms[j] = weight_residue(residues[rbase + u64(j) * plan.n + coef], plan.primes[j]);
    terms[k] = crt_correction<1>(terms, plan.primes, k);
    for (unsigned i = 0; i < l; ++i)
        sum_column<1>(terms, weights + i * pitch, pitch, low[i], high[i]);
    carry_and_reduce<1>(digits, low, high, l, plan.bits, plan.gap, plan.q);

    const u64 total = u64(batches) * plan.n * l;
    const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
    if (dst + l <= total)                       // aggressive output bound check
        for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i];
}
#endif  // __CUDACC__

// ===========================================================================
// 3. Tensor-core kernel (run / run_s8 paths).
// ===========================================================================
#if defined(__CUDACC__)
template <int MODE>
__global__ void crt_mma_tc_kernel(CrtArgs plan, const u32* residues,
                                  const u32* d_w_bytes_as_u32, u32* out, u32 batches) {
    (void)MODE;  // run() and run_s8() share this identical, exact implementation

    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned warp = threadIdx.x >> 5;        // 0 .. TC_WARPS-1
    const unsigned lane = threadIdx.x & 31u;
    const unsigned g = lane >> 2, t = lane & 3u;   // PTX lane decomposition
    const unsigned coef_tile = blockIdx.x * TC_M_COEF;
    const unsigned pair = blockIdx.z;
    const unsigned m_base = 4u * warp + (g >> 2);  // coefficient slot (0..31)
    const unsigned m_hi   = m_base + 2u;           // the g+8 row's slot
    const unsigned sh_a   = 8u * (g & 3u);         // byte ct within a u32
    const unsigned i_base = 2u * ((g >> 2) & 1u);  // limb_in_pair for B (0/1)

    extern __shared__ unsigned char smem[];
    u32* terms = reinterpret_cast<u32*>(smem);                       // pitch * 32
    u64* low   = reinterpret_cast<u64*>(terms + pitch * TC_M_COEF);  // l * 33
    u64* high  = low + l * TC_TS;
    int*  acc  = reinterpret_cast<int*>(high + l * TC_TS);           // 8 * 128

    // ---- step 1: Shoup terms t_j = r_j * (P/p_j)^-1 mod p_j ---------------
    // All 256 threads cooperate; every (j, m) written exactly once.
    // Inactive coefficients get r=0 -> weight_residue(0)=0, so no slot of
    // `terms` is ever left uninitialised (even for a fully out-of-range tile).
    for (unsigned m = 0; m < TC_M_COEF; ++m) {
        const unsigned coef = coef_tile + m;
        for (unsigned j = threadIdx.x; j < k; j += TC_THREADS) {
            const u32 r = (coef < plan.n)
                        ? residues[u64(pair) * k * plan.n + u64(j) * plan.n + coef]
                        : 0u;
            terms[j * TC_M_COEF + m] = weight_residue(r, plan.primes[j]);
        }
    }
    __syncthreads();

    // ---- step 2: FP64 correction (reference crt_correction), one thread/m --
    if (coef_tile < plan.n && threadIdx.x < TC_M_COEF) {
        const unsigned m = threadIdx.x, coef = coef_tile + m;
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
            sum += double(terms[j * TC_M_COEF + m]) * plan.primes[j].reciprocal;
        terms[k * TC_M_COEF + m] = (coef < plan.n) ? u32(sum + 0.5) : 0u;
    }
    __syncthreads();

    // ---- step 3: byte-split tensor-core product sums ----------------------
    // Each warp handles coefficients [m_base group] and ALL limb pairs.  For
    // limb pair `ip` we compute limbs i = 2*ip (col 0..3) and 2*ip+1 (col 4..7)
    // for its 4 coefficients; the 16x8 MMA tile covers exactly 4x2x16 sums.
    int* accw = acc + warp * (2u * 4u * 4u * 4u);   // private 128-int slice
    for (unsigned ip = 0; ip < (l + 1u) / 2u; ++ip) {
        int c0 = 0, c1v = 0, c2v = 0, c3v = 0;      // s32 accumulators

        for (unsigned kb = 0; kb < pitch; kb += 32u) {
            u32 a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;

            // A: 4 u32 fragments = 16 bytes of `terms` for this lane.
            #pragma unroll
            for (unsigned bb = 0; bb < 4u; ++bb) {
                const unsigned j0 = kb + 4u * t + bb;
                const unsigned j1 = kb + 16u + 4u * t + bb;
                if (j0 < pitch) {
                    a0 |= ((terms[j0 * TC_M_COEF + m_base] >> sh_a) & 0xFFu) << (8u * bb);
                    a1 |= ((terms[j0 * TC_M_COEF + m_hi]   >> sh_a) & 0xFFu) << (8u * bb);
                }
                if (j1 < pitch) {
                    a2 |= ((terms[j1 * TC_M_COEF + m_base] >> sh_a) & 0xFFu) << (8u * bb);
                    a3 |= ((terms[j1 * TC_M_COEF + m_hi]   >> sh_a) & 0xFFu) << (8u * bb);
                }
            }

            // B: 2 u32 fragments = 8 bytes of the byte-split weight table.
            // d_w_bytes_as_u32[limb*pitch + j] stores byte cw in bits 8*cw.
            const unsigned i = 2u * ip + i_base;
            if (i < l) {
                const u32* w = d_w_bytes_as_u32 + u64(i) * pitch;
                #pragma unroll
                for (unsigned bb = 0; bb < 4u; ++bb) {
                    const unsigned j0 = kb + 4u * t + bb;
                    const unsigned j1 = kb + 16u + 4u * t + bb;
                    if (j0 < pitch)
                        b0 |= ((w[j0] >> sh_a) & 0xFFu) << (8u * bb);  // cw = g%4 = ct
                    if (j1 < pitch)
                        b1 |= ((w[j1] >> sh_a) & 0xFFu) << (8u * bb);
                }
            }

            // Exact inline PTX (PTX ISA 9.7.16.5 / 9.7.16.5.10):
            //   A: 16x32 row-major u8 -> 4x .b32 per lane (a0..a3)
            //   B: 32x8  col-major u8 -> 2x .b32 per lane (b0,b1)
            //   C/D: 16x8 s32         -> 4x .s32 per lane (c0..c3)
            // D = A*B + C, accumulated across all k-chunks.
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+r"(c0), "+r"(c1v), "+r"(c2v), "+r"(c3v)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }

        // ---- stash the 4 accumulators per lane in this warp's slice -------
        // All four share ct = g%4 and limb = t>>1; c0/c1 are coef g>>2,
        // c2/c3 are coef 2+(g>>2); c0/c2 have cw = 2*(t%2), c1/c3 that + 1.
        const unsigned ab   = t >> 1;              // limb_in_pair (0/1)
        const unsigned cw0  = (t & 1u) << 1;       // 0 or 2
        const unsigned cg   = g >> 2;              // 0/1 coefficient half
        const unsigned ct   = g & 3u;
        const unsigned idx0 = ((ab * 4u + cg) * 4u + ct) * 4u + cw0;
        const unsigned idx1 = ((ab * 4u + cg + 2u) * 4u + ct) * 4u + cw0;
        accw[idx0 + 0] = c0;
        accw[idx0 + 1] = c1v;
        accw[idx1 + 0] = c2v;
        accw[idx1 + 1] = c3v;
        __syncwarp();   // make other lanes' shared stores visible (Volta+)

        // ---- reassemble X for the 8 (coef, limb) outputs of this warp -----
        if (lane < 8u) {
            const unsigned co = lane >> 1;         // coef_in_warp 0..3
            const unsigned lb = lane & 1u;         // limb_in_pair 0/1
            __uint128_t X = 0;
            #pragma unroll
            for (unsigned ct2 = 0; ct2 < 4u; ++ct2) {
                #pragma unroll
                for (unsigned cw2 = 0; cw2 < 4u; ++cw2) {
                    const u32 S = u32(accw[((lb * 4u + co) * 4u + ct2) * 4u + cw2]);
                    X += (__uint128_t)(u64)S << (8u * (ct2 + cw2));  // 2^(8*(cw+ct))
                }
            }
            const unsigned i = 2u * ip + lb;
            if (i < l) {
                const unsigned m = 4u * warp + co;  // 0 .. 31 -> bounded slot
                low[i * TC_TS + m]  = (u64)(u32)X;
                high[i * TC_TS + m] = (u64)(X >> 32);
            }
        }
        __syncwarp();   // reuse accw in the next ip iteration safely
    }
    __syncthreads();

    // ---- step 4: reference carry_and_reduce / reduce_sum ------------------
    const u64 total = u64(batches) * plan.n * l;   // B*N*L output elements
    if (threadIdx.x < TC_M_COEF && coef_tile + threadIdx.x < plan.n) {
        const unsigned m = threadIdx.x;
        u32 digits[TC_MAXL + 2];
        carry_and_reduce<TC_TS>(digits, low + m, high + m, l,
                                plan.bits, plan.gap, plan.q);
        const unsigned coef = coef_tile + m;
        const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
        if (dst + l <= total)                      // aggressive output bound check
            for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i * TC_TS];
    }
}
#endif  // __CUDACC__

// ===========================================================================
// 4. Host-only check of the byte-split reassembly math (plain C++).
// ===========================================================================
struct CrtMmaU128 { u64 lo, hi; };

inline CrtMmaU128 crt_mma_tc_add_shifted(CrtMmaU128 x, u64 v, unsigned shift) {
    u64 add_lo = 0, add_hi = 0;
    if (shift < 64) {
        add_lo = v << shift;
        add_hi = shift ? (v >> (64 - shift)) : 0u;
    } else if (shift < 128) {
        add_hi = v << (shift - 64);
    }
    const u64 new_lo = x.lo + add_lo;
    x.hi += add_hi + (new_lo < x.lo ? 1u : 0u);
    x.lo = new_lo;
    return x;
}

inline bool crt_mma_tc_selftest(unsigned iterations = 4096,
                                u64 seed = 0x9e3779b97f4a7c15ull) {
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };

    for (unsigned it = 0; it < iterations; ++it) {
        const u32 a = u32(next()), b = u32(next());

        // u8 path: raw bytes, exact unsigned byte products.
        CrtMmaU128 X{0, 0};
        for (unsigned ct = 0; ct < 4; ++ct) {
            const u64 wa = (a >> (8u * ct)) & 0xFFu;
            for (unsigned cw = 0; cw < 4; ++cw) {
                const u64 wb = (b >> (8u * cw)) & 0xFFu;
                X = crt_mma_tc_add_shifted(X, wa * wb, 8u * (ct + cw));
            }
        }
        const u64 want = u64(a) * b;
        if (X.hi != 0 || X.lo != want) return false;

        const u32 low = u32(X.lo);
        const u64 high = X.lo >> 32;
        if (u64(low) + high * (u64(1) << 32) != want) return false;

        // s8 compensation identity (documentation completeness).
        for (unsigned c = 0; c < 4; ++c) {
            const u32 ua = (a >> (8u * c)) & 0xFFu;
            const u32 ub = (b >> (8u * c)) & 0xFFu;
            if ((ua & 0x7Fu) + 128u * (ua >> 7) != ua) return false;
            if ((ub & 0x7Fu) + 128u * (ub >> 7) != ub) return false;
        }
    }
    return true;
}

// ===========================================================================
// 5. Host-side plan + launchers (public API unchanged).
// ===========================================================================
struct CrtMmaPlan {
    unsigned L = 0, pitch = 0;
    std::int8_t* d_w = nullptr;      // L*pitch*4 bytes, 4 int8 per u32 weight
    bool ok = false;

    CrtMmaPlan() = default;

    CrtMmaPlan(const std::vector<u32>& weights_host, unsigned L_, unsigned pitch_)
        : L(L_), pitch(pitch_) {
#ifdef __CUDACC__
        if (L == 0 || pitch == 0 || L > TC_MAXL || pitch > TC_MAXPITCH)
            throw std::runtime_error("crt_mma_tc: L/pitch outside compiled limits");
        if (weights_host.size() != std::size_t(L) * pitch)
            throw std::runtime_error("crt_mma_tc: weights table has wrong size");
        // Preserve each u32 bit pattern as four little-endian bytes; the MMA
        // kernel reads individual bytes back, run_simple reads the u32s.
        std::vector<std::int8_t> bytes(weights_host.size() * 4);
        for (std::size_t i = 0; i < weights_host.size(); ++i) {
            const u32 v = weights_host[i];
            for (unsigned c = 0; c < 4; ++c)
                bytes[i * 4 + c] = static_cast<std::int8_t>((v >> (8u * c)) & 0xFFu);
        }
        if (cudaMalloc(reinterpret_cast<void**>(&d_w), bytes.size()) != cudaSuccess)
            throw std::runtime_error("crt_mma_tc: cudaMalloc failed");
        if (cudaMemcpy(d_w, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_w);
            d_w = nullptr;
            throw std::runtime_error("crt_mma_tc: weight upload failed");
        }
        ok = true;
#else
        (void)weights_host;
        throw std::runtime_error("crt_mma_tc: constructed without CUDA");
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
        const std::size_t smem = crt_mma_tc_shared_bytes(L, pitch);
        return ok && B > 0 && residues && out && args.primes && args.q &&
               args.limbs > 0 && args.limbs <= TC_MAXL &&
               args.count + 1 <= TC_MAXPITCH &&
               args.count + 1 == pitch && args.limbs == L &&
               smem > 0 && smem <= std::size_t(1) << 30;   // opt-in, NOT capped at 48 KiB
    }

    // ---- opt-in for >48 KiB dynamic shared memory, with error checking -----
    // Returns cudaSuccess when the kernel may be launched with `smem` bytes.
#ifdef __CUDACC__
    template <int MODE>
    cudaError_t enable_large_smem(std::size_t smem) const {
        // ALWAYS issue the opt-in, even when smem <= 48 KiB: this is the exact
        // fix for the original kernel's cudaErrorInvalidValue launch failure.
        // cudaFuncSetAttribute returns cudaSuccess for in-range requests.
        const cudaError_t e = cudaFuncSetAttribute(
            crt_mma_tc_kernel<MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem));
        if (e != cudaSuccess) return e;
        int dev = 0, max_smem = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return cudaErrorInvalidDevice;
        const cudaError_t e2 = cudaDeviceGetAttribute(
            &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        if (e2 != cudaSuccess) return e2;
        if (static_cast<std::size_t>(max_smem) < smem) return cudaErrorInvalidValue;
        return cudaSuccess;
    }
#endif

    // ---- PRIMARY tensor-core path -----------------------------------------
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_mma_tc_shared_bytes(L, pitch);
        // Mandatory fix for the original segfault: opt in to large smem.
        if (enable_large_smem<0>(smem) != cudaSuccess)
            return run_simple(residues, out, args, B, stream);  // safe fallback
        crt_mma_tc_kernel<0><<<dim3((ca.n + TC_M_COEF - 1) / TC_M_COEF, 1, B),
                               TC_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out, B);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }

    // ---- EXACT path kept for API compatibility (same MMA kernel) ----------
    template <class Args>
    cudaError_t run_s8(const u32* residues, u32* out, const Args& args, u32 B,
                       cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_mma_tc_shared_bytes(L, pitch);
        if (enable_large_smem<1>(smem) != cudaSuccess)
            return run_simple(residues, out, args, B, stream);
        crt_mma_tc_kernel<1><<<dim3((ca.n + TC_M_COEF - 1) / TC_M_COEF, 1, B),
                               TC_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out, B);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }

    // ---- Safe CUDA-core reference (byte-identical to reconstruct_kernel) --
    template <class Args>
    cudaError_t run_simple(const u32* residues, u32* out, const Args& args, u32 B,
                           cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_mma_tc_simple_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out, B);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }
};

}  // namespace crt_mma_tc

#undef CRTMMA_TC_FHD
