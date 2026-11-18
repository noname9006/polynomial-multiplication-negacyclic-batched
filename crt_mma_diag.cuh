// crt_mma_diag.cuh — EXACT CRT reconstruction with a REAL tensor-core
// (mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32) byte-split inner product,
// PLUS first-call diagnostics that are guaranteed to reach the FHERMA stderr log.
//
// Header-only, self-contained, namespace `crt_mma_diag`. NOTHING outside this
// file is modified; integrate by including it after gpu31.cuh and swapping the
// plan type. API mirrors crt_mma::CrtMmaPlan:
//     CrtMmaPlan(weights_host, L, pitch)
//     run(residues, out, args, B, stream)        <-- duck-typed `args`
//     run_simple(residues, out, args, B, stream) <-- byte-identical CUDA-core fallback
//     run_s8(...)  (alias of run)
//     fields L / pitch / ok
// A gpu31::CrtArgs can be passed straight to run()/run_simple().
//
// ===========================================================================
// WHY THIS FILE EXISTS / WHAT WAS FIXED
// ===========================================================================
// Two earlier tensor-core attempts failed. A roofline analysis showed the
// reported 24 s was the OLD 62-bit fallback (adding the MMA plan made
// gpu31_selfcheck fail, so gpu31_ok=false and solve.cu ran gpu_multiply): the
// MMA kernel was never actually benchmarked. This file makes the MMA path
// observable: on the FIRST call to run() it prints a fixed set of [CRTMMA]
// lines to stderr (which earlier runs proved reaches the FHERMA log), including
// the cudaFuncSetAttribute error, the launch error, and a one-off timed kernel
// duration. Even a failed MMA run now becomes a fully informative log entry.
//
// Two concrete correctness bugs in crt_mma_tc.cuh were fixed here:
//   BUG 1 (wrong B operand): its B-fragment limb index was
//          i_base = 2*(g>>2)   // {0,2}  -- WRONG
//     For m16n8k32 the B fragment column is n = groupID = g (0..7). Mapping the
//     8 columns as n = 4*limb_in_pair + cw gives limb_in_pair = g>>2 (0/1), so
//     the correct limb is i = 2*ip + (g>>2). The old code loaded limb 2*ip+2
//     for the upper half of the columns, duplicating an even limb's weights
//     into the odd-limb columns -> wrong CRT -> selfcheck fails -> 24 s fallback.
//     Fixed: i_base = (g>>2).
//   BUG 2 (stack corruption -> Segmentation fault): its step 4 called
//          carry_and_reduce<TC_TS>(digits, ..., l, ...)   // TC_TS = 33
//     with `u32 digits[TC_MAXL+2]` (a LOCAL, ~130-word array). The Stride=33
//     template writes digits[i*33], i up to l+1 = 56 -> digits[1848]: a massive
//     out-of-bounds stack write. This is the likely source of the segfault.
//     Fixed: carry_and_reduce_mma() reads strided low/high from shared memory
//     but writes a CONTIGUOUS local digits[] (Stride-1 reduce_sum).
//
// Fragment layouts are taken verbatim from PTX ISA 9.5 sec. 9.7.16.5.10
// ("Matrix Fragments for mma.m16n8k32", .u8/.s8) and verified against the
// published row/col formulas:
//   groupID = lane>>2, t = lane%4
//   A (row-major, a0..a3): a0 row=g col=4t+{0..3}; a1 row=g+8 col=4t+{0..3};
//                          a2 row=g col=4t+{0..3}+16; a3 row=g+8 col=4t+{0..3}+16.
//   B (col-major, b0..b1): b0 col=g row=4t+{0..3}; b1 col=g row=4t+{0..3}+16.
//   C/D (s32, c0..c3): c0 row=g col=2t; c1 row=g col=2t+1;
//                      c2 row=g+8 col=2t; c3 row=g+8 col=2t+1.
//
// One MMA tile -> 4 coefficients x 2 limbs. MMA row m = 4*coef + ct and column
// n = 4*limb_in_pair + cw (ct,cw = byte index 0..3). After the k-loop all 16
// byte-pair sums S[ct][cw] are stashed, then X = sum S[ct][cw]<<8*(ct+cw) is
// reassembled ONCE per (coef, limb), giving low=X mod 2^32, high=(X>>32) mod
// 2^64 fed to the unmodified reference carry_and_reduce + gap-fold reduction.
// ===========================================================================

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>          // shim supplies the error enum in CPU builds

#include "fherma.h"

namespace crt_mma_diag {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__)
#define CRTMD_FHD __host__ __device__
#else
#define CRTMD_FHD
#endif

// Compile-time limits (checked on the host side).
static constexpr unsigned MD_MAXL        = 128;              // max limbs
static constexpr unsigned MD_MAXPITCH    = 512;              // max K+1 columns
static constexpr unsigned MD_M_COEF      = 32;               // coefficients per block
static constexpr unsigned MD_WARPS       = MD_M_COEF / 4;    // 8 (4 coefs / warp)
static constexpr unsigned MD_THREADS     = MD_WARPS * 32;    // 256
static constexpr unsigned MD_TS          = MD_M_COEF + 1;    // 33, ODD (bank-safe)
static constexpr unsigned MD_ACC_WORDS   = MD_WARPS * 2u * 4u * 4u * 4u; // 8*128 = 1024
static constexpr std::size_t MD_SMEM_CAP = 80u * 1024u;      // design cap (Blackwell opt-in ~99 KiB)

// Dynamic shared memory used by crt_mma_diag_kernel (host + kernel must agree).
//   terms : pitch * MD_M_COEF u32        (byte-split A operands survive in gmem)
//   low   : l * MD_TS u64
//   high  : l * MD_TS u64
//   acc   : MD_ACC_WORDS int              (per-warp 16x8 MMA accumulator staging)
// At L=55, pitch=118: 15104 + 14520 + 14520 + 4096 = 48,240 bytes (< 48 KiB even).
static inline std::size_t crt_mma_diag_shared_bytes(unsigned l, unsigned pitch) {
    const std::size_t terms = std::size_t(pitch) * MD_M_COEF * sizeof(u32);
    const std::size_t lhs   = std::size_t(l) * MD_TS * sizeof(u64);
    const std::size_t acc   = std::size_t(MD_ACC_WORDS) * sizeof(int);
    return terms + 2u * lhs + acc;
}

// Device-side mirror of the CRT structs (layout-identical to gpu31's).
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
// 1. Helpers copied from gpu31.cuh's fherma_crt section (shifted_word uses
//    q[i], the reference's Stride=1 code -- never q[i*Stride]).
// ===========================================================================
CRTMD_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template <unsigned Stride>
CRTMD_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}

template <unsigned Stride>
CRTMD_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
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
CRTMD_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}

template <unsigned Stride>
CRTMD_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
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
CRTMD_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes, unsigned count) {
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}

template <unsigned Stride>
CRTMD_FHD inline void sum_column(const u32* terms, const u32* basis_column,
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
CRTMD_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
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

// FIX for BUG 2: low/high are strided in shared memory (MD_TS), but `digits` is
// a CONTIGUOUS local array. reduce_sum therefore uses Stride=1.
CRTMD_FHD inline void carry_and_reduce_mma(u32* digits, const u64* low, const u64* high,
                                           unsigned l, unsigned bits, u32 gap, const u32* q,
                                           unsigned low_stride) {
    u64 carry = 0, prior_high = 0;
    for (unsigned i = 0; i < l; ++i) {
        carry += low[i * low_stride] + prior_high;
        digits[i] = u32(carry);
        carry >>= 32;
        prior_high = high[i * low_stride];
    }
    carry += prior_high;
    digits[l] = u32(carry);
    digits[l + 1] = u32(carry >> 32);
    reduce_sum<1>(digits, l, bits, gap, q);
}

// 128-bit accumulator for byte-split reassembly (no compiler-specific type needed).
struct CrtMmaU128 { u64 lo, hi; };

// Add v << sh (sh in [0,63]) into a 128-bit value. v <= 2^23 and sh <= 48 here,
// so v<<sh may exceed 64 bits and must be split into lo/hi parts.
CRTMD_FHD inline void u128_add_shift(CrtMmaU128& x, u64 v, unsigned sh) {
    const u64 add_lo = (sh < 64) ? (v << sh) : 0ull;
    const u64 add_hi = (sh == 0) ? 0ull : ((sh < 64) ? (v >> (64 - sh)) : 0ull);
    const u64 nl = x.lo + add_lo;
    x.hi += add_hi + (nl < x.lo ? 1ull : 0ull);
    x.lo = nl;
}

// ===========================================================================
// 2. CUDA-core reference kernel (run_simple path) — byte-identical to gpu31.
// ===========================================================================
#if defined(__CUDACC__)
__global__ void crt_mma_diag_simple_kernel(CrtArgs plan, const u32* residues,
                                           const u32* weights, u32* out, u32 batches) {
    const unsigned coef = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = blockIdx.y;
    if (coef >= plan.n) return;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;

    u32 terms[MD_MAXPITCH];
    u64 low[MD_MAXL], high[MD_MAXL];
    u32 digits[MD_MAXL + 2];

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

// ===========================================================================
// 3. Tensor-core kernel (run / run_s8 paths).
// ===========================================================================
template <int MODE>
__global__ void crt_mma_diag_kernel(CrtArgs plan, const u32* residues,
                                    const u32* d_w_bytes_as_u32, u32* out, u32 batches) {
    (void)MODE;  // run() and run_s8() share this identical, exact implementation

    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned warp = threadIdx.x >> 5;              // 0 .. MD_WARPS-1
    const unsigned lane = threadIdx.x & 31u;
    const unsigned g = lane >> 2, t = lane & 3u;         // PTX lane decomposition
    const unsigned coef_tile = blockIdx.x * MD_M_COEF;
    const unsigned pair = blockIdx.z;
    // A rows: g -> coef g>>2 (m_base), g+8 -> coef 2+(g>>2) (m_hi); ct = g%4.
    const unsigned m_base = 4u * warp + (g >> 2);
    const unsigned m_hi = m_base + 2u;
    const unsigned sh_a = 8u * (g & 3u);                 // byte ct within a u32
    // FIX for BUG 1: B column n = g -> limb_in_pair = g>>2 (0/1), NOT 2*(g>>2).
    const unsigned i_base = (g >> 2);

    extern __shared__ unsigned char smem[];
    u32* terms = reinterpret_cast<u32*>(smem);                       // pitch * 32
    u64* low = reinterpret_cast<u64*>(terms + pitch * MD_M_COEF);    // l * 33
    u64* high = low + l * MD_TS;
    int* acc = reinterpret_cast<int*>(high + l * MD_TS);             // 8 * 128

    // ---- step 1: Shoup terms t_j = r_j * (P/p_j)^-1 mod p_j ---------------
    // All 256 threads cooperate; every (j, m) written exactly once. Inactive
    // coefficients get r=0 -> weight_residue(0)=0, so no slot is uninitialised.
    for (unsigned m = 0; m < MD_M_COEF; ++m) {
        const unsigned coef = coef_tile + m;
        for (unsigned j = threadIdx.x; j < k; j += MD_THREADS) {
            const u32 r = (coef < plan.n)
                        ? residues[u64(pair) * k * plan.n + u64(j) * plan.n + coef]
                        : 0u;
            terms[j * MD_M_COEF + m] = weight_residue(r, plan.primes[j]);
        }
    }
    __syncthreads();

    // ---- step 2: FP64 correction (reference crt_correction), one thread/m --
    if (coef_tile < plan.n && threadIdx.x < MD_M_COEF) {
        const unsigned m = threadIdx.x, coef = coef_tile + m;
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
            sum += double(terms[j * MD_M_COEF + m]) * plan.primes[j].reciprocal;
        terms[k * MD_M_COEF + m] = (coef < plan.n) ? u32(sum + 0.5) : 0u;
    }
    __syncthreads();

    // ---- step 3: byte-split tensor-core product sums ----------------------
    // Each warp handles 4 coefficients and ALL limb pairs. For limb pair ip the
    // 16x8 MMA tile covers 2 limbs x 4 byte positions for those 4 coefficients.
    int* accw = acc + warp * (2u * 4u * 4u * 4u);   // private 128-int slice
    for (unsigned ip = 0; ip < (l + 1u) / 2u; ++ip) {
        int c0 = 0, c1v = 0, c2v = 0, c3v = 0;      // s32 accumulators

        // K = pitch (118) padded to 128: four m16n8k32 chunks. Accumulate ALL
        // byte-pair dot products across every j FIRST; reassemble only below.
        for (unsigned kb = 0; kb < pitch; kb += 32u) {
            u32 a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;

            // A: 4 u32 fragments = 16 bytes of `terms` for this lane.
            #pragma unroll
            for (unsigned bb = 0; bb < 4u; ++bb) {
                const unsigned j0 = kb + 4u * t + bb;
                const unsigned j1 = kb + 16u + 4u * t + bb;
                if (j0 < pitch) {
                    a0 |= ((terms[j0 * MD_M_COEF + m_base] >> sh_a) & 0xFFu) << (8u * bb);
                    a1 |= ((terms[j0 * MD_M_COEF + m_hi]   >> sh_a) & 0xFFu) << (8u * bb);
                }
                if (j1 < pitch) {
                    a2 |= ((terms[j1 * MD_M_COEF + m_base] >> sh_a) & 0xFFu) << (8u * bb);
                    a3 |= ((terms[j1 * MD_M_COEF + m_hi]   >> sh_a) & 0xFFu) << (8u * bb);
                }
            }

            // B: 2 u32 fragments = 8 bytes of the byte-split weight table.
            // d_w_bytes_as_u32[limb*pitch + j] stores byte cw in bits 8*cw.
            // Column n=g -> cw = g%4 (sh_a/8) and limb = 2*ip + (g>>2).
            const unsigned i = 2u * ip + i_base;
            if (i < l) {
                const u32* w = d_w_bytes_as_u32 + u64(i) * pitch;
                #pragma unroll
                for (unsigned bb = 0; bb < 4u; ++bb) {
                    const unsigned j0 = kb + 4u * t + bb;
                    const unsigned j1 = kb + 16u + 4u * t + bb;
                    if (j0 < pitch) b0 |= ((w[j0] >> sh_a) & 0xFFu) << (8u * bb);
                    if (j1 < pitch) b1 |= ((w[j1] >> sh_a) & 0xFFu) << (8u * bb);
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
        // c0/c1 share ct=g%4, limb=t>>1, cw=2*(t%2)+{0,1}; row g -> coef g>>2,
        // row g+8 -> coef 2+(g>>2).
        const unsigned ab = t >> 1;              // limb_in_pair (0/1)
        const unsigned cw0 = (t & 1u) << 1;      // 0 or 2
        const unsigned cg = g >> 2;              // coefficient half (0/1)
        const unsigned ct = g & 3u;
        const unsigned idx0 = ((ab * 4u + cg) * 4u + ct) * 4u + cw0;
        const unsigned idx1 = ((ab * 4u + cg + 2u) * 4u + ct) * 4u + cw0;
        accw[idx0 + 0] = c0;
        accw[idx0 + 1] = c1v;
        accw[idx1 + 0] = c2v;
        accw[idx1 + 1] = c3v;
        __syncwarp();   // make other lanes' shared stores visible (Volta+)

        // ---- reassemble X once per (coef, limb) from the 16 S[ct][cw] ------
        if (lane < 8u) {
            const unsigned co = lane >> 1;       // coef_in_warp 0..3
            const unsigned lb = lane & 1u;       // limb_in_pair 0/1
            CrtMmaU128 X{0ull, 0ull};
            #pragma unroll
            for (unsigned ct2 = 0; ct2 < 4u; ++ct2) {
                #pragma unroll
                for (unsigned cw2 = 0; cw2 < 4u; ++cw2) {
                    const u32 S = u32(accw[((lb * 4u + co) * 4u + ct2) * 4u + cw2]);
                    u128_add_shift(X, u64(S), 8u * (ct2 + cw2));
                }
            }
            const unsigned i = 2u * ip + lb;
            if (i < l) {
                const unsigned mslot = 4u * warp + co;
                // CRITICAL: carry_and_reduce() consumes `low` as the LOW chunk
                // (its low 32 bits become digit[i]; upper bits are carried via
                // `carry >>= 32`) and `high` as the NEXT 32-bit chunk (added at
                // limb i+1 through prior_high).  Therefore `low` must be exactly
                // u32(X) -- storing the full 64-bit X.lo here double-counts bits
                // 32..63 (they also live in `high`), making every limb wrong.
                low[i * MD_TS + mslot]  = u64(u32(X.lo));              // X mod 2^32
                high[i * MD_TS + mslot] = (X.lo >> 32) | (X.hi << 32); // X bits 32..95
            }
        }
        __syncwarp();   // reuse accw in the next ip iteration safely
    }
    __syncthreads();

    // ---- step 4: reference carry_and_reduce / gap-fold reduction ----------
    const u64 total = u64(batches) * plan.n * l;   // B*N*L output elements
    if (threadIdx.x < MD_M_COEF && coef_tile + threadIdx.x < plan.n) {
        const unsigned m = threadIdx.x;
        u32 digits[MD_MAXL + 2];                   // CONTIGUOUS (BUG 2 fix)
        carry_and_reduce_mma(digits, low + m, high + m, l,
                             plan.bits, plan.gap, plan.q, MD_TS);
        const unsigned coef = coef_tile + m;
        const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
        if (dst + l <= total)                      // aggressive output bound check
            for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i];
    }
}
#endif  // __CUDACC__

// ===========================================================================
// 4. Host-only check of the byte-split reassembly math (plain C++).
// ===========================================================================
inline bool crt_mma_diag_selftest(unsigned iterations = 4096,
                                  u64 seed = 0x9e3779b97f4a7c15ull) {
    // xorshift64* — deterministic, no <random> needed.
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };

    for (unsigned it = 0; it < iterations; ++it) {
        const u32 a = u32(next()), b = u32(next());

        // u8 path: raw unsigned bytes, exact byte products; must equal a*b.
        CrtMmaU128 X{0ull, 0ull};
        for (unsigned ct = 0; ct < 4u; ++ct) {
            const u64 wa = (a >> (8u * ct)) & 0xFFu;
            for (unsigned cw = 0; cw < 4u; ++cw) {
                const u64 wb = (b >> (8u * cw)) & 0xFFu;
                u128_add_shift(X, wa * wb, 8u * (ct + cw));
            }
        }
        const u64 want = u64(a) * b;
        if (X.hi != 0 || X.lo != want) return false;

        // The (low, high) pair fed to carry_and_reduce must reassemble X, where
        // low = X mod 2^32 and high = (X >> 32) mod 2^64 (bits 32..95).
        const u64 low = X.lo & 0xFFFFFFFFull;
        const u64 high = (X.lo >> 32) | (X.hi << 32);
        if (low + high * (u64(1) << 32) != want) return false;

        // s8 compensation identity (documentation completeness).
        for (unsigned c = 0; c < 4u; ++c) {
            const u32 ua = (a >> (8u * c)) & 0xFFu;
            const u32 ub = (b >> (8u * c)) & 0xFFu;
            if ((ua & 0x7Fu) + 128u * (ua >> 7) != ua) return false;
            if ((ub & 0x7Fu) + 128u * (ub >> 7) != ub) return false;
        }
    }
    return true;
}

// ===========================================================================
// 4b. Host simulation of the FULL MMA tile (fragment packing -> MMA -> stash
// -> reassembly -> carry_and_reduce), checked byte-for-byte against the
// reference sum_column/carry_and_reduce.  This exercises the lane mappings and
// the low/high split that the byte-math selftest above cannot see.  It caught
// the "low = X.lo (64-bit)" bug that made all 7.2M limbs differ on device.
// Pure C++, no CUDA; small dims (K=8, L=8, one warp / 4 coefficients).
// ===========================================================================
inline bool crt_mma_diag_tile_selftest(unsigned iterations = 16,
                                       u64 seed = 0xa5a5f00d12345678ull) {
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };
    auto byte_of = [](u32 v, unsigned c) -> u32 { return (v >> (8u * c)) & 0xFFu; };

    const unsigned K = 8, pitch = K + 1, L = 8, CO = 4;
    u32 gap = 0, q[L] = {0};
    const unsigned bits = 256;

    for (unsigned it = 0; it < iterations; ++it) {
        u32 terms[pitch][CO], w[L][pitch];
        for (unsigned j = 0; j < pitch; ++j)
            for (unsigned co = 0; co < CO; ++co) terms[j][co] = u32(next());
        for (unsigned i = 0; i < L; ++i)
            for (unsigned j = 0; j < pitch; ++j) w[i][j] = u32(next());

        // ---- simulated kernel step 3: per-lane fragments, MMA, stash -------
        u64 low[L * MD_TS], high[L * MD_TS];
        for (unsigned z = 0; z < L * MD_TS; ++z) { low[z] = 0; high[z] = 0; }

        for (unsigned ip = 0; ip < (L + 1u) / 2u; ++ip) {
            long long C[32][4];
            for (unsigned z = 0; z < 32; ++z)
                for (unsigned r = 0; r < 4; ++r) C[z][r] = 0;

            // K=8 < 32 -> a single k-chunk (k = kb + 4t + {0..3}).
            unsigned A[16][32], B[32][8];
            for (unsigned m = 0; m < 16; ++m)
                for (unsigned k = 0; k < 32; ++k) A[m][k] = 0;
            for (unsigned k = 0; k < 32; ++k)
                for (unsigned n = 0; n < 8; ++n) B[k][n] = 0;

            for (unsigned lane = 0; lane < 32; ++lane) {
                const unsigned g = lane >> 2, t = lane & 3u;
                const unsigned sh = g & 3u;              // byte ct/cw within u32
                const unsigned m_base = g >> 2, m_hi = m_base + 2u;
                const unsigned i = 2u * ip + (g >> 2);   // limb (FIX: g>>2)

                u32 a0 = 0, a1 = 0, b0 = 0;
                for (unsigned bb = 0; bb < 4u; ++bb) {
                    const unsigned j0 = 4u * t + bb;
                    if (j0 < pitch) {
                        a0 |= byte_of(terms[j0][m_base], sh) << (8u * bb);
                        a1 |= byte_of(terms[j0][m_hi],   sh) << (8u * bb);
                    }
                }
                if (i < L) {
                    for (unsigned bb = 0; bb < 4u; ++bb) {
                        const unsigned j0 = 4u * t + bb;
                        if (j0 < pitch) b0 |= byte_of(w[i][j0], sh) << (8u * bb);
                    }
                }
                for (unsigned bb = 0; bb < 4u; ++bb) {
                    A[g][4u * t + bb]     = byte_of(a0, bb);
                    A[g + 8][4u * t + bb] = byte_of(a1, bb);
                    B[4u * t + bb][g]     = byte_of(b0, bb);
                }
            }
            // D[m][n] = sum_k A[m][k]*B[k][n]; then lane accumulators.
            long long D[16][8];
            for (unsigned m = 0; m < 16; ++m)
                for (unsigned n = 0; n < 8; ++n) {
                    long long s = 0;
                    for (unsigned k = 0; k < 32; ++k) s += (long long)A[m][k] * B[k][n];
                    D[m][n] = s;
                }
            for (unsigned lane = 0; lane < 32; ++lane) {
                const unsigned g = lane >> 2, t = lane & 3u;
                C[lane][0] += D[g][2u * t];
                C[lane][1] += D[g][2u * t + 1u];
                C[lane][2] += D[g + 8][2u * t];
                C[lane][3] += D[g + 8][2u * t + 1u];
            }

            // stash: accw[limb_in_pair][coef_half][ct][cw]
            long long accw[2][4][4][4];
            for (unsigned z = 0; z < 2; ++z)
                for (unsigned a = 0; a < 4; ++a)
                    for (unsigned b = 0; b < 4; ++b)
                        for (unsigned c = 0; c < 4; ++c) accw[z][a][b][c] = 0;
            for (unsigned lane = 0; lane < 32; ++lane) {
                const unsigned g = lane >> 2, t = lane & 3u;
                const unsigned ab = t >> 1, cw0 = (t & 1u) << 1u;
                const unsigned cg = g >> 2, ct = g & 3u;
                accw[ab][cg][ct][cw0]     = C[lane][0];
                accw[ab][cg][ct][cw0 + 1] = C[lane][1];
                accw[ab][cg + 2][ct][cw0]     = C[lane][2];
                accw[ab][cg + 2][ct][cw0 + 1] = C[lane][3];
            }

            // reassemble X once per (coef, limb) and split into low/high.
            for (unsigned co = 0; co < CO; ++co) {
                for (unsigned lb = 0; lb < 2u; ++lb) {
                    CrtMmaU128 X{0ull, 0ull};
                    for (unsigned ct = 0; ct < 4u; ++ct)
                        for (unsigned cw = 0; cw < 4u; ++cw)
                            u128_add_shift(X, u64(u32(accw[lb][co][ct][cw])), 8u * (ct + cw));
                    const unsigned i = 2u * ip + lb;
                    if (i < L) {
                        low[i * MD_TS + co]  = u64(u32(X.lo));                 // FIX
                        high[i * MD_TS + co] = (X.lo >> 32) | (X.hi << 32);
                    }
                }
            }
        }

        // ---- kernel step 4 vs reference sum_column/carry_and_reduce -------
        for (unsigned co = 0; co < CO; ++co) {
            u32 dg[MD_MAXL + 2], dr[MD_MAXL + 2];
            carry_and_reduce_mma(dg, low + co, high + co, L, bits, gap, q, MD_TS);

            u64 lo[L], hi[L];
            u32 col[pitch];
            for (unsigned j = 0; j < pitch; ++j) col[j] = terms[j][co];
            for (unsigned i = 0; i < L; ++i)
                sum_column<1>(col, w[i], pitch, lo[i], hi[i]);
            carry_and_reduce<1>(dr, lo, hi, L, bits, gap, q);

            for (unsigned i = 0; i < L; ++i)
                if (dg[i] != dr[i]) return false;
        }
    }
    return true;
}

// ===========================================================================
// 5. Host-side plan + launchers with built-in first-call diagnostics.
// ===========================================================================
struct CrtMmaPlan {
    unsigned L = 0, pitch = 0;
    std::int8_t* d_w = nullptr;      // L*pitch*4 bytes, 4 int8 per u32 weight
    bool ok = false;
    mutable bool diag_done_ = false; // diagnostics fire on the FIRST run() only

    CrtMmaPlan() = default;

    CrtMmaPlan(const std::vector<u32>& weights_host, unsigned L_, unsigned pitch_)
        : L(L_), pitch(pitch_) {
#ifdef __CUDACC__
        if (L == 0 || pitch == 0 || L > MD_MAXL || pitch > MD_MAXPITCH)
            throw std::runtime_error("crt_mma_diag: L/pitch outside compiled limits");
        if (weights_host.size() != std::size_t(L) * pitch)
            throw std::runtime_error("crt_mma_diag: weights table has wrong size");
        // Preserve each u32 bit pattern as four little-endian bytes; the MMA
        // kernel reads individual bytes, run_simple reads the u32s straight back.
        std::vector<std::int8_t> bytes(weights_host.size() * 4);
        for (std::size_t i = 0; i < weights_host.size(); ++i) {
            const u32 v = weights_host[i];
            for (unsigned c = 0; c < 4u; ++c)
                bytes[i * 4 + c] = static_cast<std::int8_t>((v >> (8u * c)) & 0xFFu);
        }
        if (cudaMalloc(reinterpret_cast<void**>(&d_w), bytes.size()) != cudaSuccess)
            throw std::runtime_error("crt_mma_diag: cudaMalloc failed");
        if (cudaMemcpy(d_w, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_w);
            d_w = nullptr;
            throw std::runtime_error("crt_mma_diag: weight upload failed");
        }
        ok = true;
#else
        (void)weights_host;
        throw std::runtime_error("crt_mma_diag: constructed without CUDA");
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
        const std::size_t smem = crt_mma_diag_shared_bytes(L, pitch);
        return ok && B > 0 && residues && out && args.primes && args.q &&
               args.limbs > 0 && args.limbs <= MD_MAXL &&
               args.count + 1 <= MD_MAXPITCH &&
               args.count + 1 == pitch && args.limbs == L &&
               smem > 0 && smem <= MD_SMEM_CAP;
    }

    // ---- PRIMARY tensor-core path with built-in first-call diagnostics ----
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
        const std::size_t smem = crt_mma_diag_shared_bytes(L, pitch);
        const bool diag = !diag_done_;
        if (diag) {
            diag_done_ = true;
            std::fprintf(stderr,
                         "[CRTMMA] enter: ok=%d L=%u pitch=%u B=%u smem=%zu\n",
                         (int)ok, L, pitch, B, smem);
            std::fflush(stderr);
        }
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) {
            if (diag) {
                std::fprintf(stderr, "[CRTMMA] check FAILED; no MMA launch\n");
                std::fflush(stderr);
            }
            return cudaErrorInvalidValue;
        }
        const CrtArgs ca = make_args(args);

        // Mandatory opt-in, issued on the ACTUAL instantiated kernel with the
        // REAL dynamic smem size, with error checking. Printed as int below.
        const cudaError_t e_attr = cudaFuncSetAttribute(
            crt_mma_diag_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem));
        if (diag) {
            std::fprintf(stderr, "[CRTMMA] funcSetAttribute=%d (smem=%zu)\n",
                         (int)e_attr, smem);
            std::fflush(stderr);
        }
        if (e_attr != cudaSuccess)
            return run_simple(residues, out, args, B, stream);   // safe fallback

        // One-off timing events (only on the first call, so normal timing is
        // unaffected). Recorded on the caller's stream around the kernel.
        cudaEvent_t ev0{}, ev1{};
        bool have_events = false;
        if (diag) {
            have_events = (cudaEventCreate(&ev0) == cudaSuccess) &&
                          (cudaEventCreate(&ev1) == cudaSuccess);
            if (have_events) cudaEventRecord(ev0, stream);
        }

        (void)cudaGetLastError();   // clear any stale error before the launch
        crt_mma_diag_kernel<0><<<dim3((ca.n + MD_M_COEF - 1) / MD_M_COEF, 1, B),
                                  MD_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out, B);

        const cudaError_t e_launch = cudaGetLastError();
        if (diag) {
            std::fprintf(stderr, "[CRTMMA] launch=%d\n", (int)e_launch);
            std::fflush(stderr);
        }

        if (diag && have_events) {
            cudaEventRecord(ev1, stream);
            cudaEventSynchronize(ev1);
            float ms = 0.f;
            const cudaError_t e_el = cudaEventElapsedTime(&ms, ev0, ev1);
            std::fprintf(stderr, "[CRTMMA] first-run kernel ms=%.3f (elapsedErr=%d)\n",
                         (double)ms, (int)e_el);
            std::fflush(stderr);
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }
        return e_launch;
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;   // never called in non-CUDA builds
#endif
    }

    // ---- EXACT path kept for API compatibility (same MMA kernel) ----------
    template <class Args>
    cudaError_t run_s8(const u32* residues, u32* out, const Args& args, u32 B,
                       cudaStream_t stream = nullptr) {
        return run(residues, out, args, B, stream);
    }

    // ---- CUDA-core reference (byte-identical to gpu31 reconstruct_kernel) --
    template <class Args>
    cudaError_t run_simple(const u32* residues, u32* out, const Args& args, u32 B,
                           cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_mma_diag_simple_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out, B);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }
};

}  // namespace crt_mma_diag

#undef CRTMD_FHD
