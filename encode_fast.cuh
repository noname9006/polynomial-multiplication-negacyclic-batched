// encode_fast.cuh — faster CUDA-core replacement for gpu31::encode_tiled.
//
// Header-only.  This file does NOT modify or include any existing project file
// except "gpu31.cuh" (and, in CUDA builds, <cuda_runtime.h>).  Include it AFTER
// gpu31.cuh in an nvcc build:
//
//     #include "gpu31.cuh"
//     #include "encode_fast.cuh"
//     ...
//     encode_fast::EncodeFastPlan enc;
//     enc.run(d_a, d_b, dev.d_ra, dev.d_rb,
//             dev.d_primes, dev.d_weights, dev.d_reversed_twist,
//             n, l, k, B, stream);          // exact gpu31 encode launch args
//
// No tensor cores, no inline PTX, no <mma.h>.  No new tables: the kernel reuses
// gpu31's constant-memory Shoup row (gpu31::c_weights, stride GPU31_MAXL_C=56)
// and the existing gpu31::Prime{r2, one} fields.  Output is byte-for-byte
// identical to gpu31's encode_tiled.
//
// ---------------------------------------------------------------------------
// MATH (the H3 insight, re-derived and re-verified here)
// ---------------------------------------------------------------------------
// gpu31's convert_both() does, per output (coefficient i, prime m):
//
//     t_l   = lazy_product(a_l, W_l, p)          in [0, 2p)        (Shoup)
//     sum   = Σ_l t_l                            u64, < L·2p < 2^38
//     r     = finish(sum, w_0, p)                = sum mod p
//     out   = mont(r, reversed_twist[m][i], p)
//
// with W_l = 2^(32(l+1)) mod p = gpu31's plan.weights[m*L + l].value and
// w_0 = W_0 = 2^32 mod p (the `radix` of finish).  Since t_l ≡ a_l·W_l (mod p),
// addition is mod p, and finish() is exact:
//
//     r = (Σ_l a_l·W_l) mod p =: X mod p .
//
// So the L (≤55) per-limb Shoup reductions are unnecessary if X is accumulated
// EXACTLY and reduced ONCE.  a_l < 2^32 and W_l < p < 2^31, so each product
// a_l·W_l < 2^63; with L ≤ 55 the whole sum X < 2^70, which fits a __uint128_t.
// The kernel replaces 55 `__umulhi` Shoup ops with 55 plain 64-bit multiply-adds
// into a 128-bit accumulator, followed by a single exact mod-p reduction.
//
// REDUCTION OF X (128-bit) mod p,  p < 2^31,  X < 2^70:
// Write X = hiw·2^64 + mid32·2^32 + lo32  with hiw = (u32)(X>>64) < 16.
// Let one = 2^32 mod p (= Prime.one = w0.value) and r2 = 2^64 mod p
// (= Prime.r2 = one² mod p = W_1).  Then
//
//     X mod p = ( lo32 + mid32·one + hiw·r2 ) mod p .
//
// The three terms are folded with Shoup lazy_product() using w0 — the very same
// constant-memory weight row gpu31 already uploads — and the hiw·r2 term is
// itself split into two 32-bit words (hiw·r2 < 2^36), so EVERY multiply is at
// most 32×32→64.  Concretely reduce128() below uses two `__umulhi` total (vs 55
// in the original).  `mont(r, twist, p)` is gpu31's, copied verbatim.
//
// This is algebraically identical to the task's suggested
//     ((hi·(2^64 mod p)) + lo) mod p
// with hi = X>>64 and lo = (u64)X; the Shoup fold of lo's two 32-bit halves is
// the exact way to take `lo mod p` without a slow 64-bit integer division.
//
// VERIFICATION: encode_fast_selftest() (default 200000 random iterations) checks
// reduce128() against gpu31's lazy_product()/finish() chain AND, where the host
// compiler supports __int128, against a direct exact X % p.  It passes.
// ---------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstddef>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include "gpu31.cuh"   // gpu31::Prime, gpu31::Weight, gpu31::c_weights, gpu31::mont, ...

namespace encode_fast {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Reuse gpu31's exact POD layouts so the same device pointers / constant table
// can be passed straight through (byte-for-byte compatible).
using Prime  = gpu31::Prime;
using Weight = gpu31::Weight;

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
#define EFHD __host__ __device__
#else
#define EFHD
#endif

// gpu31::encode_tiled constants (kept local so this file stays drop-in).
static constexpr unsigned EF_WARP       = 32;   // warp_size
static constexpr unsigned EF_GROUPS     = 8;    // group_primes
static constexpr unsigned EF_MAXL_C     = 56;   // gpu31::GPU31_MAXL_C
static constexpr unsigned EF_STRIDE     = EF_MAXL_C + 1;  // odd, conflict-free (=57)

// ---------------------------------------------------------------------------
// Exact 128→32 mod-p reduction (host + device).  See MATH above.
//   lo32, mid32 : low and middle 32-bit words of X = (u64)sum
//   hiw         : high word (u32)(sum >> 64), < 16 for our use
//   w0          : Shoup weight for W0 = 2^32 mod p  (c_weights row, j=0)
//   pr          : Prime with .p = p, .r2 = 2^64 mod p
// Requires p > 2^30 (true for every gpu31 plan), so 2^32 < 4p and three
// conditional subtractions canonicalize any u32 word.
// ---------------------------------------------------------------------------
EFHD inline u32 reduce128(u32 lo32, u32 mid32, u32 hiw,
                          const Weight& w0, const Prime& pr) {
    const u32 p = pr.p;

    // acc = lo32 mod p, in [0,p)
    u32 acc = gpu31::canonical_word(lo32, p);

    // + mid32 * (2^32 mod p) mod p, in [0,p)
    u32 t = gpu31::lazy_product(mid32, w0, p);   // in [0,2p)
    if (t >= p) t -= p;
    acc += t;                                    // < 2p < 2^32
    if (acc >= p) acc -= p;

    // + hiw * (2^64 mod p) mod p.  hiw < 16, r2 < p < 2^31 -> z < 2^36.
    // Split z = zh*2^32 + zl and reuse w0 once more (zh < 16 < 2^32).
    const u64 z  = u64(hiw) * pr.r2;
    const u32 zl = u32(z);
    const u32 zh = u32(z >> 32);
    const u32 zlc = gpu31::canonical_word(zl, p);       // [0,p)
    u32 tz = gpu31::lazy_product(zh, w0, p);            // [0,2p)
    if (tz >= p) tz -= p;
    t = zlc + tz;                                       // < 2p
    if (t >= p) t -= p;

    acc += t;                                           // < 2p
    if (acc >= p) acc -= p;
    return acc;
}

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)

// ---------------------------------------------------------------------------
// Drop-in replacement for gpu31::encode_tiled.  Same parameters, same grid,
// same block, same dynamic shared-memory size, same staged shared data layout.
// ---------------------------------------------------------------------------
__global__ void encode_fast(const u32* __restrict__ ar, const u32* __restrict__ br,
                            u32* __restrict__ ra, u32* __restrict__ rb,
                            const Prime* __restrict__ primes,
                            const Weight* /*weights*/,          // unused: c_weights is used
                            const u32* __restrict__ reversed_twist,
                            unsigned n, unsigned l, unsigned count) {
    extern __shared__ u32 stage[];
    u32* sa = stage;
    u32* sb = stage + EF_WARP * EF_STRIDE;

    const unsigned x = threadIdx.x, y = threadIdx.y;
    const unsigned start = blockIdx.x * EF_WARP;
    const unsigned valid = (n - start < EF_WARP) ? (n - start) : EF_WARP;
    const u64 base = u64(blockIdx.z) * n * l;
    const u64 rbase = u64(blockIdx.z) * count * n;

    // Stage this tile's coefficient limbs (read once), same as gpu31.
    for (unsigned t = y * EF_WARP + x; t < valid * l; t += EF_WARP * EF_GROUPS) {
        const unsigned row = t / l, col = t - row * l;
        const u64 off = base + u64(start) * l + t;
        sa[row * EF_STRIDE + col] = ar[off];
        sb[row * EF_STRIDE + col] = br[off];
    }
    __syncthreads();

    const unsigned m = blockIdx.y * EF_GROUPS + y;
    if (m >= count || x >= valid) return;
    const Prime pr = primes[m];
    const Weight* wrow = gpu31::c_weights + u64(m) * EF_MAXL_C;  // broadcast per warp

    // ---- operand a --------------------------------------------------------
    __uint128_t suma = 0;
    #pragma unroll 4
    for (unsigned j = 0; j < l; ++j) {
        suma += (__uint128_t)(u64(sa[x * EF_STRIDE + j]) * u64(wrow[j].value));
    }
    const u64 loa = u64(suma);
    const u32 a = reduce128(u32(loa), u32(loa >> 32), u32(suma >> 64), wrow[0], pr);

    // ---- operand b --------------------------------------------------------
    __uint128_t sumb = 0;
    #pragma unroll 4
    for (unsigned j = 0; j < l; ++j) {
        sumb += (__uint128_t)(u64(sb[x * EF_STRIDE + j]) * u64(wrow[j].value));
    }
    const u64 lob = u64(sumb);
    const u32 b = reduce128(u32(lob), u32(lob >> 32), u32(sumb >> 64), wrow[0], pr);

    // Byte-for-byte identical final step to gpu31::encode_tiled.
    const u64 off = rbase + u64(m) * n + start + x;
    const u32 tw  = reversed_twist[u64(m) * n + start + x];
    ra[off] = gpu31::mont(a, tw, pr);
    rb[off] = gpu31::mont(b, tw, pr);
}

#endif  // __CUDACC__ && !PGPU_HOST_CHECK

// ---------------------------------------------------------------------------
// Host-side plan: stateless.  All tables are gpu31's; nothing to upload.
// ---------------------------------------------------------------------------
struct EncodeFastPlan {
    EncodeFastPlan() = default;   // no-op: reuses gpu31::c_weights + twist + primes

    EncodeFastPlan(const EncodeFastPlan&) = delete;
    EncodeFastPlan& operator=(const EncodeFastPlan&) = delete;

    // Mirrors gpu31's encode launch exactly:
    //   encode_tiled<<<dim3((n+31)/32,(k+7)/8,B), dim3(32,8), enc_shared, stream>>>
    //       (d_a, d_b, d_ra, d_rb, d_primes, d_weights, d_reversed_twist, n, l, k);
#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
    cudaError_t run(const u32* ar, const u32* br, u32* ra, u32* rb,
                    const Prime* primes, const Weight* weights,
                    const u32* reversed_twist,
                    unsigned n, unsigned l, unsigned count,
                    u32 B, cudaStream_t stream = nullptr) {
        if (B == 0) return cudaSuccess;
        if (!ar || !br || !ra || !rb || !primes || !reversed_twist)
            return cudaErrorInvalidValue;
        const std::size_t enc_shared =
            2u * EF_WARP * EF_STRIDE * sizeof(u32);   // 14592 B, < 48 KiB
        encode_fast<<<dim3((n + EF_WARP - 1) / EF_WARP,
                           (count + EF_GROUPS - 1) / EF_GROUPS, B),
                       dim3(EF_WARP, EF_GROUPS), enc_shared, stream>>>(
            ar, br, ra, rb, primes, weights, reversed_twist, n, l, count);
        return cudaGetLastError();
    }

    // Convenience overload: pass a gpu31::Device directly (duck-typed).
    template <class Dev>
    cudaError_t run(const u32* ar, const u32* br, u32* ra, u32* rb,
                    const Dev& dev, u32 B, cudaStream_t stream = nullptr) {
        const auto& pl = dev.plan();
        return run(ar, br, ra, rb,
                   dev.d_primes, dev.d_weights, dev.d_reversed_twist,
                   pl.N, pl.L, pl.K, B, stream);
    }
#endif
};

// ---------------------------------------------------------------------------
// Pure-host self-test.  Compares the new reduction math against gpu31's
// lazy_product()/finish() chain on random inputs, and (when available) against
// a direct exact 128-bit X % p.  No CUDA required.
// ---------------------------------------------------------------------------
inline bool encode_fast_selftest(unsigned iters = 200000,
                                 u64 seed = 0x9e3779b97f4a7c15ull) {
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };

    for (unsigned it = 0; it < iters; ++it) {
        // Any odd p in [2^30, 2^31) works; primality is irrelevant to Shoup.
        u32 p = u32(next()) & 0x7fffffffu;
        if (p < (1u << 30)) p += (1u << 30);
        p |= 1u;

        const u32 one = u32((u64(1) << 32) % p);
        const u32 r2  = u32(u64(one) * one % p);           // 2^64 mod p = W_1
        const Prime pr{p, u32(0), r2, one};               // neg_inv unused here
        const Weight w0 = gpu31::make_weight(one, p);     // W_0, gpu31's radix

        const unsigned L = 1u + u32(next() % 55u);

        u64 sum_t = 0;              // gpu31 chain: Σ lazy_product(a_l, W_l, p)
        u64 acc_lo = 0, acc_hi = 0; // exact X as {hi64, lo64}
        u32 factor = one;           // W_l = 2^(32(l+1)) mod p
        for (unsigned l = 0; l < L; ++l) {
            const u32 a = u32(next());
            const u32 W = factor;
            const Weight w = gpu31::make_weight(W, p);
            sum_t += gpu31::lazy_product(a, w, p);

            u64 prod = u64(a) * W;             // < 2^63
            u64 nl = acc_lo + prod;
            if (nl < acc_lo) ++acc_hi;         // carry (acc_hi stays < 64)
            acc_lo = nl;

            factor = u32(u64(factor) * one % p);
        }

        // (1) gpu31::finish == (Σ a_l·W_l) mod p  (reference chain)
        const u32 ref = gpu31::finish(sum_t, w0, p);

        // (2) new reduction from the exact 128-bit accumulator
        const u32 lo32  = u32(acc_lo);
        const u32 mid32 = u32(acc_lo >> 32);
        const u32 hiw   = u32(acc_hi);          // bits 64.. of X, < 16
        const u32 got = reduce128(lo32, mid32, hiw, w0, pr);
        if (got != ref) return false;

#if defined(__SIZEOF_INT128__) || defined(_SIZEOF_INT128_)
        // (3) independent exact check (host compilers with __int128, e.g. g++)
        typedef unsigned __int128 u128;
        const u128 X = (u128(acc_hi) << 64) | u128(acc_lo);
        const u32 direct = u32(X % p);
        if (direct != ref || direct != got) return false;
#endif
    }
    return true;
}

}  // namespace encode_fast

#undef EFHD
