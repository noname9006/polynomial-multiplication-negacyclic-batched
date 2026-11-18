// crt_barrier.cuh — EXACT CRT reconstruction for the batched 31-bit RNS
// polynomial multiplication (drop-in replacement for gpu31::reconstruct_kernel).
//
// Header-only.  Nothing outside this file is modified.  Integrate by including
// "crt_barrier.cuh" after gpu31.cuh (CrtArgs is duck-typed, so a gpu31::CrtArgs
// can be passed directly — see run()/run_s8()/run_simple() below).
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE IS (and is NOT)
// ---------------------------------------------------------------------------
// This is a CONSERVATIVE optimization of crt_mma.cuh, which is the current
// best kernel (tile C=16, Y=8, odd stride 17, 128 threads/block, ~37 ms).
// A full-warp tile-32 variant (crt_fast.cuh) was measured SLOWER (44 ms) — that
// path is deliberately NOT taken here.
//
// crt_barrier.cuh keeps the EXACT SAME tile geometry, block/grid shape, and
// dynamic shared-memory footprint as crt_mma.cuh.  The ONLY change is the
// removal of the second __syncthreads() barrier (3 -> 2), using the technique
// from crt_fast.cuh:
//
//   * crt_mma step 2 writes the FP64 correction into shared `terms[k*TS+m]`
//     from the y==0 warp, then syncs so every other warp can read column K in
//     the per-limb product sums.
//   * crt_barrier computes the same correction into a PRIVATE register of the
//     y==0 thread (no shared write, no barrier), and EXCLUDES column K from the
//     product sums.  In the final step that same y==0 thread folds
//     corr * Wcorr[i] directly into low[i]/high[i] before the carry/reduce.
//
// The two barriers that remain are the two that are genuinely required:
//   (1) after step 1 — product-sum threads must see all Shoup terms;
//   (2) after step 3 — the y==0 thread must see every low/high written by the
//       other y-warp workers.
//
// WHY THE OUTPUT IS STILL BYTE-FOR-BYTE IDENTICAL TO gpu31::reconstruct_kernel:
//   sum_column accumulates two independent u64 sums
//       lo = sum_j u32(t_j * W_j),   hi = sum_j (t_j * W_j) >> 32.
//   Splitting column K out and adding u32(corr*Wcorr) / (corr*Wcorr >> 32)
//   afterwards is just integer addition (no overflow: <= 118 * 2^32 < 2^39),
//   hence associative — the low/high pair is bit-identical.  carry_and_reduce
//   consumes exactly the same low/high, so digits and the final output match.
//   The correction value itself is computed by the same FP64 expression
//   (u32(sum + 0.5)), once per coefficient, so there is no FP64 difference.
//
// All three entry points produce byte-for-byte identical output:
//   * run() / run_s8() — the 2-barrier CUDA-core kernel below (same code).
//   * run_simple()     — the reference sum_column kernel, copied verbatim
//                         from gpu31.cuh, for A/B testing and debugging.
//
// crt_barrier_selftest() at the bottom verifies the math on the host, with no
// CUDA required.  crt_barrier_hostcheck_official() replicates both the
// reference and the barrier path at the official point (L=55, K=117,
// q = 2^1747 - 3) and returns the byte-mismatch count.
// ===========================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstddef>

#include <cuda_runtime.h>          // shim provides the error enum in host builds

#include "fherma.h"

namespace crt_barrier {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__)
#define CRTBR_FHD __host__ __device__
#else
#define CRTBR_FHD
#endif

// Compile-time limits (raise if a future point needs more; the host side checks).
static constexpr unsigned CRT_BARRIER_MAXL     = 128;   // max limbs
static constexpr unsigned CRT_BARRIER_MAXPITCH = 512;   // max K+1 columns
static constexpr unsigned CRT_BARRIER_TILE_C   = 16;    // coefficients per block
static constexpr unsigned CRT_BARRIER_TILE_Y   = 8;     // threads per coefficient
static constexpr unsigned CRT_BARRIER_TS       = CRT_BARRIER_TILE_C + 1;  // 17: ODD stride
static constexpr unsigned CRT_BARRIER_THREADS  = CRT_BARRIER_TILE_C * CRT_BARRIER_TILE_Y;  // 128

// Dynamic shared memory used by crt_barrier_kernel (must match the kernel exactly).
//   terms : pitch * CRT_BARRIER_TS u32, followed by 0/1 u32 of padding so the
//           u64 arrays start on an 8-byte boundary;
//   low   : l * CRT_BARRIER_TS u64
//   high  : l * CRT_BARRIER_TS u64
//   digits: (l+2) * CRT_BARRIER_TS u32
// This is IDENTICAL to crt_mma::crt_mma_shared_bytes (same tile, same layout).
static inline std::size_t crt_barrier_shared_bytes(unsigned l, unsigned pitch) {
    const unsigned terms_words = pitch * CRT_BARRIER_TS;
    const unsigned aligned_words = (terms_words + 1u) & ~1u;
    return std::size_t(aligned_words + CRT_BARRIER_TS * (5u * l + 2u)) * sizeof(u32);
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
// 1. Helpers copied verbatim from gpu31.cuh's fherma_crt section (same as
//    crt_mma.cuh).  `shifted_word` intentionally uses q[i], NOT q[i*Stride]:
//    q is a plain L-word array and carries no tile stride.
// ===========================================================================
CRTBR_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template <unsigned Stride>
CRTBR_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}

template <unsigned Stride>
CRTBR_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
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
CRTBR_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}

template <unsigned Stride>
CRTBR_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
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
CRTBR_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes, unsigned count) {
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}

template <unsigned Stride>
CRTBR_FHD inline void sum_column(const u32* terms, const u32* basis_column,
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
CRTBR_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
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
__global__ void crt_barrier_simple_kernel(CrtArgs plan, const u32* residues,
                                          const u32* weights, u32* out) {
    const unsigned coef = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = blockIdx.y;
    if (coef >= plan.n) return;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;

    u32 terms[CRT_BARRIER_MAXPITCH];
    u64 low[CRT_BARRIER_MAXL], high[CRT_BARRIER_MAXL];
    u32 digits[CRT_BARRIER_MAXL + 2];

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
#endif  // __CUDACC__

// ===========================================================================
// 3. Production kernel — crt_mma tile geometry, 2 barriers instead of 3.
//
//    One block handles CRT_BARRIER_TILE_C=16 coefficients for one pair; its 128
//    threads form a 16 (m) x 8 (y) grid.  Shared storage is laid out with the
//    ODD stride CRT_BARRIER_TS=17 so consecutive m never share a bank:
//        terms[j*TS + m]   pitch rows of per-coefficient terms
//        low[i*TS + m]     per-limb low  32 bits (as u64)
//        high[i*TS + m]    per-limb high bits
//        digits[i*TS + m]  final reduction workspace
//    Same block/grid/smem as crt_mma_kernel.  Only the barrier structure and
//    the treatment of correction column K differ (see file header).
// ===========================================================================
#if defined(__CUDACC__)
template <int MODE>
__global__ void crt_barrier_kernel(CrtArgs plan, const u32* residues,
                                   const u32* d_weights, u32* out) {
    (void)MODE;  // run() and run_s8() share this identical, exact implementation

    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned m = threadIdx.x % CRT_BARRIER_TILE_C;
    const unsigned y = threadIdx.x / CRT_BARRIER_TILE_C;        // 0 .. CRT_BARRIER_TILE_Y-1
    const unsigned coef = blockIdx.x * CRT_BARRIER_TILE_C + m;
    const unsigned pair = blockIdx.z;
    const bool active = coef < plan.n;

    extern __shared__ unsigned char smem[];
    u32* terms = reinterpret_cast<u32*>(smem);
    // (pitch*TS) rounded up to an even number of u32 -> 8-byte aligned u64s.
    u64* low = reinterpret_cast<u64*>(terms + ((pitch * CRT_BARRIER_TS + 1u) & ~1u));
    u64* high = low + l * CRT_BARRIER_TS;
    u32* digits = reinterpret_cast<u32*>(high + l * CRT_BARRIER_TS);

    const u64 rbase = u64(pair) * k * plan.n;

    // ---- step 1: Shoup terms t_j = r_j * (P/p_j)^-1 mod p_j, for j < k -------
    for (unsigned j = y; j < k; j += CRT_BARRIER_TILE_Y) {
        const u32 r = active
                    ? residues[rbase + u64(j) * plan.n + coef] : 0u;
        terms[j * CRT_BARRIER_TS + m] = weight_residue(r, plan.primes[j]);
    }
    __syncthreads();                                          // BARRIER 1 of 2

    // ---- step 2: FP64 correction, computed ONCE per coefficient by y==0 -----
    // crt_mma writes this into terms[k*TS+m] and adds a barrier.  Here it lives
    // in a private register: no shared write, no barrier.  Column K is excluded
    // from the product sums below and folded back in step 4.
    u32 corr = 0;
    if (y == 0 && active) {
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
            sum += double(terms[j * CRT_BARRIER_TS + m]) * plan.primes[j].reciprocal;
        corr = u32(sum + 0.5);
    }

    // ---- step 3: exact per-limb product sums over j < k (NOT column K) ------
    for (unsigned i = y; i < l; i += CRT_BARRIER_TILE_Y) {
        u64 lo = 0, hi = 0;
        const u32* w = d_weights + u64(i) * pitch;
        for (unsigned j = 0; j < k; ++j) {
            const u64 product = u64(terms[j * CRT_BARRIER_TS + m]) * w[j];
            lo += u32(product);
            hi += product >> 32;
        }
        low[i * CRT_BARRIER_TS + m] = lo;
        high[i * CRT_BARRIER_TS + m] = hi;
    }
    __syncthreads();                                          // BARRIER 2 of 2

    // ---- step 4: fold corr*Wcorr[i], then carry_and_reduce / reduce_sum ------
    // Only the y==0 worker participates.  It holds `corr` in a register and
    // reads every low/high written in step 3 (now visible via barrier 2).
    if (y == 0 && active) {
        const u32* wcorr = d_weights + u64(k);   // column K = (-P) mod q
        u64 carry = 0, prior_high = 0;
        for (unsigned i = 0; i < l; ++i) {
            const u64 cp = u64(corr) * wcorr[u64(i) * pitch];
            const u64 lo = low[i * CRT_BARRIER_TS + m] + u32(cp);
            const u64 hi = high[i * CRT_BARRIER_TS + m] + (cp >> 32);
            carry += lo + prior_high;
            digits[i * CRT_BARRIER_TS + m] = u32(carry);
            carry >>= 32;
            prior_high = hi;
        }
        carry += prior_high;
        digits[l * CRT_BARRIER_TS + m] = u32(carry);
        digits[(l + 1) * CRT_BARRIER_TS + m] = u32(carry >> 32);
        reduce_sum<CRT_BARRIER_TS>(digits + m, l, plan.bits, plan.gap, plan.q);

        const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
        for (unsigned i = 0; i < l; ++i)
            out[dst + i] = digits[i * CRT_BARRIER_TS + m];
    }
}
#endif  // __CUDACC__

// ===========================================================================
// 4. Host-only self-test of the barrier-reduction math (plain C++, no CUDA).
// ---------------------------------------------------------------------------
// Exercises exactly the change relative to crt_mma:
//   (a) weight_residue's Shoup reduction equals a direct modular product;
//   (b) splitting correction column K out of sum_column and folding
//       corr*Wcorr back into low/high in step 4 yields the SAME low/high and
//       the SAME digits as the full sum_column over all K+1 columns;
//   (c) reduce_sum returns a canonical value < q.
// Small moduli keep every intermediate inside u64 (no __uint128_t dependency).
// ===========================================================================
inline bool crt_barrier_selftest(unsigned iterations = 2048,
                                 u64 seed = 0x9e3779b97f4a7c15ull) {
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };

    // Four small primes; P = product < 2^41, q = 2^20 - 3 (gap = 3).
    const u32 primes[4] = {1009u, 1013u, 1019u, 1021u};
    const unsigned K = 4, pitch = K + 1;
    const u32 qv = (1u << 20) - 3u, gap = 3u, bits = 20u;
    u64 P = 1;
    for (unsigned j = 0; j < K; ++j) P *= u64(primes[j]);

    u32 W[pitch];
    const u32 PmodQ = u32(P % qv);
    for (unsigned j = 0; j < K; ++j) W[j] = u32((P / u64(primes[j])) % u64(qv));
    W[K] = PmodQ ? qv - PmodQ : 0u;

    PrimeInfo pi[K];
    for (unsigned j = 0; j < K; ++j) {
        const u32 p = primes[j];
        const u64 basis = P / u64(p);
        const u32 base = u32(basis % p);
        u32 inv = 1, b = base;
        for (u64 e = p - 2; e; e >>= 1) {
            if (e & 1) inv = u32(u64(inv) * b % p);
            b = u32(u64(b) * b % p);
        }
        if (u64(basis % p) * inv % p != 1) return false;
        pi[j] = PrimeInfo{p, inv, u32((u64(inv) << 32) / p), 1.0 / double(p)};
    }

    for (unsigned it = 0; it < iterations; ++it) {
        // (a) Shoup reduction.
        for (unsigned j = 0; j < K; ++j) {
            const u32 r = u32(next()) % primes[j];
            const u32 want = u32(u64(r) * pi[j].inv % primes[j]);
            if (weight_residue(r, pi[j]) != want) return false;
        }

        // Random terms and a small correction (the real correction is small:
        // |sum_j t_j/p_j| < K, so round() is in [0,K]).
        u32 terms[pitch];
        for (unsigned j = 0; j < K; ++j) terms[j] = u32(next()) % primes[j];
        terms[K] = u32(next()) % (K + 1u);

        // (b) Full reference: sum_column over all K+1 columns.
        u64 lo_full = 0, hi_full = 0;
        sum_column<1>(terms, W, pitch, lo_full, hi_full);
        u32 dig_full[CRT_BARRIER_MAXL + 2] = {};
        carry_and_reduce<1>(dig_full, &lo_full, &hi_full, 1, bits, gap, &qv);
        if (dig_full[0] >= qv) return false;                    // (c) canonical

        // Split path: sum j<K, then fold column K in afterwards (the barrier
        // kernel's step 3 + step 4).
        u32 terms_k[pitch];
        for (unsigned j = 0; j < pitch; ++j) terms_k[j] = terms[j];
        terms_k[K] = 0;
        u64 lo_split = 0, hi_split = 0;
        sum_column<1>(terms_k, W, pitch, lo_split, hi_split);
        const u64 cp = u64(terms[K]) * W[K];
        lo_split += u32(cp);
        hi_split += cp >> 32;
        if (lo_split != lo_full || hi_split != hi_full) return false;
        u32 dig_split[CRT_BARRIER_MAXL + 2] = {};
        carry_and_reduce<1>(dig_split, &lo_split, &hi_split, 1, bits, gap, &qv);
        if (dig_split[0] != dig_full[0]) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Host replication of BOTH paths at the OFFICIAL point: L=55, K=117,
// q = 2^1747 - 3 (gap = 3, bits = 1747).  Returns the number of byte mismatches
// across `coeffs` random coefficients.  This is the validation requested for
// this change: it compares the 2-barrier path against a faithful host copy of
// gpu31::reconstruct_kernel (stride crt_tile_x=16, correction column included,
// full sum_column over K+1 columns).  The equivalence is independent of the
// actual weight values, so deterministic pseudo-random weights/residues are
// used; the official q exercises the real gap-fold reduction.
// ---------------------------------------------------------------------------
inline unsigned crt_barrier_hostcheck_official(unsigned coeffs = 512,
                                                u64 seed = 0x123456789abcdef0ull) {
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };

    const unsigned K = 117, L = 55, pitch = K + 1, bits = 1747, gap = 3u;

    // q = 2^1747 - 3 in 55 little-endian 2^32 words.
    // 1747 = 54*32 + 19, so q[54] = 2^19 - 1 = 0x7FFFF; q[0] = 0xFFFFFFFD;
    // q[1..53] = 0xFFFFFFFF.
    u32 q[L];
    for (unsigned i = 0; i < L; ++i) q[i] = 0xFFFFFFFFu;
    q[0] = 0xFFFFFFFDu;
    q[54] = 0x0007FFFFu;

    // Distinct 31-bit primes and their CRT data.  Any distinct primes < 2^31
    // suffice for a byte-equivalence check; use an arithmetic progression of
    // odd numbers near 2^31 and keep only primes.
    std::vector<u32> ps;
    ps.reserve(K);
    for (u32 cand = 0x70000001u; ps.size() < K; cand += 2) {
        bool prime = cand >= 3;
        for (u64 d = 3; d * d <= cand && prime; d += 2)
            if (cand % u32(d) == 0) prime = false;
        if (prime) ps.push_back(cand);
    }

    std::vector<PrimeInfo> primes(K);
    for (unsigned j = 0; j < K; ++j) {
        const u32 p = ps[j];
        // Modular inverse of 2 mod p via Fermat (2^(p-2)); any nonzero base
        // works for this equivalence check.
        u32 inv = 1, a = 2;
        for (u64 e = p - 2; e; e >>= 1) {
            if (e & 1) inv = u32(u64(inv) * a % p);
            a = u32(u64(a) * a % p);
        }
        primes[j] = PrimeInfo{p, inv, u32((u64(inv) << 32) / p), 1.0 / double(p)};
    }

    // Deterministic pseudo-random weights: L rows of pitch columns.
    std::vector<u32> W(std::size_t(L) * pitch);
    for (auto& w : W) w = u32(next());

    const unsigned REF = CRT_BARRIER_TILE_C;   // reference stride: crt_tile_x = 16
    const unsigned BAR = CRT_BARRIER_TS;       // barrier stride: 17

    unsigned mismatches = 0;
    for (unsigned c = 0; c < coeffs; ++c) {
        // --- shared inputs for this coefficient -----------------------------
        u32 t[CRT_BARRIER_MAXPITCH] = {};       // terms[j], j < K
        for (unsigned j = 0; j < K; ++j) {
            const u32 r = u32(next()) % primes[j].p;
            t[j] = weight_residue(r, primes[j]);
        }
        // Correction: identical expression in both paths.
        const u32 corr = crt_correction<1>(t, primes.data(), K);

        // --- reference path (gpu31::reconstruct_kernel, stride 16) ----------
        // Arrays are sized for the STRIDE (not just the element count), exactly
        // like the kernel's shared-memory layout: terms has (K+1)*REF entries,
        // low/high/digits have (L+2)*REF / (L+2)*REF entries.
        u32 tref[(K + 1) * REF] = {};
        for (unsigned j = 0; j < K; ++j) tref[j * REF] = t[j];
        tref[K * REF] = corr;                    // correction column in shared
        u32 ref_out[L];
        {
            u64 low[(L + 2) * REF], high[(L + 2) * REF];
            u32 digits[(L + 2) * REF] = {};
            for (unsigned i = 0; i < L; ++i)
                sum_column<REF>(tref, W.data() + std::size_t(i) * pitch, pitch,
                                low[i * REF], high[i * REF]);
            carry_and_reduce<REF>(digits, low, high, L, bits, gap, q);
            for (unsigned i = 0; i < L; ++i) ref_out[i] = digits[i * REF];
        }

        // --- barrier path (crt_barrier_kernel, stride 17, 2 barriers) -------
        u32 bar_out[L];
        {
            u64 low[(L + 2) * BAR], high[(L + 2) * BAR];
            u32 digits[(L + 2) * BAR] = {};
            // step 3: product sums over j < K only (column K excluded).
            for (unsigned i = 0; i < L; ++i) {
                u64 lo = 0, hi = 0;
                const u32* w = W.data() + std::size_t(i) * pitch;
                for (unsigned j = 0; j < K; ++j) {
                    const u64 product = u64(t[j]) * w[j];
                    lo += u32(product);
                    hi += product >> 32;
                }
                low[i * BAR] = lo;
                high[i * BAR] = hi;
            }
            // step 4: fold corr*Wcorr[i] into low/high, then carry and reduce.
            const u32* wcorr = W.data() + K;
            u64 carry = 0, prior_high = 0;
            for (unsigned i = 0; i < L; ++i) {
                const u64 cp = u64(corr) * wcorr[std::size_t(i) * pitch];
                const u64 lo = low[i * BAR] + u32(cp);
                const u64 hi = high[i * BAR] + (cp >> 32);
                carry += lo + prior_high;
                digits[i * BAR] = u32(carry);
                carry >>= 32;
                prior_high = hi;
            }
            carry += prior_high;
            digits[L * BAR] = u32(carry);
            digits[(L + 1) * BAR] = u32(carry >> 32);
            reduce_sum<BAR>(digits, L, bits, gap, q);
            for (unsigned i = 0; i < L; ++i) bar_out[i] = digits[i * BAR];
        }

        for (unsigned i = 0; i < L; ++i)
            if (ref_out[i] != bar_out[i]) ++mismatches;
    }
    return mismatches;
}

// ===========================================================================
// 5. Host-side plan + launchers (mirrors crt_mma::CrtMmaPlan API exactly).
// ===========================================================================
struct CrtBarrierPlan {
    unsigned L = 0, pitch = 0;
    std::int8_t* d_w = nullptr;      // L*pitch*4 bytes, 4 int8 per u32
    bool ok = false;

    CrtBarrierPlan() = default;

    CrtBarrierPlan(const std::vector<u32>& weights_host, unsigned L_, unsigned pitch_)
        : L(L_), pitch(pitch_) {
#ifdef __CUDACC__
        if (L == 0 || pitch == 0 || L > CRT_BARRIER_MAXL || pitch > CRT_BARRIER_MAXPITCH)
            throw std::runtime_error("crt_barrier: L/pitch outside compiled limits");
        if (weights_host.size() != std::size_t(L) * pitch)
            throw std::runtime_error("crt_barrier: weights table has wrong size");
        if (crt_barrier_shared_bytes(L, pitch) > 48u * 1024u)
            throw std::runtime_error("crt_barrier: shared memory exceeds 48 KiB");
        // Preserve each u32 bit pattern as four little-endian bytes; the kernel
        // reads the buffer straight back as u32 (little-endian device).
        std::vector<std::int8_t> bytes(weights_host.size() * 4);
        for (std::size_t i = 0; i < weights_host.size(); ++i) {
            const u32 v = weights_host[i];
            for (unsigned c = 0; c < 4; ++c)
                bytes[i * 4 + c] = static_cast<std::int8_t>((v >> (8u * c)) & 0xFFu);
        }
        if (cudaMalloc(reinterpret_cast<void**>(&d_w), bytes.size()) != cudaSuccess)
            throw std::runtime_error("crt_barrier: cudaMalloc failed");
        if (cudaMemcpy(d_w, bytes.data(), bytes.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_w);
            d_w = nullptr;
            throw std::runtime_error("crt_barrier: weight upload failed");
        }
        ok = true;
#else
        (void)weights_host;
        throw std::runtime_error("crt_barrier: constructed without CUDA");
#endif
    }

    CrtBarrierPlan(const CrtBarrierPlan&) = delete;
    CrtBarrierPlan& operator=(const CrtBarrierPlan&) = delete;

    ~CrtBarrierPlan() {
#ifdef __CUDACC__
        if (d_w) cudaFree(d_w);
#endif
        d_w = nullptr;
    }

    // Build the device argument block from any CrtArgs-compatible struct.
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
               args.limbs > 0 && args.limbs <= CRT_BARRIER_MAXL &&
               args.count + 1 <= CRT_BARRIER_MAXPITCH &&
               args.count + 1 == pitch && args.limbs == L &&
               crt_barrier_shared_bytes(L, pitch) <= 48u * 1024u;
    }

    // ---- PRIMARY: exact 2-barrier CUDA-core CRT ----------------------------
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_barrier_shared_bytes(L, pitch);
        crt_barrier_kernel<0><<<dim3((ca.n + CRT_BARRIER_TILE_C - 1) / CRT_BARRIER_TILE_C,
                                      1, B),
                                CRT_BARRIER_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;   // never called in non-CUDA builds
#endif
    }

    // ---- EXACT path kept for API compatibility (same kernel as run()) ------
    template <class Args>
    cudaError_t run_s8(const u32* residues, u32* out, const Args& args, u32 B,
                       cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_barrier_shared_bytes(L, pitch);
        crt_barrier_kernel<1><<<dim3((ca.n + CRT_BARRIER_TILE_C - 1) / CRT_BARRIER_TILE_C,
                                      1, B),
                                CRT_BARRIER_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }

    // ---- CUDA-core reference (byte-identical to reconstruct_kernel) --------
    template <class Args>
    cudaError_t run_simple(const u32* residues, u32* out, const Args& args, u32 B,
                           cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_barrier_simple_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }
};

// Alias so crt_barrier::CrtMmaPlan(...) mirrors crt_mma::CrtMmaPlan(...) exactly.
using CrtMmaPlan = CrtBarrierPlan;

}  // namespace crt_barrier

#undef CRTBR_FHD
