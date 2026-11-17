// crt_fast.cuh — EXACT centered-CRT reconstruction, CUDA-core only, tuned for
// throughput over the conflict-free crt_mma.cuh kernel.
//
// Header-only.  Creates ONLY this file; nothing else in the repo is touched.
// Integrate by including it after gpu31.cuh (CrtArgs is duck-typed, so a
// gpu31::CrtArgs can be passed straight to run()):
//
//     #include "gpu31.cuh"
//     #include "crt_fast.cuh"
//     ...
//     auto plan = std::make_unique<crt_fast::CrtFastPlan>(
//         pl.crt_weights, pl.L, pl.K + 1);
//     plan->run(dev->residues_device(), s.d_c, dev->crt_args(), B, stream);
//     // plan->run_simple(...) is the byte-identical reference for A/B testing.
//
// ---------------------------------------------------------------------------
// WHAT IS COMPUTED (byte-for-byte identical to gpu31::reconstruct_kernel)
// ---------------------------------------------------------------------------
// For each (pair, coefficient) and prime j in [0,K):
//   t_j        = r_j * (P/p_j)^-1 mod p_j                 (Shoup)
//   correction = round( sum_j t_j / p_j )                  (binary64)
//   X          = sum_j t_j * W[j] + correction * Wcorr      (integer, L limbs)
//   out        = X mod q, canonical [0,q)                  (gap-fold reduction)
// W[j] = (P/p_j) mod q and Wcorr = q - (P mod q) are the columns of the
// weights table built by gpu31::Plan::build_crt().  Same helpers, same order
// of operations, same output as the reference.
//
// ---------------------------------------------------------------------------
// DIFFERENCES FROM crt_mma.cuh AND WHY THEY ARE SAFE
// ---------------------------------------------------------------------------
// 1. CRT_TILE_C 16 -> 32 (one full warp), CRT_TILE_Y stays 8, so the block is
//    256 threads and the grid halves (4096*128 -> 2048*128 blocks at the
//    official point).  The pad stride goes 17 -> 33 (still ODD, so consecutive
//    coefficient lanes never collide in a bank).  Dynamic shared memory is
//        terms(pitch*33 u32) + low(33*L u64) + high(33*L u64)
//      = 118*33*4 + 2*55*33*8 = 44616 bytes < 48 KiB  (no cudaFuncSetAttribute
//      opt-in needed).  This was verified for L<=55/pitch<=118; check() rejects
//    larger shapes at launch time.
//
// 2. The 2nd __syncthreads() is REMOVED.  The FP64 correction is computed by
//    the y==0 warp into a private register (no shared write, no barrier).  The
//    correction column is *excluded* from the per-limb product sums in step 3,
//    and its contribution (corr * Wcorr[i]) is added directly to low[i]/high[i]
//    inside step 4 by that same y==0 thread.  Since sum_column only accumulates
//    independent 64-bit lo/hi sums, splitting column K out and adding it later
//    is associative -> byte-identical.  CRUCIALLY the correction is still
//    computed ONCE per coefficient (not once per y-thread): no extra FP64.
//
// 3. The product-sum loop is transposed: for each j (read ONE term from shared
//    into a register), accumulate into the LPT per-thread limb accumulators.
//    The reference re-reads each shared term once per limb (LPT~7 times); here
//    each term is read from shared exactly once -> ~7x fewer shared loads,
//    usually the dominant cost of this kernel.  Accumulators are kept in
//    registers by fully unrolling the (compile-time LPT) limb loop.
//
// 4. `digits` reuses the `terms` shared buffer: by step 4 every term has been
//    consumed, so the 118*33 words are dead.  Writing the (L+2)*33 digit words
//    there is in-bounds and frees the ~7.5 KiB that would otherwise push the
//    block over 48 KiB.
//
// 5. The gap-fold fast path for q = 2^W - gap (our point has small nonzero gap)
//    is kept verbatim; the generic 43-shift loop remains only as a fallback
//    for the rare iteration where the fold does not converge in 5 steps.
//
// RISKS
//   - Templating on LPT (limbs per worker) adds 5 nearly identical kernel
//     instantiations; the dispatcher in run() selects the bucket.  All use the
//     exact same arithmetic.  If l changes, a different (still exact) instance
//     is used.  Selftest exercises the math; solve.cu's gpu31_selfcheck-style
//     byte comparison is the authoritative gate.
//   - The transposed product-sum loop relies on the compiler keeping the fully
//     unrolled acc_lo/acc_hi arrays in registers.  If a compiler spills them,
//     the kernel is still correct, just slower; there is no aliasing hazard.
// ===========================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstddef>

#include <cuda_runtime.h>          // shim provides the error enum in host builds

#include "fherma.h"

namespace crt_fast {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__)
#define CRTFAST_FHD __host__ __device__
#else
#define CRTFAST_FHD
#endif

// Compile-time limits (host side checks enforce them).
static constexpr unsigned CRT_FAST_MAXL     = 128;   // max limbs
static constexpr unsigned CRT_FAST_MAXPITCH = 512;   // max K+1 columns
static constexpr unsigned CRT_FAST_TILE_C   = 32;    // coefficients per block (full warp)
static constexpr unsigned CRT_FAST_TILE_Y   = 8;     // threads per coefficient
static constexpr unsigned CRT_FAST_TS       = 33;    // ODD pad stride (TILE_C + 1)
static constexpr unsigned CRT_FAST_THREADS  = CRT_FAST_TILE_C * CRT_FAST_TILE_Y;  // 256

// Dynamic shared memory required by the kernel.  `digits` is NOT counted: it
// reuses the `terms` storage after step 3.  Keep the term word count even so
// the u64 arrays begin on an 8-byte boundary.
static inline std::size_t crt_fast_shared_bytes(unsigned l, unsigned pitch) {
    const unsigned terms_words = pitch * CRT_FAST_TS;
    const unsigned aligned_words = (terms_words + 1u) & ~1u;
    return (std::size_t)aligned_words * sizeof(u32)
         + (std::size_t)l * CRT_FAST_TS * sizeof(u64) * 2u;
}

// ---------------------------------------------------------------------------
// Device-side mirrors of gpu31's structs (layout-identical).
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
// 1. Helpers copied verbatim from gpu31.cuh's fherma_crt section.
//    shifted_word intentionally uses q[i] (the reference's Stride=1 access):
//    q is a plain L-word array and carries no tile stride.
// ===========================================================================
CRTFAST_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template <unsigned Stride>
CRTFAST_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}

template <unsigned Stride>
CRTFAST_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
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
CRTFAST_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}

template <unsigned Stride>
CRTFAST_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
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
CRTFAST_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes, unsigned count) {
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}

// Reference sum_column (used only by run_simple / selftest).
template <unsigned Stride>
CRTFAST_FHD inline void sum_column(const u32* terms, const u32* basis_column,
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
CRTFAST_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
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
// 2. CUDA-core reference kernel (run_simple path).  Byte-identical copy of
//    gpu31::reconstruct_kernel / crt_mma::crt_simple_kernel, stride 1.
// ===========================================================================
#if defined(__CUDACC__)
__global__ void crt_fast_simple_kernel(CrtArgs plan, const u32* residues,
                                       const u32* weights, u32* out) {
    const unsigned coef = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = blockIdx.y;
    if (coef >= plan.n) return;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;

    u32 terms[CRT_FAST_MAXPITCH];
    u64 low[CRT_FAST_MAXL], high[CRT_FAST_MAXL];
    u32 digits[CRT_FAST_MAXL + 2];

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
// 3. Production kernel.
//
//    One block handles CRT_FAST_TILE_C=32 coefficients for one pair; its 256
//    threads form a 32 (m) x 8 (y) grid.  Shared storage uses ODD stride 33:
//        terms[j*TS + m]   pitch rows of per-coefficient terms
//        low[i*TS + m]     per-limb low  32 bits (as u64)
//        high[i*TS + m]    per-limb high bits
//    After step 3 `terms` is dead and is reused as `digits[i*TS + m]`.
//
//    LPT = max limbs any single worker (fixed y) must own = ceil(L/8).  It is a
//    template parameter so the limb accumulator arrays are fully unrolled into
//    registers rather than indexed dynamically into local memory.
// ===========================================================================
#if defined(__CUDACC__)
template <unsigned LPT>
__global__ void crt_fast_kernel(CrtArgs plan, const u32* residues,
                                const u32* d_weights, u32* out) {
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned m = threadIdx.x & (CRT_FAST_TILE_C - 1);  // 0..31
    const unsigned y = threadIdx.x >> 5;                      // 0..7
    const unsigned coef = blockIdx.x * CRT_FAST_TILE_C + m;
    const unsigned pair = blockIdx.z;
    const bool active = coef < plan.n;

    extern __shared__ unsigned char smem[];
    u32* terms = reinterpret_cast<u32*>(smem);
    u64* low = reinterpret_cast<u64*>(terms + ((pitch * CRT_FAST_TS + 1u) & ~1u));
    u64* high = low + l * CRT_FAST_TS;

    const u64 rbase = u64(pair) * k * plan.n;

    // ---- step 1: Shoup terms t_j = r_j * (P/p_j)^-1 mod p_j (j < k) --------
    for (unsigned j = y; j < k; j += CRT_FAST_TILE_Y) {
        const u32 r = active ? residues[rbase + u64(j) * plan.n + coef] : 0u;
        terms[j * CRT_FAST_TS + m] = weight_residue(r, plan.primes[j]);
    }
    __syncthreads();

    // ---- step 2: FP64 correction, computed ONCE per coefficient by y==0 ----
    // Kept in a private register: no shared write, no second barrier.
    u32 corr = 0;
    if (y == 0 && active) {
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
            sum += double(terms[j * CRT_FAST_TS + m]) * plan.primes[j].reciprocal;
        corr = u32(sum + 0.5);
    }

    // ---- step 3: exact per-limb product sums over j < k -------------------
    // Transposed loop: read each shared term ONCE, fan out to the LPT limb
    // accumulators held in registers.  (The reference re-reads it LPT times.)
    u64 acc_lo[LPT], acc_hi[LPT];
#pragma unroll
    for (unsigned s = 0; s < LPT; ++s) { acc_lo[s] = 0; acc_hi[s] = 0; }
    for (unsigned j = 0; j < k; ++j) {
        const u32 t = terms[j * CRT_FAST_TS + m];
#pragma unroll
        for (unsigned s = 0; s < LPT; ++s) {
            const unsigned i = y + s * CRT_FAST_TILE_Y;
            if (i < l) {
                const u64 prod = u64(t) * d_weights[u64(i) * pitch + j];
                acc_lo[s] += u32(prod);
                acc_hi[s] += prod >> 32;
            }
        }
    }
#pragma unroll
    for (unsigned s = 0; s < LPT; ++s) {
        const unsigned i = y + s * CRT_FAST_TILE_Y;
        if (i < l) {
            low[i * CRT_FAST_TS + m] = acc_lo[s];
            high[i * CRT_FAST_TS + m] = acc_hi[s];
        }
    }
    __syncthreads();

    // ---- step 4: fold in correction column, carry, reduce, write ----------
    if (y == 0 && active) {
        u32* digits = terms;                    // terms is dead after step 3
        const u32* wcorr = d_weights + k;       // column K = (-P) mod q
        u64 carry = 0, prior_high = 0;
        for (unsigned i = 0; i < l; ++i) {
            const u64 cp = u64(corr) * wcorr[u64(i) * pitch];
            const u64 lo = low[i * CRT_FAST_TS + m] + u32(cp);
            const u64 hi = high[i * CRT_FAST_TS + m] + (cp >> 32);
            carry += lo + prior_high;
            digits[i * CRT_FAST_TS + m] = u32(carry);
            carry >>= 32;
            prior_high = hi;
        }
        carry += prior_high;
        digits[l * CRT_FAST_TS + m] = u32(carry);
        digits[(l + 1) * CRT_FAST_TS + m] = u32(carry >> 32);
        reduce_sum<CRT_FAST_TS>(digits + m, l, plan.bits, plan.gap, plan.q);

        const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
#pragma unroll
        for (unsigned i = 0; i < l; ++i)
            out[dst + i] = digits[i * CRT_FAST_TS + m];
    }
}
#endif  // __CUDACC__

// ===========================================================================
// 4. Host-only self-test of the reduction math (plain C++, no CUDA needed).
// ---------------------------------------------------------------------------
// Exercises the two things this kernel changes relative to the reference:
//   (a) weight_residue's Shoup reduction equals a direct modular product;
//   (b) splitting the correction column out of the product sums and folding
//       it back in step 4 yields the SAME low/high/digits as sum_column over
//       all K+1 columns, and reduce_sum returns a canonical value < q.
// Small moduli keep every intermediate inside u64 (no __uint128_t dependency).
// ===========================================================================
inline bool crt_fast_selftest(unsigned iterations = 2048,
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

    // CRT weights, W[j] = (P/p_j) mod q, Wcorr = q - (P mod q).
    u32 W[pitch];
    const u32 PmodQ = u32(P % qv);
    for (unsigned j = 0; j < K; ++j) W[j] = u32((P / u64(primes[j])) % u64(qv));
    W[K] = PmodQ ? qv - PmodQ : 0u;

    PrimeInfo pi[K];
    for (unsigned j = 0; j < K; ++j) {
        const u32 p = primes[j];
        const u64 basis = P / u64(p);
        // inverse of (P/p) mod p via Fermat (p prime).
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

        // Full reference: sum_column over all K+1 columns.
        u64 lo_full = 0, hi_full = 0;
        sum_column<1>(terms, W, pitch, lo_full, hi_full);
        u32 dig_full[CRT_FAST_MAXL + 2] = {};
        carry_and_reduce<1>(dig_full, &lo_full, &hi_full, 1, bits, gap, &qv);
        if (dig_full[0] >= qv) return false;                    // canonical

        // Split path: sum j<K, then fold column K in afterwards.
        u32 terms_k[pitch];
        for (unsigned j = 0; j < pitch; ++j) terms_k[j] = terms[j];
        terms_k[K] = 0;
        u64 lo_split = 0, hi_split = 0;
        sum_column<1>(terms_k, W, pitch, lo_split, hi_split);
        const u64 cp = u64(terms[K]) * W[K];
        lo_split += u32(cp);
        hi_split += cp >> 32;
        if (lo_split != lo_full || hi_split != hi_full) return false;
        u32 dig_split[CRT_FAST_MAXL + 2] = {};
        carry_and_reduce<1>(dig_split, &lo_split, &hi_split, 1, bits, gap, &qv);
        if (dig_split[0] != dig_full[0]) return false;
    }
    return true;
}

// ===========================================================================
// 5. Host-side plan + launchers (mirrors crt_mma::CrtMmaPlan API).
// ===========================================================================
struct CrtFastPlan {
    unsigned L = 0, pitch = 0;
    u32* d_w = nullptr;                 // L*pitch u32, [limb][column]
    bool ok = false;

    CrtFastPlan() = default;

    CrtFastPlan(const std::vector<u32>& weights_host, unsigned L_, unsigned pitch_)
        : L(L_), pitch(pitch_) {
#ifdef __CUDACC__
        if (L == 0 || pitch == 0 || L > CRT_FAST_MAXL || pitch > CRT_FAST_MAXPITCH)
            throw std::runtime_error("crt_fast: L/pitch outside compiled limits");
        if (weights_host.size() != std::size_t(L) * pitch)
            throw std::runtime_error("crt_fast: weights table has wrong size");
        if (crt_fast_shared_bytes(L, pitch) > 48u * 1024u)
            throw std::runtime_error("crt_fast: shared memory exceeds 48 KiB");
        const std::size_t bytes = weights_host.size() * sizeof(u32);
        if (cudaMalloc(reinterpret_cast<void**>(&d_w), bytes) != cudaSuccess)
            throw std::runtime_error("crt_fast: cudaMalloc failed");
        if (cudaMemcpy(d_w, weights_host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_w);
            d_w = nullptr;
            throw std::runtime_error("crt_fast: weight upload failed");
        }
        ok = true;
#else
        (void)weights_host;
        throw std::runtime_error("crt_fast: constructed without CUDA");
#endif
    }

    CrtFastPlan(const CrtFastPlan&) = delete;
    CrtFastPlan& operator=(const CrtFastPlan&) = delete;

    ~CrtFastPlan() {
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
        ca.weights = d_w;
        ca.q = args.q;
        return ca;
    }

    template <class Args>
    bool check(const Args& args, u32 B, const u32* residues, const u32* out) const {
        return ok && B > 0 && residues && out && args.primes && args.q &&
               args.limbs > 0 && args.limbs <= CRT_FAST_MAXL &&
               args.count + 1 <= CRT_FAST_MAXPITCH &&
               args.count + 1 == pitch && args.limbs == L &&
               crt_fast_shared_bytes(L, pitch) <= 48u * 1024u;
    }

    // ---- PRIMARY: exact tiled CUDA-core CRT -------------------------------
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_fast_shared_bytes(L, pitch);
        const dim3 grid((ca.n + CRT_FAST_TILE_C - 1) / CRT_FAST_TILE_C, 1, B);
        const unsigned l = ca.limbs;
        // Dispatch on LPT = max limbs owned by one y-worker = ceil(L/8).
        if (l <= 56u)
            crt_fast_kernel<7><<<grid, CRT_FAST_THREADS, smem, stream>>>(
                ca, residues, d_w, out);
        else if (l <= 72u)
            crt_fast_kernel<9><<<grid, CRT_FAST_THREADS, smem, stream>>>(
                ca, residues, d_w, out);
        else if (l <= 88u)
            crt_fast_kernel<11><<<grid, CRT_FAST_THREADS, smem, stream>>>(
                ca, residues, d_w, out);
        else if (l <= 104u)
            crt_fast_kernel<13><<<grid, CRT_FAST_THREADS, smem, stream>>>(
                ca, residues, d_w, out);
        else
            crt_fast_kernel<16><<<grid, CRT_FAST_THREADS, smem, stream>>>(
                ca, residues, d_w, out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }

    // ---- Alias kept for API compatibility with crt_mma::CrtMmaPlan --------
    template <class Args>
    cudaError_t run_s8(const u32* residues, u32* out, const Args& args, u32 B,
                       cudaStream_t stream = nullptr) {
        return run(residues, out, args, B, stream);
    }

    // ---- Exact CUDA-core reference (byte-identical to reconstruct_kernel) -
    template <class Args>
    cudaError_t run_simple(const u32* residues, u32* out, const Args& args, u32 B,
                           cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_fast_simple_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, d_w, out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }
};

}  // namespace crt_fast

#undef CRTFAST_FHD
