// ntt_fast.cuh — faster CUDA-core replacement for gpu31's tiled DIT NTT.
//
// Header-only. Creates ONLY this file; integrate by including it AFTER
// gpu31.cuh in an nvcc build:
//
//     #include "gpu31.cuh"
//     #include "ntt_fast.cuh"
//     ...
//     ntt_fast::NttFastPlan nf(*dev);          // duck-typed gpu31::Device
//     nf.forward(dev->d_ra, dev->d_rb, *dev, B, stream);   // fwd NTT (both arrays)
//     nf.inverse(dev->d_rc, nullptr,    *dev, B, stream);   // inv NTT (one array)
//     // or: nf.run_fwd(...) / nf.run_inv(...)
//
// No tensor cores, no inline PTX, no cuPQC.  The butterfly arithmetic is
// gpu31::butterfly copied through verbatim, so the output is byte-for-byte
// identical to gpu31::transform.  The runner's gpu31_selfcheck gate therefore
// still passes.
//
// ===========================================================================
// WHAT IS DIFFERENT (and why it is safe)
// ===========================================================================
//
// 1. RUNTIME INTEGER DIVISION REMOVED  (the big one).
//    gpu31's low_pair()/high_pair() compute the twiddle exponent as
//        j * (n / (2*half))                 and
//        (within*512 + low) * (n / (2*half_high*512))
//    where n and half/half_high are RUNTIME values.  nvcc cannot prove the
//    divisor is a power of two, so every butterfly emits an integer-divide
//    sequence (~20-100 instructions).  There are ~2.4e10 butterflies at the
//    ranked point.  Here logN is passed as a parameter and the divisor becomes
//    a shift:
//        root = j << (logN - m - 1)                       (low,  half = 2^m)
//        root = idx << (logHigh - s - 1)                  (high, half_high = 2^s)
//    producing the EXACT same integer exponent, hence the exact same u32
//    twiddle read from the same table entry.
//
// 2. LOW-STAGE TWIDDLES PREPACKED + STAGED IN SHARED MEMORY.
//    At the ranked point (N=65536, tile=512), the 256 twiddles a low block
//    needs live at roots[k][0], roots[k][128], ..., roots[k][32640] — one
//    useful word per 32-byte sector.  The original inner loop re-reads them
//    from there nine times per block (~8 KiB of L2 sectors/block).  We build a
//    tiny [K][256] contiguous buffer on the host (K*256*4 = 120 KB at K=117),
//    load 256 consecutive words into shared once per block (1 KiB, 32 sectors),
//    and serve all nine stages from shared.  Same u32 values, same butterflies.
//
// 3. HIGH-STAGE TWIDDLES STAGED IN SHARED MEMORY.
//    A high block needs 16*(high_count-1) = 2032 twiddles across its seven
//    stages.  They are loaded once into a per-block shared buffer (8 KiB) with
//    the 16 lanes coalesced, then read from shared in the butterfly loop.
//    No new global buffer is needed here: the original roots table already
//    coalesces perfectly across the 16 lanes (see notes in the kernel).
//
// 4. CONFLICT-FREE SHARED DATA LAYOUT.
//    Low: sa/sb are padded with one word every 32 entries: pad(i)=i+(i>>5).
//    This turns the 2-way bank conflicts of the even-stride butterflies at
//    half=1,2,4,... into conflict-free accesses.  High: rows use stride 17
//    (odd) instead of 16.  Shared memory stays far below 48 KiB.
//
// 5. REDUNDANT BARRIER REMOVED FROM low_stages.
//    In the last low stage, thread t owns exactly the pair (t, t+512/2) it
//    writes back, so the post-final-stage __syncthreads() is unnecessary.
//
// RISKS / LIMITS
//   - Correctness relies on the shift identities above, which hold because N,
//     tile and high_count are powers of two (guaranteed by gpu31::Plan).  The
//     host selftest checks them exhaustively on small N against a plain DIT.
//   - Extra device memory: 2 * K * 256 * 4 = 240 KB for the packed low tables.
//   - Dynamically shared memory per block: low = arrays*528*4 + 1024 B
//     (<= 5 KiB); high = arrays*128*17*4 + 8128 B (<= 25 KiB).  Both < 48 KiB.
//   - The plan must be constructed from the Device (its `plan()` supplies the
//     host roots used to build the packed tables); do that once at setup.
// ===========================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
#include <cuda_runtime.h>
#endif

#include "gpu31.cuh"   // Prime, mont, butterfly, host_powmod, host_is_prime, Plan

namespace ntt_fast {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Prime = gpu31::Prime;

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
#define NTTFAST_FHD __host__ __device__
#else
#define NTTFAST_FHD
#endif

// gpu31 constants, kept local so this file stays drop-in.
static constexpr unsigned NF_MAX_TILE     = 512;   // gpu31::max_tile
static constexpr unsigned NF_LOW_COLUMNS  = 16;    // gpu31::low_columns
static constexpr unsigned NF_HIGH_WORKERS = 16;    // gpu31::high_workers

static constexpr unsigned nf_log2_c(unsigned x) { return x <= 1 ? 0u : 1u + nf_log2_c(x >> 1); }

NTTFAST_FHD inline unsigned nf_pad(unsigned x) { return x + (x >> 5); }  // one guard word / 32 lanes

// ===========================================================================
// 1. CUDA kernels (byte-identical butterflies to gpu31::low_stages/high_stages)
// ===========================================================================
#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)

// ---- low stages: log2(TILE) butterflies entirely inside one TILE strip -----
// TILE is a compile-time power of two (>= 512 on the real path).  `packed`
// layout: [prime][TILE/2], packed[k*(TILE/2)+p] = roots[k][p * (n/TILE)].
template <bool Two, unsigned TILE>
__global__ void low_stages_fast(u32* __restrict__ a, u32* __restrict__ b,
                                const Prime* __restrict__ primes,
                                const u32* __restrict__ packed,
                                unsigned n, unsigned count, unsigned /*logn*/) {
    constexpr unsigned LT = nf_log2_c(TILE);
    constexpr unsigned PT = TILE + TILE / 32;          // padded shared line count
    extern __shared__ u32 data[];
    u32* sa = data;
    u32* sb = data + PT;
    u32* sr = sb + (Two ? PT : 0u);

    const unsigned t = threadIdx.x;                    // 0 .. TILE/2-1
    const u64 base = u64(blockIdx.z) * count * n +
                     u64(blockIdx.y) * n + u64(blockIdx.x) * TILE;
    const Prime p = primes[blockIdx.y];

    // One coalesced global read of this block's twiddles, then shared only.
    sr[t] = packed[u64(blockIdx.y) * (TILE / 2) + t];

    sa[nf_pad(t)]             = a[base + t];
    sa[nf_pad(t + TILE / 2)]  = a[base + t + TILE / 2];
    if constexpr (Two) {
        sb[nf_pad(t)]            = b[base + t];
        sb[nf_pad(t + TILE / 2)] = b[base + t + TILE / 2];
    }
    __syncthreads();

    #pragma unroll
    for (unsigned m = 0; m < LT; ++m) {
        const unsigned half   = 1u << m;
        const unsigned j      = t & (half - 1);
        const unsigned first  = 2 * (t - j) + j;
        const unsigned second = first + half;
        // root = j * (n / (2*half)) / (n/TILE) = j * 2^(LT-1-m)  (exact).
        const u32 w = sr[j << (LT - 1 - m)];
        gpu31::butterfly(sa[nf_pad(first)], sa[nf_pad(second)], w, p);
        if constexpr (Two)
            gpu31::butterfly(sb[nf_pad(first)], sb[nf_pad(second)], w, p);
        // The final stage's outputs are exactly this thread's two stored
        // elements, so the trailing barrier of the original is redundant.
        if (m + 1 < LT) __syncthreads();
    }
    a[base + t]            = sa[nf_pad(t)];
    a[base + t + TILE / 2] = sa[nf_pad(t + TILE / 2)];
    if constexpr (Two) {
        b[base + t]            = sb[nf_pad(t)];
        b[base + t + TILE / 2] = sb[nf_pad(t + TILE / 2)];
    }
}

// ---- high stages: log2(n/TILE) butterflies across TILE strips -------------
// Grid/block are gpu31's: dim3(TILE/LOW_COLUMNS, K, B) blocks,
// dim3(LOW_COLUMNS, HIGH_WORKERS) threads.  sa/sb rows use an odd stride 17;
// sr holds [stage][within * LOW_COLUMNS + lane] twiddles.
template <bool Two, unsigned TILE, unsigned LOW_COLUMNS, unsigned HIGH_WORKERS>
__global__ void high_stages_fast(u32* __restrict__ a, u32* __restrict__ b,
                                 const Prime* __restrict__ primes,
                                 const u32* __restrict__ roots,
                                 unsigned n, unsigned count, unsigned logn) {
    constexpr unsigned LT = nf_log2_c(TILE);
    constexpr unsigned HS = LOW_COLUMNS + 1;           // odd pad stride
    const unsigned log_high  = logn - LT;
    const unsigned high_count = 1u << log_high;        // == n / TILE

    extern __shared__ u32 data[];
    u32* sa = data;
    u32* sb = data + high_count * HS;
    u32* sr = sb + (Two ? high_count * HS : 0u);

    const unsigned lane       = threadIdx.x;           // 0 .. LOW_COLUMNS-1
    const unsigned row_worker = threadIdx.y;           // 0 .. HIGH_WORKERS-1
    const unsigned low  = blockIdx.x * LOW_COLUMNS + lane;
    const u64 plane = u64(blockIdx.z) * count * n + u64(blockIdx.y) * n;
    const u32* table = roots + u64(blockIdx.y) * (n / 2);
    const Prime p = primes[blockIdx.y];

    // Stage twiddles -> shared once.  For a stage s with half_high=2^s,
    //   stride = n/(2*half_high*TILE) = 2^(log_high-s-1)
    // and the needed root is table[(within*TILE + low)*stride].  The original
    // inner loop computes exactly this with an integer division; here it is a
    // shift.  Lanes are consecutive in `low`, so each 16-thread half-warp
    // fetches two 32-B sectors — the minimum possible for these words.
    for (unsigned s = 0; s < log_high; ++s) {
        const unsigned half_high = 1u << s;
        const unsigned stride    = 1u << (log_high - s - 1);
        const unsigned off       = LOW_COLUMNS * (half_high - 1);
        for (unsigned wi = row_worker; wi < half_high; wi += HIGH_WORKERS)
            sr[off + wi * LOW_COLUMNS + lane] = table[(wi * TILE + low) * stride];
    }

    for (unsigned h = row_worker; h < high_count; h += HIGH_WORKERS) {
        const u64 g = plane + u64(h) * TILE + low;
        sa[h * HS + lane] = a[g];
        if constexpr (Two) sb[h * HS + lane] = b[g];
    }
    __syncthreads();

    for (unsigned s = 0; s < log_high; ++s) {
        const unsigned half_high = 1u << s;
        const unsigned off       = LOW_COLUMNS * (half_high - 1);
        for (unsigned h = row_worker; h < high_count / 2; h += HIGH_WORKERS) {
            const unsigned within = h & (half_high - 1);
            const unsigned first  = 2 * (h - within) + within;
            const unsigned second = first + half_high;
            const u32 w = sr[off + within * LOW_COLUMNS + lane];
            gpu31::butterfly(sa[first * HS + lane], sa[second * HS + lane], w, p);
            if constexpr (Two)
                gpu31::butterfly(sb[first * HS + lane], sb[second * HS + lane], w, p);
        }
        __syncthreads();
    }

    for (unsigned h = row_worker; h < high_count; h += HIGH_WORKERS) {
        const u64 g = plane + u64(h) * TILE + low;
        a[g] = sa[h * HS + lane];
        if constexpr (Two) b[g] = sb[h * HS + lane];
    }
}

// ---- small-n fallback (N < 512): single-block DIT, gpu31 arithmetic -------
template <bool Two>
__global__ void ntt_small(u32* __restrict__ a, u32* __restrict__ b,
                          const Prime* __restrict__ primes,
                          const u32* __restrict__ roots,
                          unsigned n, unsigned count, unsigned logn) {
    extern __shared__ u32 data[];
    u32* sa = data;
    u32* sb = data + n;
    const unsigned t = threadIdx.x;                    // 0 .. n/2-1
    const u64 base = u64(blockIdx.z) * count * n + u64(blockIdx.y) * n;
    const u32* table = roots + u64(blockIdx.y) * (n / 2);
    const Prime p = primes[blockIdx.y];
    sa[t] = a[base + t];
    sa[t + n / 2] = a[base + t + n / 2];
    if constexpr (Two) {
        sb[t] = b[base + t];
        sb[t + n / 2] = b[base + t + n / 2];
    }
    __syncthreads();
    for (unsigned m = 0; m < logn; ++m) {
        const unsigned half = 1u << m;
        const unsigned j = t & (half - 1);
        const unsigned first = 2 * (t - j) + j;
        const unsigned second = first + half;
        const u32 w = table[j << (logn - m - 1)];      // j*(n/(2*half)), exact
        gpu31::butterfly(sa[first], sa[second], w, p);
        if constexpr (Two) gpu31::butterfly(sb[first], sb[second], w, p);
        __syncthreads();
    }
    a[base + t] = sa[t];
    a[base + t + n / 2] = sa[t + n / 2];
    if constexpr (Two) {
        b[base + t] = sb[t];
        b[base + t + n / 2] = sb[t + n / 2];
    }
}

#endif  // __CUDACC__ && !PGPU_HOST_CHECK

// ===========================================================================
// 2. Host-side plan and launchers
// ===========================================================================
struct NttFastPlan {
    unsigned N = 0, K = 0, logN = 0;
    u32* d_packed_fwd = nullptr;    // [K][NF_MAX_TILE/2]
    u32* d_packed_inv = nullptr;

    NttFastPlan() = default;

    // Duck-typed construction from a gpu31::Device (or any object exposing
    // .plan() with .N, .K and host vectors .roots / .inverse_roots).
    template <class Dev>
    explicit NttFastPlan(const Dev& dev) { init_from(dev.plan()); }

    NttFastPlan(const NttFastPlan&) = delete;
    NttFastPlan& operator=(const NttFastPlan&) = delete;
    ~NttFastPlan() {
#ifdef __CUDACC__
        if (d_packed_fwd) cudaFree(d_packed_fwd);
        if (d_packed_inv) cudaFree(d_packed_inv);
#endif
        d_packed_fwd = d_packed_inv = nullptr;
    }

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
    // ---- forward (two arrays) / inverse (one array) ----------------------
    template <class Dev>
    cudaError_t forward(u32* a, u32* b, const Dev& dev, u32 B,
                        cudaStream_t stream = nullptr) const {
        return transform<true>(a, b, dev.primes_device(), dev.roots_device(),
                               d_packed_fwd, dev.plan().N, dev.plan().K, B, stream);
    }
    template <class Dev>
    cudaError_t inverse(u32* a, u32* b, const Dev& dev, u32 B,
                        cudaStream_t stream = nullptr) const {
        return transform<false>(a, b, dev.primes_device(), dev.inv_roots_device(),
                                d_packed_inv, dev.plan().N, dev.plan().K, B, stream);
    }
    template <class Dev>
    cudaError_t run_fwd(u32* a, u32* b, const Dev& dev, u32 B,
                        cudaStream_t stream = nullptr) const { return forward(a, b, dev, B, stream); }
    template <class Dev>
    cudaError_t run_inv(u32* a, u32* b, const Dev& dev, u32 B,
                        cudaStream_t stream = nullptr) const { return inverse(a, b, dev, B, stream); }

    // Raw-pointer overloads (mirror gpu31::transform's launch inputs).
    cudaError_t forward(u32* a, u32* b, const Prime* primes, const u32* roots,
                        u32 B, cudaStream_t stream = nullptr) const {
        return transform<true>(a, b, primes, roots, d_packed_fwd, N, K, B, stream);
    }
    cudaError_t inverse(u32* a, u32* b, const Prime* primes, const u32* roots,
                        u32 B, cudaStream_t stream = nullptr) const {
        return transform<false>(a, b, primes, roots, d_packed_inv, N, K, B, stream);
    }
#endif  // __CUDACC__ && !PGPU_HOST_CHECK  (public CUDA API)

private:
    template <class P>
    void init_from(const P& pl) {
        N = pl.N;
        K = pl.K;
        logN = 0;
        while ((1u << logN) < N) ++logN;
        if (N < 2 || (N & (N - 1)) || K == 0)
            throw std::runtime_error("ntt_fast: N must be a power of two >= 2 and K > 0");
        if (pl.roots.size() < std::size_t(K) * (N / 2) ||
            pl.inverse_roots.size() < std::size_t(K) * (N / 2))
            throw std::runtime_error("ntt_fast: root tables have wrong size");
#ifdef __CUDACC__
        const std::size_t packed_words = std::size_t(K) * (NF_MAX_TILE / 2);
        d_packed_fwd = upload_packed(pl.roots, packed_words);
        d_packed_inv = upload_packed(pl.inverse_roots, packed_words);
#endif
    }

#ifdef __CUDACC__
    u32* upload_packed(const std::vector<u32>& roots, std::size_t packed_words) {
        std::vector<u32> packed(packed_words, 0u);
        const unsigned tile = N < NF_MAX_TILE ? N : NF_MAX_TILE;
        const unsigned stride = N / tile;              // n/TILE (power of two)
        for (unsigned k = 0; k < K; ++k) {
            const u32* src = roots.data() + std::size_t(k) * (N / 2);
            u32* dst = packed.data() + std::size_t(k) * (NF_MAX_TILE / 2);
            // For N < 512 only the first N/2 entries per prime are meaningful;
            // the small-N kernel reads the original roots directly anyway.
            const unsigned pcount = (tile / 2 < NF_MAX_TILE / 2) ? tile / 2 : NF_MAX_TILE / 2;
            for (unsigned p = 0; p < pcount; ++p) dst[p] = src[p * stride];
        }
        u32* d = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&d), packed_words * sizeof(u32)) != cudaSuccess)
            throw std::runtime_error("ntt_fast: cudaMalloc failed");
        if (cudaMemcpy(d, packed.data(), packed_words * sizeof(u32), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d);
            throw std::runtime_error("ntt_fast: table upload failed");
        }
        return d;
    }
#endif

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
    template <bool Two>
    cudaError_t transform(u32* a, u32* b, const Prime* primes, const u32* roots,
                          const u32* packed, unsigned n, unsigned k, u32 batches,
                          cudaStream_t stream) const {
        if (batches == 0) return cudaSuccess;
        if (!a || !primes || !roots || !packed || (Two && !b)) return cudaErrorInvalidValue;
        const unsigned arrays = Two ? 2u : 1u;
        if (n < NF_MAX_TILE) {
            // N is a power of two, so n/2 <= 256 threads.
            const std::size_t smem = std::size_t(arrays) * n * sizeof(u32);
            ntt_small<Two><<<dim3(1, k, batches), n / 2, smem, stream>>>(
                a, b, primes, roots, n, k, logN);
            return cudaGetLastError();
        }
        const unsigned tile = NF_MAX_TILE;
        const std::size_t low_smem =
            std::size_t(arrays) * (tile + tile / 32) * sizeof(u32) +
            (tile / 2) * sizeof(u32);
        low_stages_fast<Two, NF_MAX_TILE><<<dim3(n / tile, k, batches), tile / 2,
                                            low_smem, stream>>>(
            a, b, primes, packed, n, k, logN);
        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess || n == tile) return st;
        const unsigned high_count = n / tile;
        const std::size_t high_smem =
            std::size_t(arrays) * high_count * (NF_LOW_COLUMNS + 1) * sizeof(u32) +
            std::size_t(NF_LOW_COLUMNS) * (high_count - 1) * sizeof(u32);
        high_stages_fast<Two, NF_MAX_TILE, NF_LOW_COLUMNS, NF_HIGH_WORKERS>
            <<<dim3(NF_MAX_TILE / NF_LOW_COLUMNS, k, batches),
               dim3(NF_LOW_COLUMNS, NF_HIGH_WORKERS), high_smem, stream>>>(
            a, b, primes, roots, n, k, logN);
        return cudaGetLastError();
    }
#endif  // __CUDACC__ && !PGPU_HOST_CHECK  (CUDA-facing API)
};

// ===========================================================================
// 3. Pure-host selftest: DIT butterfly math vs a reference CPU NTT.
//
// Simulates the low/high kernel loops (including the shift-based root
// indices) on small N and compares, word for word, with a textbook
// bit-reversed-input DIT using the same Montgomery twiddle table.  Exercises
// both the low-only (tile = N) and low+high (tile = N/4) decompositions and
// both forward and inverse root tables.  No CUDA needed.
// ===========================================================================
namespace detail {

inline unsigned nf_log2(unsigned x) { unsigned r = 0; while ((1u << r) < x) ++r; return r; }

inline bool find_prime_for_n(unsigned n, u32& p) {
    const u64 m = 2ull * n;
    u64 a = (u64(1) << 30) / m + 1;
    for (unsigned tries = 0; tries < 20000; ++tries, ++a) {
        u64 cand = a * m + 1;
        if (cand >= (u64(1) << 31)) return false;
        p = u32(cand);
        if (gpu31::host_is_prime(p)) return true;
    }
    return false;
}

inline u32 primitive_root(u32 p, unsigned order) {
    for (u32 g = 2; g < 1u << 20; ++g) {
        const u32 r = gpu31::host_powmod(g, (p - 1) / order, p);
        if (gpu31::host_powmod(r, order / 2, p) != p - 1) continue;
        bool ok = true;
        for (unsigned q = 2; q * q <= order; ++q) {
            if (order % q) continue;
            if (gpu31::host_powmod(r, order / q, p) == 1) { ok = false; break; }
        }
        if (ok) return r;
    }
    return 0;
}

// Textbook iterative DIT; input is in bit-reversed order, output natural.
inline void ref_dit(std::vector<u32>& v, const std::vector<u32>& roots,
                    unsigned n, const Prime& pr) {
    const unsigned logn = nf_log2(n);
    for (unsigned len = 2; len <= n; len <<= 1) {
        const unsigned half = len >> 1;
        for (unsigned i = 0; i < n; i += len)
            for (unsigned j = 0; j < half; ++j) {
                const u32 w = roots[j << (logn - nf_log2(len))];  // j * (n/len)
                gpu31::butterfly(v[i + j], v[i + j + half], w, pr);
            }
    }
}

// CPU emulation of low_stages_fast + high_stages_fast (one prime, one pair).
inline void sim_fast(std::vector<u32>& v, const std::vector<u32>& roots,
                     const std::vector<u32>& packed, unsigned n,
                     unsigned tile, const Prime& pr) {
    const unsigned logn = nf_log2(n), LT = nf_log2(tile);
    // ---- low stages ----
    for (unsigned bx = 0; bx < n / tile; ++bx) {
        std::vector<u32> s(v.begin() + bx * tile, v.begin() + (bx + 1) * tile);
        for (unsigned m = 0; m < LT; ++m) {
            const unsigned half = 1u << m;
            for (unsigned t = 0; t < tile / 2; ++t) {
                const unsigned j = t & (half - 1);
                const unsigned first = 2 * (t - j) + j;
                const unsigned second = first + half;
                const u32 w = packed[j << (LT - 1 - m)];
                gpu31::butterfly(s[first], s[second], w, pr);
            }
        }
        for (unsigned i = 0; i < tile; ++i) v[bx * tile + i] = s[i];
    }
    if (tile == n) return;
    // ---- high stages ----
    const unsigned log_high = logn - LT;
    const unsigned high_count = 1u << log_high;
    std::vector<u32> s(v);                       // row h, column c -> s[h*tile+c]
    for (unsigned st = 0; st < log_high; ++st) {
        const unsigned half_high = 1u << st;
        const unsigned stride = 1u << (log_high - st - 1);
        for (unsigned h = 0; h < high_count / 2; ++h) {
            const unsigned within = h & (half_high - 1);
            const unsigned first = 2 * (h - within) + within;
            const unsigned second = first + half_high;
            for (unsigned c = 0; c < tile; ++c) {
                const u32 w = roots[(within * tile + c) * stride];
                gpu31::butterfly(s[first * tile + c], s[second * tile + c], w, pr);
            }
        }
    }
    v = s;
}

inline bool check_one(unsigned n, unsigned tile, u64& seed) {
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed << 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };
    u32 p = 0;
    if (!find_prime_for_n(n, p)) return false;
    const u32 psi = primitive_root(p, 2 * n);
    if (!psi) return false;
    const u32 omega = u32(u64(psi) * psi % p);
    const u32 omega_inv = gpu31::host_powmod(omega, p - 2, p);

    u32 inv = 1;
    for (unsigned i = 0; i < 5; ++i) inv *= 2 - p * inv;
    const u32 one = u32((u64(1) << 32) % p);
    const Prime pr{p, u32(0 - inv), u32(u64(one) * one % p), one};

    std::vector<u32> roots(n / 2), invroots(n / 2);
    {
        u32 a = 1, b = 1;
        for (unsigned i = 0; i < n / 2; ++i) {
            roots[i] = gpu31::mont(a, pr.r2, pr);
            invroots[i] = gpu31::mont(b, pr.r2, pr);
            a = u32(u64(a) * omega % p);
            b = u32(u64(b) * omega_inv % p);
        }
    }
    std::vector<u32> packed(n == tile ? 0 : tile / 2);   // filled per tile below

    for (int dir = 0; dir < 2; ++dir) {
        const std::vector<u32>& rt = dir ? invroots : roots;
        packed.assign(tile / 2, 0u);
        const unsigned stride = n / tile;
        for (unsigned q = 0; q < tile / 2; ++q) packed[q] = rt[q * stride];

        std::vector<u32> v(n);
        for (unsigned i = 0; i < n; ++i) {
            const u32 x = u32(next()) % p;
            v[i] = gpu31::mont(x, pr.r2, pr);             // Montgomery residue
        }
        std::vector<u32> want = v;
        ref_dit(want, rt, n, pr);
        sim_fast(v, rt, packed, n, tile, pr);
        if (v != want) return false;
    }
    return true;
}

}  // namespace detail

inline bool ntt_fast_selftest(u64 seed = 0x9e3779b97f4a7c15ull) {
    // (a) Exhaustive index identity: gpu31's runtime-division exponents equal
    //     this file's shift exponents, for every stage/lane of small AND the
    //     ranked (N=65536, tile=512) shape.
    for (unsigned n : {16u, 64u, 512u, 1024u, 65536u}) {
        const unsigned logn = detail::nf_log2(n);
        const unsigned tile = n < NF_MAX_TILE ? n : NF_MAX_TILE;
        const unsigned LT = detail::nf_log2(tile);
        // low stages: root = j*(n/(2*half)), half = 2^m.
        for (unsigned m = 0; m < LT; ++m) {
            const unsigned half = 1u << m;
            for (unsigned t = 0; t < tile / 2; ++t) {
                const unsigned j = t & (half - 1);
                if (j * (n / (2 * half)) != j * (1u << (logn - m - 1))) return false;
            }
        }
        if (n == tile) continue;
        // high stages: root = (within*tile+low)*(n/(2*half_high*tile)).
        const unsigned log_high = logn - LT;
        for (unsigned s = 0; s < log_high; ++s) {
            const unsigned half_high = 1u << s;
            for (unsigned h = 0; h < n / tile / 2; ++h) {
                const unsigned within = h & (half_high - 1);
                for (unsigned low = 0; low < tile; ++low) {
                    const unsigned idx = within * tile + low;
                    if (idx * (n / (2 * half_high * tile)) !=
                        idx * (1u << (log_high - s - 1))) return false;
                }
            }
        }
    }
    // (b) End-to-end butterfly math: simulated tiling vs textbook DIT.
    for (unsigned n : {16u, 32u, 64u}) {
        if (!detail::check_one(n, n, seed)) return false;        // low only
        if (!detail::check_one(n, n / 4, seed)) return false;    // low + high
    }
    // (c) The ranked-point decomposition shape, at a CPU-affordable size.
    if (!detail::check_one(1024, 512, seed)) return false;       // 9 low + 1 high
    if (!detail::check_one(1024, 128, seed)) return false;       // 7 low + 3 high
    return true;
}

}  // namespace ntt_fast

#undef NTTFAST_FHD
