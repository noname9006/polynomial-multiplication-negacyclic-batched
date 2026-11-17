// crt_cupqc.cuh — EXACT centered-CRT reconstruction using NVIDIA cuPQC-BigInt
// in warp-cooperative (TPI) mode.  Header-only; creates only this file.
//
// Integrate (do NOT modify this file) by including it after gpu31.cuh in
// solve.cu, exactly like crt_mma.cuh:
//
//     #include "gpu31.cuh"
//     #include "crt_cupqc.cuh"
//     ...
//     auto plan = std::make_unique<crt_cupqc::Plan>(
//         pl.crt_weights, pl.L, pl.K + 1, pl.crt_q, pl.K);
//     plan->run(dev->residues_device(), s.d_c, dev->crt_args(), B, stream);
//     // or, for A/B testing: plan->run_simple(...)
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE COMPUTES (identical math to gpu31::reconstruct_kernel)
// ---------------------------------------------------------------------------
// For every (pair, coefficient) and prime j in [0,K):
//
//   t_j       = r_j * (P/p_j)^-1  mod p_j          (Shoup, FP64-free)
//   correction= round( sum_j t_j / p_j )           (binary64, replicated)
//   X         = sum_j t_j * W[j] + correction * Wcorr      (integer)
//   out       = X mod q                                    (canonical [0,q))
//
// where W[j] = (P/p_j) mod q and Wcorr = q - (P mod q) = (-P) mod q are the
// columns of the weights table produced by gpu31::Plan::build_crt().  This is
// the centered representative C: since sum_j t_j*(P/p_j) = P*S with
// S = sum_j t_j/p_j, we have X = P*(S - round(S)) = C with |C| < P/2, and
// C mod q is the required coefficient.  X < 2^1786 here, so a 2048-bit
// accumulator never wraps and every cuPQC operation is exact.
//
// ---------------------------------------------------------------------------
// cuPQC API ASSUMPTIONS (verified against the cuPQC 0.6.0 on-line docs)
// ---------------------------------------------------------------------------
// * Header: <bigint.hpp>  (namespace cupqc).  The docs' canonical descriptor
//   is composed with operators.  cuPQC 0.6.0 requires a warp descriptor to
//   contain BOTH TPI<32>() and Warp(); we use:
//
//       using Desc = decltype(cupqc::BitWidth<2048>() + cupqc::SM<800>()
//                             + cupqc::TPI<32>() + cupqc::Warp()
//                             + cupqc::OnErrorNone());
//       using BI = typename Desc::bigint;
//
//   *** IMPORTANT CORRECTION TO THE TASK'S SUGGESTED TYPE ***
//   `bigint<1792, TPI<32>, ...>` (56 limbs) is NOT a shipped configuration:
//   for TPI<32> cuPQC only builds limb counts 64..128 (2048..4096 bits).
//   A 1792-bit instance compiles but FAILS TO LINK.  We therefore use 2048
//   bits (64 limbs), still far above the 55 limbs actually needed.
//
//   If your image really uses the older `cupqc::bigint<BW, TPI, Policy>`
//   spelling, define CRT_CUPQC_DESCRIPTOR to that type before including this
//   header (the only requirement is that `Desc::bigint`, `Desc::bigint_wide`,
//   `Desc::num_limbs` and `Desc::tpi` exist, and that the member names below
//   match).
//
// * Header probing: <bigint.hpp> (documented), then <cupqc/bigint.hpp>,
//   <cupqc/bigint.h>, <cupqc_bigint.h> via __has_include.  The real SDK puts
//   the header at $CUPQC_SDK_DIR/include/bigint.hpp.
//
// * Warp-mode restrictions that shaped this code (from "Warp Execution
//   Coverage" in the docs; calling a thread-only op on a Warp descriptor
//   compiles and then fails to link):
//     - operator+= / operator-= with a *bigint* RHS are thread-only  ==> we
//       always use the non-mutating `acc = acc + prod` (operator+ is covered).
//     - div_rem, operator/ and operator% are thread-only ==> we reduce with
//       Barrett (setup_barrett + reduce_barrett), which IS warp-capable.
//     - to_uint32/to_uint64 and operator[] are thread-only ==> results are
//       read back with store() into shared memory.
//     - mul_scalar, add_scalar, operator+, mod, setup_barrett,
//       reduce_barrett, compare and the batched constructor/store are all
//       available in warp mode.
// * setup_barrett is a __device__ static function, so the "host constructor"
//   uploads q and then launches a tiny 1-warp init kernel (on stream 0,
//   synchronized) to produce the opaque (approx, den_clz) state.
// * reduce_barrett requires num.hi < den.  Our accumulator fits in one
//   2048-bit value, so we build a bigint_wide with hi == 0 and den = q > 0.
// * Linking: the caller must link the cuPQC BigInt static library with LTO,
//   e.g. -L$STATIC_LIB -lcupqc-bigint -dlto (see docs).  If that is missing,
//   this header compiles but the link fails; `run_simple()` needs no cuPQC.
//
// ---------------------------------------------------------------------------
// LAUNCH GEOMETRY AND SAFETY
// ---------------------------------------------------------------------------
// One 32-lane warp = one coefficient (TPI<32> = one bigint per warp).  A block
// has 256 threads = 8 warps = 8 coefficients; grid = ((N+7)/8, 1, B).  All 32
// lanes of a group execute identical cuPQC calls with identical scalar args
// and the SAME pointer arguments (weights, q, approx, shared scratch).
//
// Default reduction is Barrett (CRT_CUPQC_REDUCE_MODE=0).  Set
// CRT_CUPQC_REDUCE_MODE=1 to use the simpler (warp-capable) `mod`, or =2 to
// reduce the extracted limbs with the reference reduce_sum logic.  All three
// produce the same canonical remainder.  `run_simple()` is a pure CUDA-core
// copy of gpu31's reference CRT for A/B testing; it never touches cuPQC.
//
// RISKS (nothing could be compiled locally):
//   - The descriptor/member spellings above follow the 0.6.0 docs exactly,
//     but a custom FHERMA image could expose a different spelling; use the
//     CRT_CUPQC_DESCRIPTOR / CRT_CUPQC_SM_ARCH overrides in that case.
//   - SM<800> is the documented default; set CRT_CUPQC_SM_ARCH to your GPU's
//     compute capability (e.g. 900, 1200) if the shipped library dispatches
//     per-architecture.  This does NOT change nvcc's -arch flag.
//   - Barrett is silent under OnErrorNone; the precondition (hi=0 < q) is
//     established by construction, and q != 0 is validated in the Plan ctor.
//   - cuPQC ops may use warp shuffles; __syncwarp() (not __syncthreads()) is
//     used between our shared-memory stores and loads inside a warp group.
//   - We construct bigint_wide from a *shared-memory* pointer (its documented
//     constructor is pointer-generic). If an image's implementation assumes
//     global memory there, switch CRT_CUPQC_REDUCE_MODE to 1 (`mod`), which
//     needs no wide value and no shared-memory round-trip.
// ===========================================================================

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>   // shim provides this in non-CUDA host builds

#ifdef __CUDACC__
#  if defined(__has_include)
#    if __has_include(<bigint.hpp>)
#      include <bigint.hpp>
#      define CRT_CUPQC_HAVE_BIGINT_HPP 1
#    elif __has_include(<cupqc/bigint.hpp>)
#      include <cupqc/bigint.hpp>
#      define CRT_CUPQC_HAVE_BIGINT_HPP 1
#    elif __has_include(<cupqc/bigint.h>)
#      include <cupqc/bigint.h>
#      define CRT_CUPQC_HAVE_BIGINT_HPP 1
#    elif __has_include(<cupqc_bigint.h>)
#      include <cupqc_bigint.h>
#      define CRT_CUPQC_HAVE_BIGINT_HPP 1
#    endif
#  else
#    include <bigint.hpp>
#    define CRT_CUPQC_HAVE_BIGINT_HPP 1
#  endif
#endif

namespace crt_cupqc {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Minimal host shims (tools/fakecuda) do not declare cudaErrorInvalidValue;
// in the real CUDA runtime it equals 1, so a literal is portable and identical.
static constexpr cudaError_t kInvalidValue = static_cast<cudaError_t>(1);

#if defined(__CUDACC__)
#  define CRT_CUPQC_FHD __host__ __device__
#else
#  define CRT_CUPQC_FHD
#endif

// Compile-time limits for the run_simple() reference kernel.
static constexpr unsigned CRT_CUPQC_MAXL     = 128;   // max limbs
static constexpr unsigned CRT_CUPQC_MAXPITCH = 512;   // max K+1 columns
static constexpr unsigned CRT_CUPQC_WARPS    = 8;     // 8 warps/block
static constexpr unsigned CRT_CUPQC_THREADS  = CRT_CUPQC_WARPS * 32;

// ---------------------------------------------------------------------------
// cuPQC descriptor / types.  Override CRT_CUPQC_DESCRIPTOR if your image uses
// a different descriptor spelling.
// ---------------------------------------------------------------------------
#if defined(CRT_CUPQC_HAVE_BIGINT_HPP)
#  ifndef CRT_CUPQC_SM_ARCH
#    define CRT_CUPQC_SM_ARCH 800        // documented default; see risks above
#  endif
#  ifndef CRT_CUPQC_DESCRIPTOR
#    define CRT_CUPQC_DESCRIPTOR                                              \
        decltype(::cupqc::BitWidth<2048>()                                    \
                 + ::cupqc::SM<CRT_CUPQC_SM_ARCH>()                           \
                 + ::cupqc::TPI<32>()                                         \
                 + ::cupqc::Warp()                                            \
                 + ::cupqc::OnErrorNone())
#  endif
using Desc = CRT_CUPQC_DESCRIPTOR;
using BI = typename Desc::bigint;
using WIDE = typename Desc::bigint_wide;
static_assert(Desc::num_limbs >= 56, "crt_cupqc: need at least 56 limbs");
static_assert(Desc::tpi == 32, "crt_cupqc: this path requires TPI<32>");
#  define CRT_CUPQC_CUPQC_AVAILABLE 1
#endif

#ifndef CRT_CUPQC_REDUCE_MODE
#  define CRT_CUPQC_REDUCE_MODE 0   // 0=Barrett, 1=mod, 2=reference reduce_sum
#endif

// ---------------------------------------------------------------------------
// Device-side mirrors of gpu31's CRT argument structs (layout-identical).
// ---------------------------------------------------------------------------
struct PrimeInfo {
    u32 p, inv, inv_shoup;
    double reciprocal;
};

struct CrtArgs {
    unsigned n, limbs, bits, count;
    u32 gap;
    const PrimeInfo* primes;
    const u32* weights;   // unused by the cuPQC path; used by run_simple()
    const u32* q;
};

// ===========================================================================
// 1. Helpers copied verbatim from gpu31.cuh's fherma_crt section (for the
//    exact, cuPQC-free run_simple() path and for REDUCE_MODE=2).
// ===========================================================================
CRT_CUPQC_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template <unsigned Stride>
CRT_CUPQC_FHD inline u32 shifted_word(const u32* q, unsigned limbs,
                                      unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}

template <unsigned Stride>
CRT_CUPQC_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
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
CRT_CUPQC_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}

template <unsigned Stride>
CRT_CUPQC_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits,
                                     u32 gap, const u32* q) {
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
CRT_CUPQC_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes,
                                        unsigned count) {
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}

template <unsigned Stride>
CRT_CUPQC_FHD inline void sum_column(const u32* terms, const u32* basis_column,
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
CRT_CUPQC_FHD inline void carry_and_reduce(u32* digits, const u64* low,
                                           const u64* high, unsigned l,
                                           unsigned bits, u32 gap, const u32* q) {
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

#if defined(__CUDACC__)

// ===========================================================================
// 2. Pure CUDA-core reference kernel (byte-identical to reconstruct_kernel).
//    No cuPQC dependency — safe even if <bigint.hpp> is unavailable.
// ===========================================================================
__global__ void crt_simple_kernel(CrtArgs plan, const u32* residues,
                                  const u32* weights, u32* out) {
    const unsigned coef = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pair = blockIdx.y;
    if (coef >= plan.n) return;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;

    u32 terms[CRT_CUPQC_MAXPITCH];
    u64 low[CRT_CUPQC_MAXL], high[CRT_CUPQC_MAXL];
    u32 digits[CRT_CUPQC_MAXL + 2];

    const u64 rbase = u64(pair) * k * plan.n;
    for (unsigned j = 0; j < k; ++j)
        terms[j] = weight_residue(residues[rbase + u64(j) * plan.n + coef],
                                  plan.primes[j]);
    terms[k] = crt_correction<1>(terms, plan.primes, k);
    for (unsigned i = 0; i < l; ++i)
        sum_column<1>(terms, weights + i * pitch, pitch, low[i], high[i]);
    carry_and_reduce<1>(digits, low, high, l, plan.bits, plan.gap, plan.q);

    const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
    for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i];
}

#if defined(CRT_CUPQC_CUPQC_AVAILABLE)

// ===========================================================================
// 3. cuPQC device kernels.
// ===========================================================================

// One-time Barrett setup.  A single warp calls the static __device__ helper;
// `approx` and `den_clz` are opaque state consumed by every later reduction.
__global__ void cupqc_barrett_setup_kernel(const u32* d_q, u32* d_approx,
                                           int* d_clz) {
    BI q(d_q, 0);          // all 32 lanes, same pointer/index
    BI approx;             // default zero
    int clz = 0;
    BI::setup_barrett(q, approx, clz);
    approx.store(d_approx, 0);
    if ((threadIdx.x & 31u) == 0u) d_clz[0] = clz;
}

// One warp (= TPI<32> lanes) per coefficient.  Scratch: 8 warps * 2*num_limbs
// u32; each warp uses `s = smem + warp*2*num_limbs`, with low half = single
// accumulator/remainder and high half = 0 (so bigint_wide.hi == 0 < q).
__global__ void cupqc_crt_kernel(CrtArgs ca, const u32* residues,
                                 const u32* d_weights, const u32* d_q,
                                 const u32* d_approx, const int* d_clz,
                                 u32* out) {
    constexpr unsigned LIMBS = unsigned(Desc::num_limbs);
    extern __shared__ u32 smem[];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned coef = blockIdx.x * CRT_CUPQC_WARPS + warp;
    const unsigned pair = blockIdx.z;
    u32* const s = smem + warp * 2u * LIMBS;   // identical address in the warp

    // Zero the high half of this warp's bigint_wide scratch (limbs LIMBS..2L-1).
    // LIMBS is a multiple of 32; each lane zero-owns LIMBS/32 of those limbs.
    for (unsigned t = 0; t < LIMBS; t += 32u) s[LIMBS + t + lane] = 0u;

    const unsigned n = ca.n, l = ca.limbs, k = ca.count;
    if (coef >= n) return;                      // whole warp exits together

    // ---- Step 1+2: Shoup weights, exact weighted sum, FP64 correction. ----
    const u64 rbase = u64(pair) * k * n;
    double fsum = 0.0;
    BI acc;                                     // zero-initialized by cuPQC
    for (unsigned j = 0; j < k; ++j) {
        const u32 r = residues[rbase + u64(j) * n + coef];  // warp-uniform load
        const PrimeInfo& pi = ca.primes[j];
        const u32 term = weight_residue(r, pi);              // identical in all lanes
        BI w(d_weights, j);                                  // W[j], batched load
        BI prod = w.mul_scalar(term);                        // fixed-width, exact
        acc = acc + prod;          // operator+ (NOT +=: += bigint is thread-only)
        fsum += double(term) * pi.reciprocal;
    }
    const u32 correction = u32(fsum + 0.5);
    BI wcorr(d_weights, k);                                  // column K = -P mod q
    BI cprod = wcorr.mul_scalar(correction);
    acc = acc + cprod;

    // ---- Step 3: reduce X modulo q. ----
#if CRT_CUPQC_REDUCE_MODE == 0
    {   // Barrett (default).  setup_barrett state was built in the Plan ctor.
        BI q(d_q, 0);
        BI approx(d_approx, 0);
        const int clz = *d_clz;                  // uniform load
        acc.store(s, 0);                         // low half = X; 64 limbs
        __syncwarp();
        WIDE wide(s);                            // hi = 0, lo = X  => hi < q
        BI rem;
        BI::reduce_barrett(rem, wide, q, approx, clz);
        rem.store(s, 0);                         // canonical [0,q)
    }
#elif CRT_CUPQC_REDUCE_MODE == 1
    {   // Direct remainder (warp-capable `mod`; no setup state needed).
        BI q(d_q, 0);
        BI rem = acc.mod(q);
        rem.store(s, 0);
    }
#else
    {   // Reference reduction on the extracted limbs (no cuPQC reduction).
        acc.store(s, 0);
        // s[0..LIMBS-1] holds X; reduce_sum needs L+2 <= LIMBS limbs.
    }
#endif
    __syncwarp();

    // ---- Step 4: write the L canonical limbs to out[pair][coef][limb]. ----
    if (lane == 0) {
#if CRT_CUPQC_REDUCE_MODE == 2
        reduce_sum<1>(s, l, ca.bits, ca.gap, ca.q);
#endif
        const u64 dst = u64(pair) * n * l + u64(coef) * l;
        for (unsigned i = 0; i < l; ++i) out[dst + i] = s[i];
    }
}

#endif  // CRT_CUPQC_CUPQC_AVAILABLE
#endif  // __CUDACC__

// ===========================================================================
// 4. Host-side Plan: uploads tables, builds Barrett state, launches kernels.
// ===========================================================================
struct Plan {
    unsigned L = 0, pitch = 0, K = 0;
    unsigned limbs = 0;                 // cuPQC instance limb count (64)
    u32* d_weights_row = nullptr;       // [pitch][limbs], for BI(ptr,j) loads
    u32* d_weights_col = nullptr;       // [L][pitch], for run_simple()
    u32* d_q_pad = nullptr;             // [limbs], zero-padded q
    u32* d_approx = nullptr;            // [limbs], Barrett reciprocal state
    int* d_clz = nullptr;               // [1], Barrett leading-zero count
    bool ok = false;                    // reference (run_simple) tables ready
    bool cupqc_ok = false;              // cuPQC tables + Barrett state ready

    Plan() = default;

    Plan(const std::vector<u32>& weights_host, unsigned L_, unsigned pitch_,
         const std::vector<u32>& q_limbs, unsigned K_)
        : L(L_), pitch(pitch_), K(K_) {
#ifdef __CUDACC__
        auto dev_u32 = [](const std::vector<u32>& v) -> u32* {
            u32* p = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&p), v.size() * sizeof(u32)) != cudaSuccess)
                throw std::runtime_error("crt_cupqc: cudaMalloc failed");
            if (cudaMemcpy(p, v.data(), v.size() * sizeof(u32),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(p);
                throw std::runtime_error("crt_cupqc: upload failed");
            }
            return p;
        };
#endif
#if defined(__CUDACC__) && defined(CRT_CUPQC_CUPQC_AVAILABLE)
        limbs = unsigned(Desc::num_limbs);
        if (L == 0 || L > CRT_CUPQC_MAXL || pitch == 0 ||
            pitch > CRT_CUPQC_MAXPITCH || K + 1 != pitch ||
            weights_host.size() != std::size_t(L) * pitch ||
            q_limbs.empty() || q_limbs.size() > L)
            throw std::runtime_error("crt_cupqc: bad Plan dimensions");
        // The reference only needs L limbs of q; reject an all-zero modulus.
        bool nonzero = false;
        for (u32 w : q_limbs) nonzero |= (w != 0u);
        if (!nonzero) throw std::runtime_error("crt_cupqc: zero modulus q");

        // weights_host is gpu31's [limb][column] layout: weights[i*pitch+j].
        // cuPQC needs [column][limb] so BI(d_weights_row, j) loads column j.
        std::vector<u32> row(std::size_t(pitch) * limbs, 0u);
        for (unsigned j = 0; j < pitch; ++j)
            for (unsigned i = 0; i < L; ++i)
                row[std::size_t(j) * limbs + i] = weights_host[std::size_t(i) * pitch + j];
        // q padded to the full instance width (upper limbs zero).
        std::vector<u32> qpad(std::size_t(limbs), 0u);
        for (unsigned i = 0; i < q_limbs.size(); ++i) qpad[i] = q_limbs[i];
        std::vector<u32> approx0(std::size_t(limbs), 0u);

        try {
            d_weights_row = dev_u32(row);
            d_weights_col = dev_u32(weights_host);
            d_q_pad       = dev_u32(qpad);
            d_approx      = dev_u32(approx0);
            if (cudaMalloc(reinterpret_cast<void**>(&d_clz), sizeof(int)) != cudaSuccess)
                throw std::runtime_error("crt_cupqc: cudaMalloc(clz) failed");
            int zero = 0;
            if (cudaMemcpy(d_clz, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("crt_cupqc: clz upload failed");

            // setup_barrett is __device__-only: build the opaque state once.
            cupqc_barrett_setup_kernel<<<1, 32, 0, 0>>>(d_q_pad, d_approx, d_clz);
            cudaError_t st = cudaGetLastError();
            if (st != cudaSuccess)
                throw std::runtime_error("crt_cupqc: Barrett setup launch failed");
            st = cudaStreamSynchronize(0);   // state must be ready on any stream
            if (st != cudaSuccess)
                throw std::runtime_error("crt_cupqc: Barrett setup sync failed");
            ok = cupqc_ok = true;
        } catch (...) {
            release();
            throw;
        }
#elif defined(__CUDACC__)
        // No cuPQC header found (or non-CUDA image): still provide the
        // byte-identical CUDA-core reference path for A/B testing.
        if (L == 0 || L > CRT_CUPQC_MAXL || pitch == 0 ||
            pitch > CRT_CUPQC_MAXPITCH || K + 1 != pitch ||
            weights_host.size() != std::size_t(L) * pitch)
            throw std::runtime_error("crt_cupqc: bad Plan dimensions");
        d_weights_col = dev_u32(weights_host);
        limbs = 0;
        ok = true;
        (void)q_limbs;
#else
        (void)weights_host; (void)q_limbs;
        throw std::runtime_error("crt_cupqc: constructed without CUDA");
#endif
    }

    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;

    ~Plan() { release(); }

    void release() {
#ifdef __CUDACC__
        if (d_weights_row) cudaFree(d_weights_row);
        if (d_weights_col) cudaFree(d_weights_col);
        if (d_q_pad) cudaFree(d_q_pad);
        if (d_approx) cudaFree(d_approx);
        if (d_clz) cudaFree(d_clz);
#endif
        d_weights_row = d_weights_col = d_q_pad = d_approx = nullptr;
        d_clz = nullptr;
        ok = cupqc_ok = false;
    }

    // Build a local, layout-identical CrtArgs from any gpu31::CrtArgs-like
    // struct (duck typing avoids an include-order dependency).
    template <class Args>
    CrtArgs make_args(const Args& args) const {
        CrtArgs ca{};
        ca.n = args.n;
        ca.limbs = args.limbs;
        ca.bits = args.bits;
        ca.count = args.count;
        ca.gap = args.gap;
        ca.primes = reinterpret_cast<const PrimeInfo*>(args.primes);
        ca.q = args.q;
        ca.weights = d_weights_col;   // used only by run_simple()
        return ca;
    }

    template <class Args>
    bool check(const Args& args, u32 B, const u32* residues, const u32* out) const {
        return ok && B > 0 && residues && out && args.primes && args.q &&
               args.limbs > 0 && args.limbs <= L &&
               args.count > 0 && args.count + 1 == pitch;
    }

    // ---- PRIMARY: cuPQC BigInt weighted sum + exact reduction mod q. ------
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
#if defined(__CUDACC__) && defined(CRT_CUPQC_CUPQC_AVAILABLE)
        if (!cupqc_ok || !check(args, B, residues, out)) return cudaErrorInvalidValue;
        if (L > limbs) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem =
            std::size_t(CRT_CUPQC_WARPS) * 2u * limbs * sizeof(u32);
        cupqc_crt_kernel<<<dim3((ca.n + CRT_CUPQC_WARPS - 1) / CRT_CUPQC_WARPS,
                                1, B),
                          CRT_CUPQC_THREADS, smem, stream>>>(
            ca, residues, d_weights_row, d_q_pad, d_approx, d_clz, out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return kInvalidValue;
#endif
    }

    // ---- EXACT CUDA-core reference (A/B / debugging, no cuPQC needed). ---
    template <class Args>
    cudaError_t run_simple(const u32* residues, u32* out, const Args& args, u32 B,
                           cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_simple_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, d_weights_col, out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return kInvalidValue;
#endif
    }
};

}  // namespace crt_cupqc

#undef CRT_CUPQC_FHD
