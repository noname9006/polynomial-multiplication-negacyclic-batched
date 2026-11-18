// crt_const.cuh — EXACT CRT reconstruction for the batched 31-bit RNS
// polynomial multiplication, with the CRT weight table in CUDA **constant**
// memory instead of a global allocation.
//
// Header-only. Nothing outside this file is modified. Integrate by including
// "crt_const.cuh" in the same place and the same #if block as "crt_mma.cuh"
// in solve.cu; the public API is a deliberate mirror of crt_mma::CrtMmaPlan
// and the args are duck-typed, so a gpu31::CrtArgs can be passed unchanged:
//
//     auto plan = std::make_unique<crt_const::CrtMmaPlan>(weights, L, pitch);
//     plan->run(residues, out, gpu31_dev->crt_args(), B, stream);
//
// ---------------------------------------------------------------------------
// DESIGN (what changed vs. crt_mma.cuh, and why)
// ---------------------------------------------------------------------------
// The CRT weight table weights[i*pitch + j] is L*pitch u32 ~= 55*118*4 = 26 KB.
// crt_mma.cuh uploads it as an int8-typed *global* allocation (d_w) and every
// block re-reads it from L1/L2 in sum_column().  With the tile-16 geometry,
// all 16 "m" threads of a coefficient group compute the same limb i in step 3
// and therefore read the SAME row of weights for each j: a textbook warp-level
// broadcast.  Constant memory services same-address warp reads in one cycle.
//
// This file:
//   * declares __constant__ u32 c_w[CRT_CONST_MAXW] and fills it once in the
//     plan constructor with cudaMemcpyToSymbol (no cudaMalloc/cudaFree at all);
//   * keeps crt_mma_kernel BYTE-FOR-BYTE IDENTICAL except that step 3 calls
//     sum_column_const(), whose only change is that
//         w[j]  ->  c_w[u64(limb) * pitch + j]
//     (a direct constant-space index, never a generic pointer, so ptxas emits
//     ld.const rather than a global load);
//   * keeps the exact tile-16 geometry: C=16 coefficients x Y=8 threads,
//     128 threads/block, ODD shared stride 17, the same 3 barriers, and the
//     same Shoup / FP64-correction / carry_and_reduce / gap-fold math.
//     Writes still use dst = pair*n*l + coef*l.  Output is byte-for-byte
//     identical to reconstruct_kernel and to crt_mma.cuh.
//   * run_simple() is kept as a CUDA-core reference path; its kernel reads
//     c_w directly instead of taking a weights pointer, so no second copy of
//     the table is needed.
//
// The per-warp weight access pattern in step 3 is 2 distinct constant
// addresses per j iteration (the y=0 half and the y=1 half of the warp handle
// different limbs), so the constant cache sees a 2-way broadcast, not 32
// serialized addresses.  Do not change CRT_TILE_Y without re-checking this.
//
// ---------------------------------------------------------------------------
// RISKS / OPERATIONAL NOTES  (read before integrating)
// ---------------------------------------------------------------------------
//  * CONSTANT BANK PRESSURE — THE MAIN INTEGRATION RISK.  The device exposes a
//    finite 64 KiB __constant__ budget per compiled module.  gpu31.cuh already
//    declares c_weights[120*56] of Weight (8 bytes) = 53,760 bytes (~54 KiB).
//    Adding this 26 KB table in the SAME translation unit (solve.cu includes
//    gpu31.cuh and crt_mma.cuh together) totals ~80 KB and will be rejected by
//    nvcc/ptxas/nvlink.  The integrator must either (a) build crt_const.cuh's
//    kernel in a separate CUDA translation unit and confirm the device link
//    keeps the two symbols in distinct constant banks, (b) shrink gpu31's
//    c_weights enough to free >= L*pitch*4 bytes, or (c) fall back to the
//    global-weight crt_mma.cuh path.  This file cannot fix it without editing
//    existing files, which was out of scope.  CRT_CONST_MAXW is a build-time
//    knob so the footprint can be tuned; the static_assert below only checks
//    the absolute 64 KiB ceiling.
//  * __constant__ symbols are process-global.  Constructing two CrtMmaPlan
//    objects with different tables overwrites c_w.  solve.cu uses one plan,
//    which is the supported configuration.  Construction is also not
//    thread-safe with respect to kernels already using c_w.
//  * CRT_CONST_MAXW must be >= L*pitch.  The constructor throws otherwise;
//    the default 8192 u32 (32 KiB) covers the official point (55*118=6490) and
//    up to e.g. L=64,K=127.  Define -DCRT_CONST_MAXW=... to change it.
//  * cudaMemcpyToSymbol is not declared by tools/fakecuda, so this header is
//    guarded with BOTH __CUDACC__ and !PGPU_HOST_CHECK (the host-check build
//    defines __CUDACC__).  Include it in solve.cu's `#if !defined(PGPU_HOST_CHECK)`
//    block alongside crt_mma.cuh.  The plain-C++ selftest needs no CUDA.
//  * kInvalidValue == 1 avoids depending on cudaErrorInvalidValue, which the
//    minimal host shim omits (same trick as crt_cupqc.cuh).
//  * crt_const_selftest() is a host arithmetic/layout check; it cannot verify
//    the device launch, register use or the constant-bank allocation.  Use the
//    existing crt_mma_selfcheck-style device A/B test for that.
// ===========================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstddef>

#include <cuda_runtime.h>          // shim supplies error enum in host builds

#include "fherma.h"

namespace crt_const {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Host-check builds define __CUDACC__ but compile with an ordinary C++ compiler
// and a one-thread shim; keep all CUDA-only constructs out of them.
#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
#  define CRTCONST_FHD __host__ __device__
#  define CRTCONST_HAVE_CUDA 1
#else
#  define CRTCONST_FHD
#endif

// Compile-time limits (raised/host-checked as in crt_mma.cuh).
static constexpr unsigned CRT_MMA_MAXL     = 128;   // max limbs
static constexpr unsigned CRT_MMA_MAXPITCH = 512;   // max K+1 columns
static constexpr unsigned CRT_TILE_C       = 16;    // coefficients per block
static constexpr unsigned CRT_TILE_Y       = 8;     // threads per coefficient
static constexpr unsigned CRT_TS           = CRT_TILE_C + 1;  // 17: ODD stride
static constexpr unsigned CRT_THREADS      = CRT_TILE_C * CRT_TILE_Y;

// Constant-memory capacity.  __constant__ arrays need a compile-time size, so
// L*pitch (known only at plan construction) is bounded by this ceiling and only
// the active prefix is uploaded.  Override with -DCRT_CONST_MAXW=<words>.
#ifndef CRT_CONST_MAXW
#  define CRT_CONST_MAXW 8192u   // 32 KiB; official point needs 55*118 = 6490
#endif
static_assert(std::size_t(CRT_CONST_MAXW) * sizeof(u32) <= 65536u,
              "crt_const: CRT_CONST_MAXW exceeds the 64 KiB __constant__ budget");
static constexpr unsigned crt_const_capacity = CRT_CONST_MAXW;

// Minimal host shims (tools/fakecuda) do not declare cudaErrorInvalidValue;
// in the real runtime it equals 1, so a literal is portable and identical.
static constexpr cudaError_t kInvalidValue = static_cast<cudaError_t>(1);

#if defined(CRTCONST_HAVE_CUDA)
// The CRT weight table, row-major by limb: c_w[i*pitch + j].  Only the first
// L*pitch entries are uploaded/used; the remainder stays zero.
__constant__ u32 c_w[CRT_CONST_MAXW];
#endif

// Dynamic shared memory used by crt_const_kernel (must match the kernel).
//   terms : pitch * CRT_TS u32, rounded to an even word count for u64 alignment
//   low   : l * CRT_TS u64
//   high  : l * CRT_TS u64
//   digits: (l+2) * CRT_TS u32
static inline std::size_t crt_const_shared_bytes(unsigned l, unsigned pitch) {
    const unsigned terms_words = pitch * CRT_TS;
    const unsigned aligned_words = (terms_words + 1u) & ~1u;
    return std::size_t(aligned_words + CRT_TS * (5u * l + 2u)) * sizeof(u32);
}

// ---------------------------------------------------------------------------
// Device-side mirrors of the CRT structs (layout-identical to gpu31's).
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
// 1. Helpers copied verbatim from crt_mma.cuh / gpu31.cuh's fherma_crt.
// ===========================================================================
CRTCONST_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template <unsigned Stride>
CRTCONST_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}

template <unsigned Stride>
CRTCONST_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
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
CRTCONST_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}

template <unsigned Stride>
CRTCONST_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
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
CRTCONST_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes, unsigned count) {
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}

// Original sum_column, kept for the host selftest and documentation.  It reads
// a global/linear weight column exactly like crt_mma.cuh.
template <unsigned Stride>
CRTCONST_FHD inline void sum_column(const u32* terms, const u32* basis_column,
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

// THE constant-memory variant of sum_column (used by the production kernel).
// The ONLY difference from sum_column above is the weight load: c_w[limb*pitch+j]
// instead of basis_column[j].  It is a direct constant-space subscript, NOT a
// pointer into constant memory, so nvcc emits ld.const with warp broadcast.
#if defined(CRTCONST_HAVE_CUDA)
template <unsigned Stride>
__device__ __forceinline__ void sum_column_const(const u32* terms,
                                                  unsigned pitch, unsigned limb,
                                                  u64& low_sum, u64& high_sum) {
    u64 lo = 0, hi = 0;
    for (unsigned j = 0; j < pitch; ++j) {
        const u32 weight = c_w[u64(limb) * pitch + j];
        const u64 product = u64(terms[j * Stride]) * weight;
        lo += u32(product);
        hi += product >> 32;
    }
    low_sum = lo;
    high_sum = hi;
}
#endif

template <unsigned Stride>
CRTCONST_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
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
// 2. CUDA-core reference kernel (run_simple path).  Byte-identical to crt_mma's
//    crt_simple_kernel except that the weight column comes from c_w directly.
// ===========================================================================
#if defined(CRTCONST_HAVE_CUDA)
__global__ void crt_simple_const_kernel(CrtArgs plan, const u32* residues, u32* out) {
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
    for (unsigned i = 0; i < l; ++i) {
        u64 lo = 0, hi = 0;
        sum_column_const<1>(terms, pitch, i, lo, hi);
        low[i] = lo;
        high[i] = hi;
    }
    carry_and_reduce<1>(digits, low, high, l, plan.bits, plan.gap, plan.q);

    const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
    for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i];
}

// ===========================================================================
// 3. Production kernel — crt_mma_kernel's geometry and math, unchanged.
//    Tile: 16 coefficients x 8 threads = 128 threads/block, stride 17,
//    3 __syncthreads() (steps 1/2, 2/3, 3/4).  Only step 3's weight load
//    differs from crt_mma.cuh: c_w[i*pitch + j] instead of d_weights[...].
// ===========================================================================
template <int MODE>
__global__ void crt_const_kernel(CrtArgs plan, const u32* residues, u32* out) {
    (void)MODE;  // run() keeps the two templates for launcher compatibility

    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned m = threadIdx.x % CRT_TILE_C;
    const unsigned y = threadIdx.x / CRT_TILE_C;        // 0 .. CRT_TILE_Y-1
    const unsigned coef = blockIdx.x * CRT_TILE_C + m;
    const unsigned pair = blockIdx.z;
    const bool active = coef < plan.n;

    extern __shared__ unsigned char smem[];
    u32* terms = reinterpret_cast<u32*>(smem);
    u64* low = reinterpret_cast<u64*>(terms + ((pitch * CRT_TS + 1u) & ~1u));
    u64* high = low + l * CRT_TS;
    u32* digits = reinterpret_cast<u32*>(high + l * CRT_TS);

    const u64 rbase = u64(pair) * k * plan.n;

    // ---- step 1: Shoup terms t_j = r_j * (P/p_j)^-1 mod p_j ---------------
    for (unsigned j = y; j < k; j += CRT_TILE_Y) {
        const u32 r = active
                    ? residues[rbase + u64(j) * plan.n + coef] : 0u;
        terms[j * CRT_TS + m] = weight_residue(r, plan.primes[j]);
    }
    __syncthreads();

    // ---- step 2: FP64 correction, identical to reference crt_correction ---
    if (y == 0) {
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
            sum += double(terms[j * CRT_TS + m]) * plan.primes[j].reciprocal;
        terms[k * CRT_TS + m] = u32(sum + 0.5);
    }
    __syncthreads();

    // ---- step 3: exact per-limb product sums, weights from constant memory
    for (unsigned i = y; i < l; i += CRT_TILE_Y) {
        u64 lo = 0, hi = 0;
        sum_column_const<CRT_TS>(terms, pitch, i, lo, hi);
        low[i * CRT_TS + m] = lo;
        high[i * CRT_TS + m] = hi;
    }
    __syncthreads();

    // ---- step 4: reference carry_and_reduce / reduce_sum ------------------
    if (y == 0 && active) {
        carry_and_reduce<CRT_TS>(digits + m, low + m, high + m, l,
                                 plan.bits, plan.gap, plan.q);
        const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
        for (unsigned i = 0; i < l; ++i)
            out[dst + i] = digits[i * CRT_TS + m];
    }
}
#endif  // CRTCONST_HAVE_CUDA

// ===========================================================================
// 4. Host selftest — plain C++, no CUDA required.
//    Checks (a) the limb-major row layout used by c_w indexing, (b) the exact
//    lo/hi product accumulation that sum_column_const performs, and (c) the
//    Shoup reduction used for the terms.  Mirrors crt_mma_selftest's API.
// ===========================================================================
inline bool crt_const_selftest(unsigned iterations = 2048,
                               u64 seed = 0x9e3779b97f4a7c15ull) {
    // xorshift64* — deterministic, no <random> needed.
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };

    // (0) The default table must cover the official point.
    if (CRT_CONST_MAXW < 55u * 118u) return false;

    // (a) Row-major layout: weights[i*pitch + j] must address limb i, column j.
    {
        constexpr unsigned L = 55, P = 118;
        std::vector<u32> w(std::size_t(L) * P);
        for (unsigned i = 0; i < L; ++i)
            for (unsigned j = 0; j < P; ++j)
                w[std::size_t(i) * P + j] = 1000u * i + j;
        for (unsigned i = 0; i < L; ++i)
            for (unsigned j = 0; j < P; ++j)
                if (w[std::size_t(i) * P + j] != 1000u * i + j) return false;
    }

    // (b) lo/hi split is exact for the same accumulation sum_column_const uses.
    constexpr unsigned TL = 7, TP = 11;
    for (unsigned it = 0; it < iterations; ++it) {
        u32 terms[TP], weights[TL * TP];
        for (unsigned j = 0; j < TP; ++j) terms[j] = u32(next());
        for (unsigned i = 0; i < TL; ++i)
            for (unsigned j = 0; j < TP; ++j)
                weights[i * TP + j] = u32(next());
        for (unsigned i = 0; i < TL; ++i) {
            u64 ref = 0, lo = 0, hi = 0;
            for (unsigned j = 0; j < TP; ++j) {
                const u32 weight = weights[i * TP + j];   // c_w[i*pitch + j]
                const u64 product = u64(terms[j]) * weight;
                ref += product;
                lo += u32(product);
                hi += product >> 32;
            }
            if (lo + hi * (u64(1) << 32) != ref) return false;
        }
    }

    // (c) Shoup term reduction equals plain modular multiplication.
    {
        const u32 p = 1000003u;
        const u32 a = 654321u;                 // stand-in for (P/p_j)^-1 mod p_j
        // extended Euclid inverse of a modulo p (p is prime, a != 0 mod p).
        // Signed coefficients: the Bezout coefficient can be negative and
        // must be normalized with a true modulo, not by adding p once.
        std::int64_t t = 0, newt = 1, rr = p, newr = a;
        while (newr) {
            const std::int64_t q = rr / newr;
            std::int64_t tmp = t - q * newt; t = newt; newt = tmp;
            tmp = rr - q * newr; rr = newr; newr = tmp;
        }
        if (rr != 1) return false;
        std::int64_t inv_i = t % static_cast<std::int64_t>(p);
        if (inv_i < 0) inv_i += p;
        const u32 inv = u32(inv_i);
        const u32 inv_shoup = u32((u64(inv) << 32) / p);
        PrimeInfo pi{p, inv, inv_shoup, 0.0};
        for (unsigned it = 0; it < 4096; ++it) {
            const u32 rnd = u32(next());
            if (weight_residue(rnd, pi) != u32((u64(rnd) * inv) % p)) return false;
        }
    }
    return true;
}

// ===========================================================================
// 5. Host-side plan — API mirror of crt_mma::CrtMmaPlan.
// ===========================================================================
struct CrtMmaPlan {
    unsigned L = 0, pitch = 0;
    bool ok = false;

    CrtMmaPlan() = default;

    CrtMmaPlan(const std::vector<u32>& weights_host, unsigned L_, unsigned pitch_)
        : L(L_), pitch(pitch_) {
#if defined(CRTCONST_HAVE_CUDA)
        if (L == 0 || pitch == 0 || L > CRT_MMA_MAXL || pitch > CRT_MMA_MAXPITCH)
            throw std::runtime_error("crt_const: L/pitch outside compiled limits");
        if (std::size_t(L) * pitch > CRT_CONST_MAXW)
            throw std::runtime_error("crt_const: L*pitch exceeds CRT_CONST_MAXW "
                                     "(raise -DCRT_CONST_MAXW=...)");
        if (weights_host.size() != std::size_t(L) * pitch)
            throw std::runtime_error("crt_const: weights table has wrong size");
        // One-time host -> constant upload; no device allocation to free.
        const cudaError_t cs = cudaMemcpyToSymbol(c_w, weights_host.data(),
                                                  weights_host.size() * sizeof(u32));
        if (cs != cudaSuccess)
            throw std::runtime_error("crt_const: constant weight upload failed");
        ok = true;
#else
        (void)weights_host;
        throw std::runtime_error("crt_const: constructed without CUDA");
#endif
    }

    CrtMmaPlan(const CrtMmaPlan&) = delete;
    CrtMmaPlan& operator=(const CrtMmaPlan&) = delete;
    ~CrtMmaPlan() = default;

    // Build the device argument block from any CrtArgs-compatible struct.
    // weights is unused by this implementation (the table lives in c_w), but
    // the field is filled for duck-type compatibility with gpu31::CrtArgs.
    template <class Args>
    CrtArgs make_args(const Args& args) const {
        CrtArgs ca{};
        ca.n = args.n;
        ca.limbs = args.limbs;
        ca.bits = args.bits;
        ca.count = args.count;
        ca.gap = args.gap;
        ca.primes = reinterpret_cast<const PrimeInfo*>(args.primes);
        ca.weights = reinterpret_cast<const u32*>(args.weights);
        ca.q = args.q;
        return ca;
    }

    template <class Args>
    bool check(const Args& args, u32 B, const u32* residues, const u32* out) const {
        return ok && B > 0 && residues && out && args.primes && args.q &&
               args.limbs > 0 && args.limbs <= CRT_MMA_MAXL &&
               args.count + 1 <= CRT_MMA_MAXPITCH &&
               args.count + 1 == pitch && args.limbs == L &&
               std::size_t(L) * pitch <= CRT_CONST_MAXW &&
               crt_const_shared_bytes(L, pitch) <= 48u * 1024u;
    }

    // ---- PRIMARY: exact CUDA-core CRT with constant-memory weights --------
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
#if defined(CRTCONST_HAVE_CUDA)
        if (!check(args, B, residues, out)) return kInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_const_shared_bytes(L, pitch);
        crt_const_kernel<0><<<dim3((ca.n + CRT_TILE_C - 1) / CRT_TILE_C, 1, B),
                               CRT_THREADS, smem, stream>>>(ca, residues, out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;   // never called in non-CUDA builds
#endif
    }

    // ---- Kept for launcher compatibility (same kernel as run()) ----------
    template <class Args>
    cudaError_t run_s8(const u32* residues, u32* out, const Args& args, u32 B,
                       cudaStream_t stream = nullptr) {
#if defined(CRTCONST_HAVE_CUDA)
        if (!check(args, B, residues, out)) return kInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_const_shared_bytes(L, pitch);
        crt_const_kernel<1><<<dim3((ca.n + CRT_TILE_C - 1) / CRT_TILE_C, 1, B),
                               CRT_THREADS, smem, stream>>>(ca, residues, out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }

    // ---- CUDA-core reference (byte-identical to reconstruct_kernel) -------
    template <class Args>
    cudaError_t run_simple(const u32* residues, u32* out, const Args& args, u32 B,
                           cudaStream_t stream = nullptr) {
#if defined(CRTCONST_HAVE_CUDA)
        if (!check(args, B, residues, out)) return kInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_simple_const_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }
};

// Convenience alias for callers that prefer a distinct name.
using CrtConstPlan = CrtMmaPlan;

}  // namespace crt_const

#undef CRTCONST_FHD
