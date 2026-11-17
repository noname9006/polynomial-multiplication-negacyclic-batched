// encode_mma.cuh — tensor-core (MMA) replacement for gpu31::encode_tiled.
//
// Header-only. This file does NOT modify or include any existing project file
// except <cuda_runtime.h> and "fherma.h". Integrate by including it after
// gpu31.cuh in an nvcc build:
//
//     #include "gpu31.cuh"
//     #include "encode_mma.cuh"
//     ...
//     encode_mma::EncodeMmaPlan enc(dev);      // or enc(dev.plan())
//     enc.run(d_a, d_b, dev.d_ra, dev.d_rb, stream, B, N, L, K);
//
// The constructor is duck-typed: anything exposing N, L, K, primes[] (with
// .p/.neg_inv), weights[] (with .value) and reversed_twist[] works, as does a
// gpu31::Device (its .plan() is used).  Exactly the gpu31 buffers can therefore
// be passed to run().
//
// ---------------------------------------------------------------------------
// MATH — why this is byte-for-byte identical to encode_tiled
// ---------------------------------------------------------------------------
// gpu31 computes, for every (coefficient i, prime k)
//
//     t_l     = lazy_product(a_l, W_l, p) = a_l*W_l - hi(a_l*Q_l)*p   (mod 2^32)
//     sum     = Σ_l t_l                                     (u64, < 55*2p)
//     r       = finish(sum, w_0, p) = sum mod p, reduced to [0,p)
//     out     = mont(r, reversed_twist[k][i], p)
//
// with W_l = 2^(32(l+1)) mod p, Q_l = floor(W_l * 2^32 / p), and w_0 = W_0.
// Because t_l ≡ a_l * W_l (mod p) and 0 <= t_l < 2p, we have
//
//     r = (Σ_l a_l * W_l) mod p =: P mod p.
//
// So the per-limb Shoup reduction (which is what makes the CUDA-core version
// compute-bound) is NOT needed if P is accumulated exactly and reduced once.
// Split a_l = Σ_c1 A_l[c1]*256^c1 and W_l = Σ_c2 B_l[c2]*256^c2 (4 bytes each):
//
//     P = Σ_{c1,c2} (Σ_l A_l[c1]*B_l[c2]) * 2^(8(c1+c2))
//       = Σ_{s=0..6} 2^(8s) * S_s,   S_s = Σ_{c1+c2=s} Σ_l A_l[c1]*B_l[c2].
//
// Each S_s fits in s32: at most 4 byte pairs * 55 limbs * 255^2 ≈ 14.3e6.
// Each of the 16 inner dot products Σ_l A_l[c1]*B_l[c2] is formed with
// `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32`.  `.u8` is essential:
// bytes 0x80..0xff must be treated as UNSIGNED (crt_mma.cuh documents that raw
// `.s8` is not exact).  With non-negative u8 operands and <=32 terms per MMA a
// single s32 accumulator is exact (max 32*255^2 = 2,080,800 < 2^31).
//
// P mod p is then obtained with the very same Shoup trick on the 7 reduced
// coefficients: for pow_s = 2^(8s) mod p and powq_s = floor(pow_s*2^32/p),
//     term_s = S_s*pow_s - hi(S_s*powq_s)*p  ∈ [0, 2p),
// summed mod p (one or two conditional subtractions; p < 2^31 keeps u32 safe).
// The scalar tail `mont(r, twist, p)` is copied verbatim from gpu31.
//
// The task description suggested ceil(55*4/32)=7 K-blocks per byte pair.  Here
// K is taken over LIMBS instead (ceil(55/32) = 2 blocks of 32, zero-padded to
// 64), and the 4x4 byte positions are 16 separate MMA accumulators — strictly
// fewer MMA instructions and exactly the same result.
//
// ---------------------------------------------------------------------------
// FRAGMENT LAYOUT (PTX ISA 9.5, m16n8k32 .u8), used verbatim below
// ---------------------------------------------------------------------------
// groupID = lane>>2, tid = lane&3.  M = coefficient (0..15), N = prime (0..7),
// K = limb (0..31).  Rows/cols of the three fragments:
//   A a0: (M=g,   K=tid*4+0..3)   a1: (M=g+8, K=tid*4+0..3)
//     a2: (M=g,   K=tid*4+16..19) a3: (M=g+8, K=tid*4+16..19)
//   B b0: (K=tid*4+0..3,   N=g)   b1: (K=tid*4+16..19, N=g)
//   C c0: (M=g,   N=tid*2)   c1: (M=g,   N=tid*2+1)
//     c2: (M=g+8, N=tid*2)   c3: (M=g+8, N=tid*2+1)
// (This is the standard distribution: each (M,K), (K,N) and (M,N) element is
// held by exactly one lane.  The hardware combines the warp-wide fragments.)
//
// ---------------------------------------------------------------------------
// RISKS / OPERATIONAL NOTES
// ---------------------------------------------------------------------------
//  * Inline PTX is used directly; <mma.h> is deliberately NOT included (it
//    breaks the local host shim type-check).  The mma body is compiled only
//    under __CUDA_ARCH__, so a host compiler never sees the PTX asm.
//  * No dynamic shared memory > 48 KiB is used (max ~16 KiB), but
//    cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...) is
//    still called defensively — the omission of exactly that call caused the
//    earlier MMA CRT launch failure documented in crt_mma.cuh.
//  * No large __constant__ symbol is declared: gpu31's c_weights already uses
//    ~54 KiB of the 64 KiB constant bank.  Weights are read from a small (≤30
//    KiB) global table that resides in L1/L2.  This also makes run_simple()
//    self-contained (it uses global rather than constant weights).
//  * The host self-test `encode_mma_selftest()` checks the byte-split dot
//    product and the Shoup fold against plain u64 arithmetic on random inputs.
// ===========================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cstdio>

#include <cuda_runtime.h>          // shim declares the error enum in host builds

#include "fherma.h"

namespace encode_mma {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
#define ENC_FHD __host__ __device__
#else
#define ENC_FHD
#endif

// Compile-time limits (the official point has L=55, K=117, N=65536).
static constexpr unsigned ENC_MAX_L   = 128;
static constexpr unsigned ENC_MAX_K   = 512;
static constexpr unsigned ENC_MAX_N   = 1u << 21;
static constexpr unsigned ENC_WARPS   = 4;      // warps per block
static constexpr unsigned ENC_THREADS = ENC_WARPS * 32;
static constexpr unsigned ENC_M       = 16;     // MMA M: coefficients per warp
static constexpr unsigned ENC_N       = 8;      // MMA N: primes per warp
static constexpr unsigned ENC_SHUFFTS = 7;      // shifts 8*s, s=0..6

// Device argument blocks (layout mirrors gpu31 where relevant).
struct Prime {
    u32 p, neg_inv;
};
struct Pow {
    u32 v, q;                      // 2^(8s) mod p, floor(v*2^32/p)
};
struct Weight {
    u32 value, quotient;           // gpu31::Weight-compatible (Shoup row)
};

// ---- arithmetic helpers ----------------------------------------------------
ENC_FHD inline u32 mulhi(u32 a, u32 b) {
#if defined(__CUDA_ARCH__)
    return __umulhi(a, b);
#else
    return u32((u64(a) * b) >> 32);
#endif
}
ENC_FHD inline u32 mont(u32 a, u32 b, const Prime& pr) {
    const u64 t = u64(a) * b;
    const u32 m = u32(t) * pr.neg_inv;
    const u32 u = u32((t + u64(m) * pr.p) >> 32);
    return u >= pr.p ? u - pr.p : u;
}
ENC_FHD inline u32 lazy_product(u32 limb, const Weight& w, u32 p) {
    return limb * w.value - mulhi(limb, w.quotient) * p;
}
ENC_FHD inline u32 canonical_word(u32 x, u32 p) {
    if (x >= p) x -= p;
    if (x >= p) x -= p;
    if (x >= p) x -= p;
    return x;
}
ENC_FHD inline u32 finish(u64 sum, const Weight& radix, u32 p) {
    const u32 lo = canonical_word(u32(sum), p);
    u32 hi = lazy_product(u32(sum >> 32), radix, p);
    if (hi >= p) hi -= p;
    const u32 s = lo + hi;              // < 2p < 2^32 for p < 2^31
    return s >= p ? s - p : s;
}

// Pack four u8 values (low byte first) into one u32 register.
ENC_FHD inline u32 pack4(u32 b0, u32 b1, u32 b2, u32 b3) {
    return (b0 & 0xffu) | ((b1 & 0xffu) << 8) |
           ((b2 & 0xffu) << 16) | ((b3 & 0xffu) << 24);
}
ENC_FHD inline u32 byte_at(u32 x, unsigned c) {
    return (x >> (8u * c)) & 0xffu;
}

// ---------------------------------------------------------------------------
// 1. tensor-core kernels (nvcc only)
// ---------------------------------------------------------------------------
#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)

__device__ __forceinline__ void mma_u8(u32& d0, u32& d1, u32& d2, u32& d3,
                                       u32 a0, u32 a1, u32 a2, u32 a3,
                                       u32 b0, u32 b1) {
#if defined(__CUDA_ARCH__)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

// One warp's 16x8x(Kblocks*32) byte-split dot product.  `coefs` is a pointer to
// the 16*lpad u32 staging array for one operand (a or b).  Results are ADDED
// into ss[7*4]: ss[s*4+o] = Σ_{c1+c2=s} D_o[c1][c2] for the warp's four outputs.
__device__ __forceinline__ void dot_bytes(const u32* __restrict__ coefs,
                                          const u32* __restrict__ wpad,
                                          unsigned lpad, unsigned nblocks,
                                          unsigned g, unsigned t, unsigned nB,
                                          unsigned count, u32* ss) {
    #pragma unroll 1
    for (unsigned kb = 0; kb < nblocks; ++kb) {
        const unsigned k0 = kb * 32 + t * 4;
        const unsigned k1 = k0 + 16;
        // 16 coefficient words: A rows g and g+8, K slices k0..k0+3 and k1..k1+3.
        u32 ca[4], cb[4], ca2[4], cb2[4];
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            ca[j]  = coefs[g * lpad + k0 + j];
            cb[j]  = coefs[(g + 8) * lpad + k0 + j];
            ca2[j] = coefs[g * lpad + k1 + j];
            cb2[j] = coefs[(g + 8) * lpad + k1 + j];
        }
        // 8 weight words: lane owns prime nB = groupID, limb slices k0/k1.
        u32 w[8];
        const bool wb = nB < count;
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            w[j]     = wb ? wpad[nB * lpad + k0 + j] : 0u;
            w[4 + j] = wb ? wpad[nB * lpad + k1 + j] : 0u;
        }
        #pragma unroll
        for (unsigned c1 = 0; c1 < 4; ++c1) {
            const u32 a0 = pack4(byte_at(ca[0], c1), byte_at(ca[1], c1),
                                 byte_at(ca[2], c1), byte_at(ca[3], c1));
            const u32 a1 = pack4(byte_at(cb[0], c1), byte_at(cb[1], c1),
                                 byte_at(cb[2], c1), byte_at(cb[3], c1));
            const u32 a2 = pack4(byte_at(ca2[0], c1), byte_at(ca2[1], c1),
                                 byte_at(ca2[2], c1), byte_at(ca2[3], c1));
            const u32 a3 = pack4(byte_at(cb2[0], c1), byte_at(cb2[1], c1),
                                 byte_at(cb2[2], c1), byte_at(cb2[3], c1));
            #pragma unroll
            for (unsigned c2 = 0; c2 < 4; ++c2) {
                const u32 b0 = pack4(byte_at(w[0], c2), byte_at(w[1], c2),
                                     byte_at(w[2], c2), byte_at(w[3], c2));
                const u32 b1 = pack4(byte_at(w[4], c2), byte_at(w[5], c2),
                                     byte_at(w[6], c2), byte_at(w[7], c2));
                u32 d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                mma_u8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                const unsigned off = (c1 + c2) * 4;
                ss[off + 0] += d0;
                ss[off + 1] += d1;
                ss[off + 2] += d2;
                ss[off + 3] += d3;
            }
        }
    }
}

// Fold the 7 shift-sums for one output into a residue in [0,p).
__device__ __forceinline__ u32 fold_residue(const u32* ss, unsigned o,
                                            const Pow* pt, u32 p) {
    u32 acc = 0;
    #pragma unroll
    for (unsigned s = 0; s < ENC_SHUFFTS; ++s) {
        const u32 v = ss[s * 4 + o];
        const Pow q = pt[s];
        u32 term = v * q.v - mulhi(v, q.q) * p;   // exact value in [0,2p)
        if (term >= p) term -= p;
        acc += term;                              // < 2p < 2^32 (p < 2^31)
        if (acc >= p) acc -= p;
    }
    return acc;
}

__global__ void encode_mma_kernel(const u32* ar, const u32* br,
                                  u32* ra, u32* rb,
                                  const Prime* primes, const u32* wpad,
                                  const Pow* powtab, const u32* twist,
                                  unsigned n, unsigned l, unsigned count,
                                  unsigned lpad, unsigned nblocks) {
    extern __shared__ u32 shared[];
    u32* sca = shared;                       // 16 * lpad, operand a
    u32* scb = sca + ENC_M * lpad;            // 16 * lpad, operand b

    const unsigned start = blockIdx.x * ENC_M;
    const unsigned pg = blockIdx.y * ENC_WARPS + (threadIdx.x >> 5);
    const unsigned pair = blockIdx.z;
    const unsigned lane = threadIdx.x & 31;
    const unsigned g = lane >> 2;            // MMA row group / B prime group
    const unsigned t = lane & 3;             // MMA K sub-index

    // Stage L limbs of the 16 coefficients of this tile (zero fill out of range).
    const u64 abase = u64(pair) * n * l;
    for (unsigned idx = threadIdx.x; idx < ENC_M * lpad; idx += ENC_THREADS) {
        const unsigned row = idx / lpad;
        const unsigned limb = idx - row * lpad;
        const unsigned coef = start + row;
        const u32 va = (row < ENC_M && coef < n && limb < l)
                     ? ar[abase + u64(coef) * l + limb] : 0u;
        const u32 vb = (row < ENC_M && coef < n && limb < l)
                     ? br[abase + u64(coef) * l + limb] : 0u;
        sca[idx] = va;
        scb[idx] = vb;
    }
    __syncthreads();

    // ---- operand a -------------------------------------------------------
    u32 ss[ENC_SHUFFTS * 4] = {};
    dot_bytes(sca, wpad, lpad, nblocks, g, t, pg * ENC_N + g, count, ss);

    const u64 rbase = u64(pair) * count * n;
    const unsigned p0 = pg * ENC_N + t * 2;
    const unsigned p1 = p0 + 1;
    #pragma unroll
    for (unsigned o = 0; o < 4; ++o) {
        const unsigned prime = (o & 1u) ? p1 : p0;
        const unsigned coef = start + ((o < 2u) ? g : g + 8);
        if (prime < count && coef < n) {
            const Prime pr = primes[prime];
            const u32 r = fold_residue(ss, o, powtab + u64(prime) * ENC_SHUFFTS, pr.p);
            const u32 tw = twist[u64(prime) * n + coef];
            ra[rbase + u64(prime) * n + coef] = mont(r, tw, pr);
        }
    }

    // ---- operand b (same weights, reuse the staging) ----------------------
    #pragma unroll
    for (unsigned s = 0; s < ENC_SHUFFTS * 4; ++s) ss[s] = 0;
    dot_bytes(scb, wpad, lpad, nblocks, g, t, pg * ENC_N + g, count, ss);
    #pragma unroll
    for (unsigned o = 0; o < 4; ++o) {
        const unsigned prime = (o & 1u) ? p1 : p0;
        const unsigned coef = start + ((o < 2u) ? g : g + 8);
        if (prime < count && coef < n) {
            const Prime pr = primes[prime];
            const u32 r = fold_residue(ss, o, powtab + u64(prime) * ENC_SHUFFTS, pr.p);
            const u32 tw = twist[u64(prime) * n + coef];
            rb[rbase + u64(prime) * n + coef] = mont(r, tw, pr);
        }
    }
}

// ---- CUDA-core fallback: exact copy of encode_tiled's math ---------------
// Odd shared stride 57 (= 56+1) avoids bank conflicts; weights live in a small
// global table with stride `lpad` (constant memory is already occupied by
// gpu31::c_weights in the intended integration, see header notes).
__global__ void encode_simple_kernel(const u32* ar, const u32* br,
                                     u32* ra, u32* rb,
                                     const Prime* primes, const Weight* weights,
                                     const Weight* radix, const u32* twist,
                                     unsigned n, unsigned l, unsigned count,
                                     unsigned lpad) {
    extern __shared__ u32 stage[];
    const unsigned stride = 57;               // odd, conflict-free
    u32* sa = stage;
    u32* sb = stage + 32 * stride;
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const unsigned start = blockIdx.x * 32;
    const unsigned valid = (n - start < 32) ? (n - start) : 32;
    const u64 base = u64(blockIdx.z) * n * l;
    const u64 rbase = u64(blockIdx.z) * count * n;
    for (unsigned tt = y * 32 + x; tt < valid * l; tt += 32 * 8) {
        const unsigned row = tt / l, col = tt - row * l;
        const u64 off = base + u64(start) * l + tt;
        sa[row * stride + col] = ar[off];
        sb[row * stride + col] = br[off];
    }
    __syncthreads();
    const unsigned m = blockIdx.y * 8 + y;
    if (m >= count || x >= valid) return;
    const u32 p = primes[m].p;
    const Weight* wrow = weights + u64(m) * lpad;
    u64 sa_sum = 0, sb_sum = 0;
    #pragma unroll 4
    for (unsigned j = 0; j < l; ++j) {
        const Weight w = wrow[j];
        sa_sum += lazy_product(sa[x * stride + j], w, p);
        sb_sum += lazy_product(sb[x * stride + j], w, p);
    }
    const u32 a = finish(sa_sum, radix[m], p);
    const u32 b = finish(sb_sum, radix[m], p);
    const u64 off = rbase + u64(m) * n + start + x;
    const u32 tw = twist[u64(m) * n + start + x];
    ra[off] = mont(a, tw, primes[m]);
    rb[off] = mont(b, tw, primes[m]);
}

#endif  // __CUDACC__ && !PGPU_HOST_CHECK

// ===========================================================================
// 2. Host-side plan: builds tables once, owns device copies, launches.
// ===========================================================================
struct EncodeMmaPlan {
    unsigned N = 0, L = 0, K = 0, lpad = 0, nblocks = 0;

    Prime* d_primes = nullptr;
    u32*   d_wpad = nullptr;       // K*lpad, W_l (zero padded)
    Weight* d_weights = nullptr;   // K*lpad, (value,quotient) for run_simple
    Weight* d_radix = nullptr;     // K, W_0
    Pow*   d_pow = nullptr;        // K*7
    u32*   d_twist = nullptr;      // K*N

    EncodeMmaPlan() = default;

    // Accept either a gpu31::Plan-like object (N,L,K,primes,weights,
    // reversed_twist) or a wrapper exposing .plan() (e.g. gpu31::Device).
    template <class T>
    explicit EncodeMmaPlan(const T& src) { init(plan_of(src, 0)); }

    EncodeMmaPlan(const EncodeMmaPlan&) = delete;
    EncodeMmaPlan& operator=(const EncodeMmaPlan&) = delete;
    ~EncodeMmaPlan() { release(); }

    // gpu31-style signature (N/L/K are validated against the plan).
    cudaError_t run(const u32* ar, const u32* br, u32* ra, u32* rb,
                    cudaStream_t stream, u32 B, u32 n_, u32 l_, u32 k_) {
        if (n_ != N || l_ != L || k_ != K) return invalid();
        return launch(false, ar, br, ra, rb, stream, B);
    }
    cudaError_t run_simple(const u32* ar, const u32* br, u32* ra, u32* rb,
                           cudaStream_t stream, u32 B, u32 n_, u32 l_, u32 k_) {
        if (n_ != N || l_ != L || k_ != K) return invalid();
        return launch(true, ar, br, ra, rb, stream, B);
    }
    // Shorthand (the plan already knows N/L/K).
    template <class T>
    cudaError_t run(const u32* ar, const u32* br, u32* ra, u32* rb,
                    cudaStream_t stream, T batches) {
        return launch(false, ar, br, ra, rb, stream, u32(batches));
    }
    template <class T>
    cudaError_t run_simple(const u32* ar, const u32* br, u32* ra, u32* rb,
                           cudaStream_t stream, T batches) {
        return launch(true, ar, br, ra, rb, stream, u32(batches));
    }

private:
    static cudaError_t invalid() {
#if defined(PGPU_HOST_CHECK)
        return static_cast<cudaError_t>(1);
#else
        return cudaErrorInvalidValue;
#endif
    }
    template <class T>
    static auto const& plan_of(const T& x, int) -> decltype(x.plan()) {
        return x.plan();
    }
    template <class T>
    static T const& plan_of(const T& x, long) { return x; }

    template <class P>
    void init(const P& src) {
        N = src.N; L = src.L; K = src.K;
        if (N == 0 || N > ENC_MAX_N || L == 0 || L > ENC_MAX_L ||
            K == 0 || K > ENC_MAX_K)
            throw std::runtime_error("encode_mma: N/L/K outside compiled limits");
        if (src.primes.size() != K || src.weights.size() != std::size_t(K) * L ||
            src.reversed_twist.size() != std::size_t(K) * N)
            throw std::runtime_error("encode_mma: source tables have unexpected sizes");
        lpad = ((L + 31) / 32) * 32;
        nblocks = lpad / 32;

        std::vector<Prime> hp(K);
        std::vector<Weight> hw(std::size_t(K) * lpad), hr(K);
        std::vector<u32> hwpad(std::size_t(K) * lpad);
        std::vector<Pow> hpow(std::size_t(K) * ENC_SHUFFTS);
        for (unsigned k = 0; k < K; ++k) {
            const u32 p = src.primes[k].p;
            if (p < (1u << 30) || p >= (1u << 31))
                throw std::runtime_error("encode_mma: primes must be in [2^30,2^31)");
            hp[k] = Prime{src.primes[k].p, src.primes[k].neg_inv};
            for (unsigned j = 0; j < L; ++j) {
                const Weight w = {src.weights[std::size_t(k) * L + j].value,
                                  src.weights[std::size_t(k) * L + j].quotient};
                hw[std::size_t(k) * lpad + j] = w;
                hwpad[std::size_t(k) * lpad + j] = w.value;
            }
            hr[k] = hw[std::size_t(k) * lpad];        // W_0 = 2^32 mod p
            u32 pw = 1;
            for (unsigned s = 0; s < ENC_SHUFFTS; ++s) {
                hpow[std::size_t(k) * ENC_SHUFFTS + s] =
                    Pow{pw, u32((u64(pw) << 32) / p)};
                pw = u32((u64(pw) * 256u) % p);       // next 2^(8(s+1)) mod p
            }
        }

#if defined(__CUDACC__)
        try {
            upload<Prime>(d_primes, hp);
            upload<u32>(d_wpad, hwpad);
            upload<Weight>(d_weights, hw);
            upload<Weight>(d_radix, hr);
            upload<Pow>(d_pow, hpow);
            upload<u32>(d_twist, src.reversed_twist);
#if !defined(PGPU_HOST_CHECK)
            // Defensive opt-in (not strictly needed: smem <= 16 KiB here).
            // Omitting this is exactly what broke the earlier MMA CRT kernel.
            (void)cudaFuncSetAttribute(
                encode_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                int(ENC_THREADS * ENC_MAX_L / 32u * ENC_M * 2u * sizeof(u32)));
#endif
        } catch (...) {
            release();
            throw;
        }
#else
        (void)hp; (void)hw; (void)hwpad; (void)hr; (void)hpow;
        throw std::runtime_error("encode_mma: constructed without CUDA");
#endif
    }

    cudaError_t launch(bool simple, const u32* ar, const u32* br,
                       u32* ra, u32* rb, cudaStream_t stream, u32 B) {
        if (B == 0) return cudaSuccess;
        if (!ar || !br || !ra || !rb || !d_primes || !d_wpad || !d_pow || !d_twist)
            return invalid();
#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
        if (simple) {
            if (!d_weights || !d_radix) return cudaErrorInvalidValue;
            const std::size_t smem = 2u * 32u * 57u * sizeof(u32);
            (void)cudaFuncSetAttribute(
                encode_simple_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                int(smem));
            encode_simple_kernel<<<dim3((N + 31) / 32, (K + 7) / 8, B),
                                    dim3(32, 8), smem, stream>>>(
                ar, br, ra, rb, d_primes, d_weights, d_radix, d_twist,
                N, L, K, lpad);
        } else {
            const std::size_t smem = 2u * ENC_M * lpad * sizeof(u32);
            encode_mma_kernel<<<dim3((N + ENC_M - 1) / ENC_M,
                                     (K + ENC_WARPS * ENC_N - 1) /
                                         (ENC_WARPS * ENC_N), B),
                                ENC_THREADS, smem, stream>>>(
                ar, br, ra, rb, d_primes, d_wpad, d_pow, d_twist,
                N, L, K, lpad, nblocks);
        }
        return cudaGetLastError();
#else
        (void)simple; (void)ar; (void)br; (void)ra; (void)rb; (void)stream;
        return cudaSuccess;
#endif
    }

    static void* raw_alloc(std::size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess)
            throw std::runtime_error("encode_mma: cudaMalloc failed");
        return p;
    }
    template <class T>
    static void upload(T*& dst, const std::vector<T>& src) {
        const std::size_t bytes = src.size() * sizeof(T);
        dst = static_cast<T*>(raw_alloc(bytes));
        if (cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(dst);
            dst = nullptr;
            throw std::runtime_error("encode_mma: table upload failed");
        }
    }
    void release() noexcept {
#if defined(__CUDACC__)
        if (d_primes) cudaFree(d_primes);
        if (d_wpad) cudaFree(d_wpad);
        if (d_weights) cudaFree(d_weights);
        if (d_radix) cudaFree(d_radix);
        if (d_pow) cudaFree(d_pow);
        if (d_twist) cudaFree(d_twist);
#endif
        d_primes = nullptr; d_wpad = nullptr; d_weights = nullptr;
        d_radix = nullptr; d_pow = nullptr; d_twist = nullptr;
    }
};

// ===========================================================================
// 3. Host self-test of the byte-split dot product / Shoup fold (no CUDA).
// ===========================================================================
inline bool encode_mma_selftest(unsigned iterations = 4096,
                                u64 seed = 0x9e3779b97f4a7c15ull) {
    auto next = [&seed]() -> u64 {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 0x2545F4914F6CDD1Dull;
    };
    for (unsigned it = 0; it < iterations; ++it) {
        // Any odd p in [2^30,2^31) works — primality is irrelevant to Shoup.
        u32 p = u32(next()) & 0x7fffffffu;
        if (p < (1u << 30)) p += (1u << 30);
        p |= 1u;
        const unsigned L = 1 + u32(next()) % 55u;

        u32 ref = 0;                       // Σ_l a_l*W_l mod p, plain u64
        u32 ref2 = 0;                      // gpu31 lazy_product + finish chain
        u32 S[ENC_SHUFFTS] = {};           // Σ_{c1+c2=s} Σ_l byte*byte
        u64 sum_t = 0;
        Weight radix{0, 0};
        for (unsigned l = 0; l < L; ++l) {
            const u32 a = u32(next());
            const u32 W = u32(next()) % p;
            const u32 Q = u32((u64(W) << 32) / p);
            const Weight w{W, Q};
            if (l == 0) radix = w;
            // exact reference: (u64 product) % p is exact (product < 2^63)
            u32 pm = u32((u64(a) * W) % p);
            ref += pm; if (ref >= p) ref -= p;
            const u32 t = a * W - u32((u64(a) * Q) >> 32) * p;
            sum_t += t;
            for (unsigned c1 = 0; c1 < 4; ++c1)
                for (unsigned c2 = 0; c2 < 4; ++c2)
                    S[c1 + c2] += byte_at(a, c1) * byte_at(W, c2);
        }
        // finish(sum_t) must equal the direct modular sum.
        {
            u32 lo = canonical_word(u32(sum_t), p);
            u32 hi = lazy_product(u32(sum_t >> 32), radix, p);
            if (hi >= p) hi -= p;
            u32 s = lo + hi;
            if (s >= p) s -= p;
            ref2 = s;
        }
        if (ref2 != ref) return false;

        // MMA path: fold the 7 shift sums with Shoup weights.
        u32 got = 0;
        for (unsigned s = 0; s < ENC_SHUFFTS; ++s) {
            u32 pw = 1;
            for (unsigned z = 0; z < s; ++z) pw = u32((u64(pw) * 256u) % p);
            const Pow q{pw, u32((u64(pw) << 32) / p)};
            u32 term = S[s] * q.v - u32((u64(S[s]) * q.q) >> 32) * p;
            if (term >= p) term -= p;
            got += term; if (got >= p) got -= p;
        }
        if (got != ref) return false;
    }
    return true;
}

}  // namespace encode_mma

#undef ENC_FHD
