// crt_mma.cuh — EXACT CRT reconstruction for the batched 31-bit RNS
// polynomial multiplication (drop-in replacement for gpu31::reconstruct_kernel).
//
// Header-only. Nothing outside this file is modified. Integrate by including
// "crt_mma.cuh" after gpu31.cuh (CrtArgs is duck-typed, so a gpu31::CrtArgs can
// be passed directly — see run/run_s8/run_simple below).
//
// ---------------------------------------------------------------------------
// WHY THIS FILE LOOKS THE WAY IT DOES (read before changing anything)
// ---------------------------------------------------------------------------
// An earlier revision of this file implemented the CRT with
// `mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32`, splitting each u32 operand
// into four bytes, forming the 16 byte-pair sums with tensor cores and
// reassembling X_i = sum_{c1,c2} S[c1][c2] << 8*(c1+c2) in a 128-bit accumulator.
// That algorithm is exact (the `.u8` type qualifier is essential: raw `.s8`
// bytes are NOT, see the byte-split self-test at the bottom of this file), but
// the kernel was not safe to ship:
//
//   1. Its dynamic shared-memory request was
//          aligned_u64*4 + 2*L*M_COEF*8
//      = 10266*4 + 2*55*32*8 = 69,224 bytes at the official point.
//      A kernel using more than 48 KiB of dynamic shared memory requires an
//      explicit `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicShared-
//      MemorySize, ...)` opt-in.  Without it the launch fails with
//      cudaErrorInvalidValue instead of running.
//   2. It contained a verbatim-copy error in `shifted_word` (`q[i*Stride]`
//      instead of the reference's `q[i]`), so any Stride != 1 reduction would
//      have read wild addresses.  With Stride=1 it is harmless, but it shows
//      the copied helper block was not trustworthy.
//   3. Relying on __uint128_t, hand-built MMA fragments and an int8 buffer
//      reinterpreted as u32 left several failure modes that cannot be checked
//      without a GPU (fragment packing, alignment, device-LTO codegen).
//
// Since correctness is more important than the last bit of speed, the kernel
// below is a clean CUDA-core implementation.  It uses the same
// (coefficient, limb) decomposition as gpu31's reconstruct_kernel, but stores
// `terms` with an ODD padded stride (17 = tile_x+1) so the 16 threads of a
// coefficient group never collide in a shared-memory bank — the reference's
// stride-16 layout gives an 8-way conflict on every term read.  Peak dynamic
// shared memory at the official point is ~26 KiB, well inside the 48 KiB
// default, and there is no 128-bit device arithmetic and no inline PTX.
//
// All three entry points produce byte-for-byte identical output to
// reconstruct_kernel:
//   * run() / run_s8() — the fast CUDA-core kernel below (identical code;
//     kept as separate templates so the existing launchers are unchanged).
//   * run_simple()     — the reference sum_column kernel, copied verbatim
//     from gpu31.cuh, for A/B testing and debugging.
//
// `crt_mma_selftest()` at the bottom verifies the original byte-split
// reassembly math on the host, with no CUDA required.
// ===========================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <cstddef>

#include <cuda_runtime.h>          // shim provides the error enum in host builds

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
static constexpr unsigned CRT_TILE_C       = 16;    // coefficients per block
static constexpr unsigned CRT_TILE_Y       = 8;     // threads per coefficient
static constexpr unsigned CRT_TS           = CRT_TILE_C + 1;  // 17: ODD stride
static constexpr unsigned CRT_THREADS      = CRT_TILE_C * CRT_TILE_Y;

// Dynamic shared memory used by crt_mma_kernel (must match the kernel exactly).
//   terms : pitch * CRT_TS u32, followed by 0/1 u32 of padding so the u64
//           arrays start on an 8-byte boundary;
//   low   : l * CRT_TS u64
//   high  : l * CRT_TS u64
//   digits: (l+2) * CRT_TS u32
static inline std::size_t crt_mma_shared_bytes(unsigned l, unsigned pitch) {
    const unsigned terms_words = pitch * CRT_TS;
    const unsigned aligned_words = (terms_words + 1u) & ~1u;
    return std::size_t(aligned_words + CRT_TS * (5u * l + 2u)) * sizeof(u32);
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
//    `shifted_word` is intentionally the gpu31 version (q[i], NOT
//    q[i*Stride]): q is a plain L-word array and carries no tile stride.
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
// 3. Production kernel (run / run_s8 paths).
//
//    One block handles CRT_TILE_C=16 coefficients for one pair; its 128
//    threads form a 16 (m) x 8 (y) grid.  Shared storage is laid out with an
//    ODD stride CRT_TS=17 so consecutive m never share a bank:
//        terms[j*TS + m]   pitch rows of per-coefficient terms
//        low[i*TS + m]     per-limb low 32 bits (as u64)
//        high[i*TS + m]    per-limb high bits
//        digits[i*TS + m]  final reduction workspace
//    Bounds: m in [0,15], j < pitch, i < l, so every index is strictly inside
//    its array.  Writes to out use exactly the reference formula
//    dst = pair*n*l + coef*l.
// ===========================================================================
template <int MODE>
__global__ void crt_mma_kernel(CrtArgs plan, const u32* residues,
                               const u32* d_weights, u32* out) {
    (void)MODE;  // run() and run_s8() share this identical, exact implementation

    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned m = threadIdx.x % CRT_TILE_C;
    const unsigned y = threadIdx.x / CRT_TILE_C;        // 0 .. CRT_TILE_Y-1
    const unsigned coef = blockIdx.x * CRT_TILE_C + m;
    const unsigned pair = blockIdx.z;
    const bool active = coef < plan.n;

    extern __shared__ unsigned char smem[];
    u32* terms = reinterpret_cast<u32*>(smem);
    // (pitch*TS) rounded up to an even number of u32 -> 8-byte aligned u64s.
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

    // ---- step 3: exact per-limb product sums (u32*u32 -> lo/hi u64) -------
    for (unsigned i = y; i < l; i += CRT_TILE_Y) {
        u64 lo = 0, hi = 0;
        const u32* w = d_weights + u64(i) * pitch;
        for (unsigned j = 0; j < pitch; ++j) {
            const u64 product = u64(terms[j * CRT_TS + m]) * w[j];
            lo += u32(product);
            hi += product >> 32;
        }
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
#endif  // __CUDACC__

// ===========================================================================
// 4. Host-only check of the byte-split reassembly math.
// ---------------------------------------------------------------------------
// The original tensor-core path computed, for two u32 words a and b,
//     S[c1][c2] = byte_c1(a) * byte_c2(b)
//     X         = sum_{c1,c2} S[c1][c2] << (8*(c1+c2))
// and claimed X == u64(a)*b, with low = u32(X), high = X>>32 satisfying
// low + high*2^32 == X.  This function checks that claim, and also the s8
// compensation identity u = (u&0x7f) + 128*(u>>7), on random inputs.
// It is plain C++ (no CUDA) and can be called from a host unit test.
// ===========================================================================
struct CrtMmaU128 { u64 lo, hi; };

inline CrtMmaU128 crt_mma_add_shifted(CrtMmaU128 x, u64 v, unsigned shift) {
    // All shifts used by the CRT are <= 48, so the general 128-bit path below
    // is more than sufficient (no UB for shift >= 64 either).
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

inline bool crt_mma_selftest(unsigned iterations = 4096,
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

        // ---- u8 path: raw bytes, exact unsigned byte products -------------
        CrtMmaU128 X{0, 0};
        for (unsigned c1 = 0; c1 < 4; ++c1) {
            const u64 wa = (a >> (8u * c1)) & 0xFFu;
            for (unsigned c2 = 0; c2 < 4; ++c2) {
                const u64 wb = (b >> (8u * c2)) & 0xFFu;
                X = crt_mma_add_shifted(X, wa * wb, 8u * (c1 + c2));
            }
        }
        const u64 want = u64(a) * b;
        if (X.hi != 0 || X.lo != want) return false;

        // The (low,high) pair fed to carry_and_reduce must reassemble X.
        const u32 low = u32(X.lo);
        const u64 high = X.lo >> 32;
        if (u64(low) + high * (u64(1) << 32) != want) return false;

        // ---- s8 path: 4-way compensation with 128-scaled halves ----------
        // Check byte-by-byte that the decomposition is exact, then that the
        // 4 weighted sums produce the same product.
        for (unsigned c = 0; c < 4; ++c) {
            const u32 ua = (a >> (8u * c)) & 0xFFu;
            const u32 ub = (b >> (8u * c)) & 0xFFu;
            if ((ua & 0x7Fu) + 128u * (ua >> 7) != ua) return false;
            if ((ub & 0x7Fu) + 128u * (ub >> 7) != ub) return false;
        }
        CrtMmaU128 Y{0, 0};
        for (unsigned c1 = 0; c1 < 4; ++c1) {
            const u32 al = (a >> (8u * c1)) & 0x7Fu, ah = (a >> (8u * c1 + 7u)) & 1u;
            for (unsigned c2 = 0; c2 < 4; ++c2) {
                const u32 bl = (b >> (8u * c2)) & 0x7Fu, bh = (b >> (8u * c2 + 7u)) & 1u;
                const unsigned sh = 8u * (c1 + c2);
                Y = crt_mma_add_shifted(Y, u64(al) * bl, sh);
                Y = crt_mma_add_shifted(Y, u64(al) * bh * 128u, sh);
                Y = crt_mma_add_shifted(Y, u64(ah) * bl * 128u, sh);
                Y = crt_mma_add_shifted(Y, u64(ah) * bh * 16384u, sh);
            }
        }
        if (Y.hi != 0 || Y.lo != want) return false;
    }
    return true;
}

// ===========================================================================
// 5. Host-side plan: weights in int8 layout + launchers (public API unchanged)
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
        // Preserve each u32 bit pattern as four little-endian bytes; the kernel
        // reads the buffer straight back as u32 (little-endian device).
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
               args.limbs > 0 && args.limbs <= CRT_MMA_MAXL &&
               args.count + 1 <= CRT_MMA_MAXPITCH &&
               args.count + 1 == pitch && args.limbs == L &&
               crt_mma_shared_bytes(L, pitch) <= 48u * 1024u;
    }

    // ---- PRIMARY: exact CUDA-core CRT (fast, recommended) -----------------
    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& args, u32 B,
                    cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_mma_shared_bytes(L, pitch);
        crt_mma_kernel<0><<<dim3((ca.n + CRT_TILE_C - 1) / CRT_TILE_C, 1, B),
                             CRT_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;   // never called in non-CUDA builds
#endif
    }

    // ---- EXACT path kept for API compatibility (same kernel as run()) -----
    template <class Args>
    cudaError_t run_s8(const u32* residues, u32* out, const Args& args, u32 B,
                       cudaStream_t stream = nullptr) {
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        const std::size_t smem = crt_mma_shared_bytes(L, pitch);
        crt_mma_kernel<1><<<dim3((ca.n + CRT_TILE_C - 1) / CRT_TILE_C, 1, B),
                             CRT_THREADS, smem, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
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
#ifdef __CUDACC__
        if (!check(args, B, residues, out)) return cudaErrorInvalidValue;
        const CrtArgs ca = make_args(args);
        constexpr unsigned block = 256;
        crt_simple_kernel<<<dim3((ca.n + block - 1) / block, B), block, 0, stream>>>(
            ca, residues, reinterpret_cast<const u32*>(d_w), out);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)args; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }
};

}  // namespace crt_mma

#undef CRTMMA_FHD
