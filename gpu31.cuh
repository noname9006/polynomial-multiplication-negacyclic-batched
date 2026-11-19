// gpu31.cuh — batched, 31-bit-RNS NTT GPU path.
//
// This is a NEW fast path.  It is CUDA-only: it is included under
// `#ifdef __CUDACC__` and is not meant to compile with tools/shim/cuda_shim.hpp.
// The original pgpu / polymul_gpu.cuh path and the host path are left intact and
// are what tools/gpucheck.cpp and the fallback in solve.cu keep using.
//
// Provenance.  The arithmetic below is a near-verbatim transcription of the
// public single-product reference that passes 40/40 cases at ~4.46 ms for B=1:
//   https://github.com/aritomas3123-boop/fherma-gpu-polymul/blob/f388bb1a373168bc9e0fd319daca6943c95cd80e/solve.cu
// specifically its `ari` (rns_core.h), `ari_shoup`, `fherma_crt` (rns_crt_core.hpp
// / rns_crt.cuh) and `ari_tile` (ntt_tile.cuh) sections.  The only structural
// change here is batching: every kernel gains a `blockIdx.z = pair` dimension and
// every residue buffer is offset by `pair * K * N` (coefficient buffers by
// `pair * N * L`).  Tables (primes, twiddles, twists, Shoup weights, CRT tables)
// are shared by all pairs and have no pair stride.
#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "fherma.h"

namespace gpu31 {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__)
#define GPU31_FHD __host__ __device__
#else
#define GPU31_FHD
#endif

// ===========================================================================
// 1. Device helpers
// ===========================================================================

// ---- rns_core.h (reference `ari`) -------------------------------------------
// Every prime is < 2^31, so t + m*p fits in u64 and a Montgomery product costs
// one 64-bit multiply plus one 32x32 high half.
struct Prime {
    u32 p, neg_inv, r2, one;
};

GPU31_FHD inline u32 mont(u32 a, u32 b, const Prime& p) {
    const u64 t = u64(a) * b;
    const u32 m = u32(t) * p.neg_inv;              // neg_inv = -p^-1 mod 2^32
    const u32 u = u32((t + u64(m) * p.p) >> 32);
    return u >= p.p ? u - p.p : u;
}
GPU31_FHD inline u32 add(u32 a, u32 b, u32 p) {
    const u32 s = a + b;
    return s >= p ? s - p : s;
}
GPU31_FHD inline u32 sub(u32 a, u32 b, u32 p) {
    return a >= b ? a - b : a + p - b;
}
GPU31_FHD inline u32 reverse_bits(u32 x, unsigned bits) {
    u32 r = 0;
    for (unsigned i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}
GPU31_FHD inline void butterfly(u32& a, u32& b, u32 w, const Prime& p) {
    const u32 u = a, v = mont(b, w, p);
    a = add(u, v, p.p);
    b = sub(u, v, p.p);
}

// ---- ari_shoup (reference solve.cu) -----------------------------------------
// Shoup-style limb reduction and the R=2^32 Montgomery `convert`.
struct Weight {
    u32 value, quotient;
};

// Encode weights in constant memory. A warp processing one prime reads the same
// K-L weight row; constant memory broadcasts that to every lane, replacing what
// would otherwise be ~430 GB of L2 reads at the biggest point. 120*56*8 = 53.8 KB
// fits in the 64 KB constant bank. Stride 56 covers the largest official L=55.
#define GPU31_MAXK_C 120
#define GPU31_MAXL_C 56
#if defined(__CUDACC__)
__constant__ Weight c_weights[GPU31_MAXK_C * GPU31_MAXL_C];
#endif

GPU31_FHD inline Weight make_weight(u32 value, u32 p) {
    return {value, u32((u64(value) << 32) / p)};
}
GPU31_FHD inline u32 lazy_product(u32 limb, Weight w, u32 p) {
    // For any limb < 2^32 the approximate quotient underestimates the true one
    // by at most 1; the residual is < 2p < 2^32.
#ifdef __CUDA_ARCH__
    const u32 quotient = __umulhi(limb, w.quotient);
#else
    const u32 quotient = u32((u64(limb) * w.quotient) >> 32);
#endif
    return limb * w.value - quotient * p;
}
GPU31_FHD inline u32 canonical_word(u32 x, u32 p) {
    if (x >= p) x -= p;
    if (x >= p) x -= p;
    if (x >= p) x -= p;
    return x;
}
GPU31_FHD inline u32 finish(u64 sum, Weight radix, u32 p) {
    const u32 lo = canonical_word(u32(sum), p);
    u32 hi = lazy_product(u32(sum >> 32), radix, p);
    if (hi >= p) hi -= p;
    return add(lo, hi, p);
}

// Fast path: a single __uint128_t accumulation replaces L per-limb Shoup
// reductions. Since lazy_product(a_l,W_l,p) == a_l*W_l (mod p),
//   r = (sum_l a_l*W_l) mod p.
// sum < 2^70; reduce via X = hiw*2^64 + mid32*2^32 + lo32 with w0 = 2^32 mod p.
// Requires p > 2^30 (true for every gpu31 plan). 2 mul reductions total.
GPU31_FHD inline u32 reduce128(u32 lo32, u32 mid32, u32 hiw,
                               const Weight& w0, u32 p, u64 r2) {
    u32 acc = canonical_word(lo32, p);
    u32 t = lazy_product(mid32, w0, p);
    if (t >= p) t -= p;
    acc += t; if (acc >= p) acc -= p;
    const u64 z = u64(hiw) * r2;            // hiw < 16, r2 < p < 2^31 -> z < 2^36
    const u32 zl = u32(z), zh = u32(z >> 32);
    const u32 zlc = canonical_word(zl, p);
    u32 tz = lazy_product(zh, w0, p);
    if (tz >= p) tz -= p;
    t = zlc + tz; if (t >= p) t -= p;
    acc += t; if (acc >= p) acc -= p;
    return acc;
}
GPU31_FHD inline u32 convert(const u32* input, const Weight* w, unsigned limbs, u32 p) {
    u64 sum = 0;
    for (unsigned j = 0; j < limbs; ++j) sum += lazy_product(input[j], w[j], p);
    return finish(sum, w[0], p);
}
// Two independent chains, to interleave the GPU's instruction streams.
GPU31_FHD inline void convert_both(const u32* a, const u32* b, const Weight* w,
                                   unsigned limbs, u32 p, u32& x, u32& y) {
    u64 sa = 0, sb = 0;
    for (unsigned j = 0; j < limbs; ++j) {
        const Weight t = w[j];
        sa += lazy_product(a[j], t, p);
        sb += lazy_product(b[j], t, p);
    }
    x = finish(sa, w[0], p);
    y = finish(sb, w[0], p);
}

// ---- fherma_crt (reference rns_crt_core.hpp / rns_crt.cuh) -------------------
// Exact centered CRT.  `inv` = (P/p_j)^-1 mod p_j, `inv_shoup` = floor(inv*2^32/p),
// `reciprocal` = 1/p_j (binary64).  All big integers are little-endian 2^32 words.
struct PrimeInfo {
    u32 p, inv, inv_shoup;
    double reciprocal;
};
GPU31_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}
template <unsigned Stride>
GPU31_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}
template <unsigned Stride>
GPU31_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
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
GPU31_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}
template <unsigned Stride>
GPU31_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
    const unsigned size = limbs + 2;
    // The sum is < 1025 * 2^31 * q.  With q = 2^W - gap and small gap, fold the
    // bits above W straight back into the low words (the official points all
    // use this path; the reference's generic shift loop remains as a fallback).
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
GPU31_FHD inline u32 crt_correction(const u32* terms, const PrimeInfo* primes, unsigned count) {
    // terms[j] = r_j (P/p_j)^-1 mod p_j, in [0,p_j).  round(sum_j terms_j/p_j)
    // is exactly the multiple of P to subtract (valid since |C| < P/2).  With
    // <= 1024 terms the binary64 error is < 1e-9, far inside the 1/4 margin.
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}
template <unsigned Stride>
GPU31_FHD inline void sum_column(const u32* terms, const u32* basis_column,
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
GPU31_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
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
// 2. Host-side plan construction (reference rns_core.h + rns_crt_core.hpp)
// ===========================================================================

inline u32 host_powmod(u32 a, u64 e, u32 p) {
    u32 r = 1;
    while (e) {
        if (e & 1) r = u32(u64(r) * a % p);
        a = u32(u64(a) * a % p);
        e >>= 1;
    }
    return r;
}
inline bool host_is_prime(u32 p) {
    if (p < 2) return false;
    for (u32 a : {2u, 3u, 5u, 7u, 11u, 13u, 17u}) {
        if (p == a) return true;
        if (p % a == 0) return false;
    }
    u32 d = p - 1, s = 0;
    while (!(d & 1)) { d >>= 1; ++s; }
    for (u32 a : {2u, 3u, 5u, 7u, 11u}) {
        u32 x = host_powmod(a, d, p);
        if (x == 1 || x == p - 1) continue;
        bool pass = false;
        for (unsigned i = 1; i < s; ++i) {
            x = u32(u64(x) * x % p);
            if (x == p - 1) { pass = true; break; }
        }
        if (!pass) return false;
    }
    return true;
}

// ---- CRT host helpers (reference fherma_crt) --------------------------------
using Words = std::vector<u32>;
inline unsigned word_bits(u32 a) {
    unsigned b = 0;
    while (a) { ++b; a >>= 1; }
    return b;
}
inline unsigned bit_length(const Words& a) {
    std::size_t size = a.size();
    while (size && !a[size - 1]) --size;
    return size ? unsigned(32 * (size - 1) + word_bits(a[size - 1])) : 0;
}
inline void multiply_small(Words& a, u32 b) {
    u64 carry = 0;
    for (u32& v : a) {
        carry += u64(v) * b;
        v = u32(carry);
        carry >>= 32;
    }
    if (carry) a.push_back(u32(carry));
}
inline Words divide_small(const Words& a, u32 b) {
    Words out(a.size());
    u64 rem = 0;
    for (std::size_t j = a.size(); j-- > 0;) {
        const u64 x = (rem << 32) | a[j];
        out[j] = u32(x / b);
        rem = x % b;
    }
    if (rem) throw std::logic_error("gpu31 CRT basis division");
    return out;
}
inline u32 mod_small(const Words& a, u32 p) {
    u64 r = 0;
    for (std::size_t i = a.size(); i-- > 0;) r = ((r << 32) | a[i]) % p;
    return u32(r);
}
inline Words reduce_wide(const Words& value, const Words& q) {
    Words out = value;
    const unsigned bits = bit_length(value), qb = bit_length(q);
    if (bits >= qb)
        for (unsigned s = bits - qb + 1; s-- > 0;)
            subtract_if_ge<1>(out.data(), unsigned(out.size()), q.data(), unsigned(q.size()), s);
    out.resize(q.size());
    return out;
}
inline Words subtract_words(const Words& a, const Words& b) {
    Words out(a.size());
    u64 borrow = 0;
    for (std::size_t j = 0; j < a.size(); ++j) {
        const u64 v = u64(j < b.size() ? b[j] : 0) + borrow;
        out[j] = a[j] - u32(v);
        borrow = u64(a[j]) < v;
    }
    if (borrow) throw std::logic_error("gpu31 negative CRT weight");
    return out;
}

// Everything the Plan / Device needs, built from a fherma::Point.
struct Plan {
    unsigned N, L, W, logN, K;

    // RNS + NTT tables (Montgomery form where applicable).
    std::vector<Prime> primes;
    std::vector<u32> roots, inverse_roots;  // K * N/2 each
    std::vector<u32> twist, untwist;        // K * N each
    std::vector<u32> reversed_twist;        // K * N, bit-reversed twist for encode
    std::vector<Weight> weights;            // K * L, Shoup limb weights

    // Centered-CRT tables (base 2^32), reference make_plan().
    unsigned crt_bits;
    u32 crt_gap;
    std::vector<u32> crt_q;          // L
    std::vector<PrimeInfo> crt_primes; // K
    std::vector<u32> crt_weights;    // L * (K+1); last column is the correction

    explicit Plan(const fherma::Point& point)
        : N(point.N), L(point.L), W(point.W), logN(0), K(0) {
        if (N < 2 || (N & (N - 1)) || N > 65536 || W < 2 || W > 1747 ||
            L != (W + 31) / 32 || point.q.data.size() != L)
            throw std::runtime_error("gpu31: supported points: power-of-two 2<=N<=65536, 2<=W<=1747, canonical L");
        while ((1u << logN) < N) ++logN;

        // Every prime > 2^30, so P > 2^(30K) >= 16*N*2^(2W): the centered CRT is
        // unambiguous with a wide margin.  K from the reference verbatim.
#ifndef TC24_TIGHT_K
#define TC24_TIGHT_K 1
#endif
#if TC24_TIGHT_K
        // TC24: primes sit just below 2^31 (~30.99 bits each), so pick primes
        // until their product has 2W + logN + 5 bits (build_crt demands
        // 2W + logN + 3); the 30-bit formula below over-provisions ~4 primes.
        K = 0;
        {
            const double need = 2.0 * W + logN + 5.0;
            double have = 0;
            u32 q = ((u32(1) << 31) - 2) / (2 * N) * (2 * N) + 1;
            while (have < need) {
                while (q > (1u << 30) && !host_is_prime(q)) q -= 2 * N;
                if (q <= (1u << 30)) break;
                have += std::log2(double(q));
                ++K;
                q -= 2 * N;
            }
            const unsigned classic = (2 * W + logN + 4 + 29) / 30;
            if (have < need || K > classic) K = classic;
        }
#else
        K = (2 * W + logN + 4 + 29) / 30;
#endif
        u32 p = ((u32(1) << 31) - 2) / (2 * N) * (2 * N) + 1;
        for (unsigned k = 0; k < K; ++k) {
            while (p > (1u << 30) && !host_is_prime(p)) p -= 2 * N;
            if (p <= (1u << 30)) throw std::runtime_error("gpu31: not enough suitable RNS primes");
            u32 inv = 1;
            for (unsigned j = 0; j < 5; ++j) inv *= 2 - p * inv;   // p^-1 mod 2^32
            const u32 one = u32((u64(1) << 32) % p);               // 2^32 mod p
            primes.push_back({p, u32(0 - inv), u32(u64(one) * one % p), one});
            p -= 2 * N;
        }

        // Primitive 2N-th roots (psi), N-th roots (omega=psi^2), twists.
        roots.resize(std::size_t(K) * (N / 2));
        inverse_roots.resize(roots.size());
        twist.resize(std::size_t(K) * N);
        untwist.resize(twist.size());
        reversed_twist.resize(twist.size());
        for (unsigned k = 0; k < K; ++k) {
            const Prime& pr = primes[k];
            u32 psi = 0;
            for (u32 a = 2; !psi; ++a) {
                const u32 cand = host_powmod(a, (pr.p - 1) / (2 * N), pr.p);
                if (host_powmod(cand, N, pr.p) == pr.p - 1) psi = cand;
            }
            const u32 psi_inv = host_powmod(psi, pr.p - 2, pr.p);
            const u32 omega = u32(u64(psi) * psi % pr.p);
            const u32 omega_inv = host_powmod(omega, pr.p - 2, pr.p);
            const u32 n_inv = host_powmod(N, pr.p - 2, pr.p);
            u32 a = 1, b = 1;
            for (unsigned i = 0; i < N / 2; ++i) {
                roots[std::size_t(k) * (N / 2) + i] = mont(a, pr.r2, pr);
                inverse_roots[std::size_t(k) * (N / 2) + i] = mont(b, pr.r2, pr);
                a = u32(u64(a) * omega % pr.p);
                b = u32(u64(b) * omega_inv % pr.p);
            }
            a = 1; b = n_inv;
            for (unsigned i = 0; i < N; ++i) {
                twist[std::size_t(k) * N + i] = mont(a, pr.r2, pr);
                untwist[std::size_t(k) * N + i] = mont(b, pr.r2, pr);
                a = u32(u64(a) * psi % pr.p);
                b = u32(u64(b) * psi_inv % pr.p);
            }
            for (unsigned i = 0; i < N; ++i)
                reversed_twist[std::size_t(k) * N + reverse_bits(i, logN)] =
                    twist[std::size_t(k) * N + i];
        }

        // Shoup limb weights W_j = 2^(32(j+1)) mod p (reference make_weights).
        weights.resize(std::size_t(K) * L);
        for (unsigned k = 0; k < K; ++k) {
            u32 factor = primes[k].one;   // R mod p
            for (unsigned j = 0; j < L; ++j) {
                weights[std::size_t(k) * L + j] = make_weight(factor, primes[k].p);
                factor = u32(u64(factor) * primes[k].one % primes[k].p);
            }
        }

        // ---- centered CRT plan (reference make_plan) -----------------------
        std::vector<u32> pvals;
        pvals.reserve(K);
        for (unsigned k = 0; k < K; ++k) pvals.push_back(primes[k].p);
        build_crt(pvals, point.q.data.data());
    }

private:
    void build_crt(const std::vector<u32>& ps, const u32* q) {
        crt_q.assign(q, q + L);
        crt_bits = bit_length(crt_q);
        if (crt_bits < 2 || (crt_bits + 31) / 32 != L)
            throw std::runtime_error("gpu31: invalid CRT q width");
        const Words two_w(crt_bits / 32 + 1, 0u);
        Words delta(two_w);
        delta[crt_bits / 32] = u32(1) << (crt_bits % 32);
        Words gap_words = subtract_words(delta, crt_q);
        crt_gap = bit_length(gap_words) <= 32 ? gap_words[0] : 0;

        Words product{1};
        for (u32 p : ps) {
            if (p < 3 || p >= 0x80000000u)
                throw std::runtime_error("gpu31: CRT expects primes below 2^31");
            multiply_small(product, p);
        }
        if (bit_length(product) < 2 * crt_bits + word_bits(N) + 2)
            throw std::runtime_error("gpu31: CRT prime product lacks two guard bits");

        const std::size_t count = ps.size(), pitch = count + 1;
        crt_weights.assign(std::size_t(L) * pitch, 0);
        for (std::size_t j = 0; j <= count; ++j) {
            Words reduced;
            if (j < count) {
                const u32 prime = ps[j];
                const Words basis = divide_small(product, prime);
                const u32 basis_r = mod_small(basis, prime);
                const u32 inv = host_powmod(basis_r, prime - 2, prime);
                if (u64(basis_r) * inv % prime != 1)
                    throw std::runtime_error("gpu31: CRT moduli must be distinct primes");
                crt_primes.push_back({prime, inv, u32((u64(inv) << 32) / prime),
                                      1.0 / double(prime)});
                reduced = reduce_wide(basis, crt_q);
            } else {
                // q - (P mod q): the correction column (-P mod q).
                reduced = subtract_words(crt_q, reduce_wide(product, crt_q));
            }
            for (unsigned i = 0; i < L; ++i)
                crt_weights[i * pitch + j] = reduced[i];
        }
    }
};

// ===========================================================================
// 3. Batched device kernels (blockIdx.z = pair; see provenance header)
// ===========================================================================

constexpr unsigned warp_size    = 32;
constexpr unsigned reindex_rows = 8;
constexpr unsigned group_primes = 8;
constexpr unsigned max_tile     = 512;   // ari_tile::max_tile
constexpr unsigned low_columns  = 16;    // ari_tile::low_columns
constexpr unsigned high_workers = 16;    // ari_tile::high_workers
constexpr unsigned crt_tile_x   = 16;    // fherma_crt::tile_x
constexpr unsigned crt_tile_y   = 8;     // fherma_crt::tile_y

// ---- reindex (reference ari_shoup::reindex) --------------------------------
// Bit reversal is an involution; swap whole coefficient rows (all L limbs) of
// a pair's limb-major array.  No two blocks touch the same address.
__global__ void reindex(u32* a, u32* b, unsigned n, unsigned l, unsigned bits) {
    const unsigned i = blockIdx.x * reindex_rows + threadIdx.y;
    if (i >= n) return;
    const unsigned r = reverse_bits(i, bits);
    if (r <= i) return;
    const u64 base = u64(blockIdx.z) * n * l;
    u32* pa = a + base;
    u32* pb = b + base;
    for (unsigned j = threadIdx.x; j < l; j += warp_size) {
        const u64 u = u64(i) * l + j, v = u64(r) * l + j;
        const u32 ai = pa[u], ar = pa[v];
        const u32 bi = pb[u], br = pb[v];
        pa[u] = ar; pa[v] = ai;
        pb[u] = br; pb[v] = bi;
    }
}

// ---- encode_tiled (reference ari_shoup::encode_tiled) ----------------------
// Reduce L limbs of each coefficient with Shoup weights, multiply by the
// bit-reversed twist, and write pair's residues [pair][prime][coefficient].
__global__ void encode_tiled(const u32* ar, const u32* br, u32* ra, u32* rb,
                             const Prime* primes, const Weight* /*weights*/,
                             const u32* reversed_twist,
                             unsigned n, unsigned l, unsigned count) {
    // Shared staging for the coefficient bytes (read once per tile) plus
    // constant-memory broadcast for the Shoup weight row. The direct-global
    // variant re-read ~430 GB from L2 and was 1.4x slower; the earlier odd
    // stride (l|1) caused bank conflicts. Stride MAXL_C=56 is conflict-free.
    extern __shared__ u32 stage[];
    // Odd stride coprime with 32 avoids shared-memory bank conflicts: with an
    // even stride, 32 consecutive coefficient rows hit only 4 distinct banks.
    const unsigned stride = GPU31_MAXL_C + 1;
    u32* sa = stage;
    u32* sb = stage + warp_size * stride;
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const unsigned start = blockIdx.x * warp_size;
    const unsigned valid = (n - start < warp_size) ? (n - start) : warp_size;
    const u64 base = u64(blockIdx.z) * n * l;
    const u64 rbase = u64(blockIdx.z) * count * n;

    for (unsigned t = y * warp_size + x; t < valid * l; t += warp_size * group_primes) {
        const unsigned row = t / l, col = t - row * l;
        const u64 off = base + u64(start) * l + t;
        sa[row * stride + col] = ar[off];
        sb[row * stride + col] = br[off];
    }
    __syncthreads();
    const unsigned m = blockIdx.y * group_primes + y;
    if (m >= count || x >= valid) return;
    const Prime p = primes[m];
    const u64 off = rbase + u64(m) * n + start + x;
    u32 a, b;
#if defined(__CUDACC__)
    const Weight* wrow = c_weights + u64(m) * GPU31_MAXL_C;
    // Fast single-pass reduction: sum a_l*W_l in 128 bits, reduce once.
    __uint128_t suma = 0, sumb = 0;
    const u32* ca = sa + x * stride;
    const u32* cb = sb + x * stride;
    #pragma unroll 4
    for (unsigned j = 0; j < l; ++j) {
        suma += (__uint128_t)(u64(ca[j]) * u64(wrow[j].value));
        sumb += (__uint128_t)(u64(cb[j]) * u64(wrow[j].value));
    }
    const u64 loa = u64(suma), lob = u64(sumb);
    a = reduce128(u32(loa), u32(loa >> 32), u32(suma >> 64), wrow[0], p.p, p.r2);
    b = reduce128(u32(lob), u32(lob >> 32), u32(sumb >> 64), wrow[0], p.p, p.r2);
#else
    const Weight* wrow = weights + u64(m) * l;
    convert_both(sa + x * stride, sb + x * stride, wrow, l, p.p, a, b);
#endif
    const u32 twist = reversed_twist[u64(m) * n + start + x];
    ra[off] = mont(a, twist, p);
    rb[off] = mont(b, twist, p);
}

// ---- tiled DIT NTT (reference ari_tile::low_stages / high_stages) -----------
struct NttPair { unsigned first, second, root; };
GPU31_FHD inline NttPair low_pair(unsigned t, unsigned half, unsigned n) {
    const unsigned j = t & (half - 1), first = 2 * (t - j) + j;
    return {first, first + half, j * (n / (2 * half))};
}
GPU31_FHD inline NttPair high_pair(unsigned h, unsigned half_high, unsigned low, unsigned n) {
    const unsigned within = h & (half_high - 1);
    const unsigned first = 2 * (h - within) + within;
    return {first, first + half_high,
            (within * max_tile + low) * (n / (2 * half_high * max_tile))};
}

template <bool Two>
__global__ void low_stages(u32* a, u32* b, const Prime* primes,
                           const u32* roots, unsigned n, unsigned count, unsigned tile) {
    extern __shared__ u32 data[];
    u32* sa = data;
    u32* sb = data + tile;
    const unsigned t = threadIdx.x;
    const u64 base = u64(blockIdx.z) * count * n + u64(blockIdx.y) * n + u64(blockIdx.x) * tile;
    const u32* table = roots + u64(blockIdx.y) * (n / 2);
    const Prime p = primes[blockIdx.y];
    sa[t] = a[base + t];
    sa[t + blockDim.x] = a[base + t + blockDim.x];
    if constexpr (Two) {
        sb[t] = b[base + t];
        sb[t + blockDim.x] = b[base + t + blockDim.x];
    }
    __syncthreads();
    for (unsigned half = 1; half < tile; half *= 2) {
        const NttPair pair = low_pair(t, half, n);
        const u32 w = table[pair.root];
        butterfly(sa[pair.first], sa[pair.second], w, p);
        if constexpr (Two) butterfly(sb[pair.first], sb[pair.second], w, p);
        __syncthreads();
    }
    a[base + t] = sa[t];
    a[base + t + blockDim.x] = sa[t + blockDim.x];
    if constexpr (Two) {
        b[base + t] = sb[t];
        b[base + t + blockDim.x] = sb[t + blockDim.x];
    }
}

template <bool Two>
__global__ void high_stages(u32* a, u32* b, const Prime* primes,
                            const u32* roots, unsigned n, unsigned count) {
    extern __shared__ u32 data[];
    const unsigned high_count = n / max_tile;
    u32* sa = data;
    u32* sb = data + high_count * low_columns;
    const unsigned lane = threadIdx.x, row_worker = threadIdx.y;
    const unsigned low = blockIdx.x * low_columns + lane;
    const u64 plane = u64(blockIdx.z) * count * n + u64(blockIdx.y) * n;
    const u32* table = roots + u64(blockIdx.y) * (n / 2);
    const Prime p = primes[blockIdx.y];
    for (unsigned h = row_worker; h < high_count; h += high_workers) {
        const unsigned local = h * low_columns + lane;
        const u64 global = plane + u64(h) * max_tile + low;
        sa[local] = a[global];
        if constexpr (Two) sb[local] = b[global];
    }
    __syncthreads();
    for (unsigned half_high = 1; half_high < high_count; half_high *= 2) {
        for (unsigned h = row_worker; h < high_count / 2; h += high_workers) {
            const NttPair pair = high_pair(h, half_high, low, n);
            const unsigned first = pair.first * low_columns + lane;
            const unsigned second = pair.second * low_columns + lane;
            const u32 w = table[pair.root];
            butterfly(sa[first], sa[second], w, p);
            if constexpr (Two) butterfly(sb[first], sb[second], w, p);
        }
        __syncthreads();
    }
    for (unsigned h = row_worker; h < high_count; h += high_workers) {
        const unsigned local = h * low_columns + lane;
        const u64 global = plane + u64(h) * max_tile + low;
        a[global] = sa[local];
        if constexpr (Two) b[global] = sb[local];
    }
}

// ---- pointwise product, written to bit-reversed position for the inverse ---
__global__ void multiply_reverse(const u32* a, const u32* b, u32* c,
                                 const Prime* primes, unsigned n, unsigned count,
                                 unsigned logN) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned m = blockIdx.y;
    if (i >= n) return;
    const u64 base = u64(blockIdx.z) * count * n + u64(m) * n;
    const u64 at = base + i;
    c[base + reverse_bits(i, logN)] = mont(a[at], b[at], primes[m]);
}

// ---- untwist + Montgomery normalize (reference solve.cu::decode) -----------
__global__ void decode(u32* a, const Prime* primes, const u32* untwist,
                       unsigned n, unsigned count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned m = blockIdx.y;
    if (i >= n) return;
    const u64 at = u64(blockIdx.z) * count * n + u64(m) * n + i;
    const Prime p = primes[m];
    a[at] = mont(mont(a[at], untwist[u64(m) * n + i], p), 1u, p);
}

// ---- centered CRT reconstruction (reference fherma_crt::reconstruct_kernel) -
struct CrtArgs {
    unsigned n, limbs, bits, count;
    u32 gap;
    const PrimeInfo* primes;
    const u32* weights;
    const u32* q;
};
__global__ void reconstruct_kernel(CrtArgs plan, const u32* residues, u32* out) {
    extern __shared__ unsigned long long shared[];
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const unsigned coef = blockIdx.x * crt_tile_x + x;
    const unsigned pair = blockIdx.z;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    u64* low = reinterpret_cast<u64*>(shared);
    u64* high = low + l * crt_tile_x;
    u32* terms = reinterpret_cast<u32*>(high + l * crt_tile_x);
    u32* digits = terms + pitch * crt_tile_x;
    const bool active = coef < plan.n;
    const u64 rbase = u64(pair) * k * plan.n;

    for (unsigned j = y; j < k; j += crt_tile_y) {
        const u32 r = active ? residues[rbase + u64(j) * plan.n + coef] : 0;
        terms[j * crt_tile_x + x] = weight_residue(r, plan.primes[j]);
    }
    __syncthreads();
    if (y == 0) terms[k * crt_tile_x + x] =
        crt_correction<crt_tile_x>(terms + x, plan.primes, k);
    __syncthreads();
    for (unsigned i = y; i < l; i += crt_tile_y)
        sum_column<crt_tile_x>(terms + x, plan.weights + i * pitch, pitch,
                               low[i * crt_tile_x + x], high[i * crt_tile_x + x]);
    __syncthreads();
    if (y == 0 && active) {
        carry_and_reduce<crt_tile_x>(digits + x, low + x, high + x, l,
                                     plan.bits, plan.gap, plan.q);
        const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
#pragma unroll
        for (unsigned i = 0; i < l; ++i)
            out[dst + i] = digits[i * crt_tile_x + x];
    }
}

}  // namespace gpu31
#include "tc.cuh"
namespace gpu31 {

// ===========================================================================
// 4. Host-side Device: uploads tables once, owns scratch, launches the pipeline
// ===========================================================================

struct Device {
    Plan plan_;
    unsigned capacity_;   // max pairs, from point.B
    CrtArgs crt_args_{};
    std::size_t crt_shared_ = 0;

    // device tables (shared across pairs)
    Prime* d_primes = nullptr;
    Weight* d_weights = nullptr;
    u32 *d_roots = nullptr, *d_inv_roots = nullptr;
    u32 *d_twist = nullptr, *d_untwist = nullptr, *d_reversed_twist = nullptr;
    PrimeInfo* d_crt_primes = nullptr;
    u32 *d_crt_weights = nullptr, *d_crt_q = nullptr;

    // per-pair scratch, sized for capacity_ pairs
    u32 *d_ra = nullptr, *d_rb = nullptr, *d_rc = nullptr;

    // Tensor-core encode / CRT (tc.cuh).  Built in the constructor when the
    // point fits; enabled by the caller only after a self-check passes.
    uint2 *d_fenc = nullptr, *d_fcrt = nullptr;
    u32* d_untw_inv = nullptr;   // mont-form untwist * (P/p)^-1, for fused decode
    u32* d_corr_m = nullptr;     // floor(2^62 / p_k), fixed-point CRT correction
    bool tc_fuse_decode = false;
    // TC3 fused NTT pipeline tables
    u32 *d_pk_fwd = nullptr, *d_pk_inv = nullptr, *d_twf = nullptr, *d_twi = nullptr,
        *d_untw_slot = nullptr;
    bool tc_ntt_ready = false, tc_ntt = false;
    bool tc_efl = false;   // TC5: encode fused with fwd low stages
    bool tc_crt2 = false;  // TC15 CRT variant
    bool tc_invcrt = false; // TC26: inv_low fused into CRT (cluster of 8, DSMEM)
    std::size_t invcrt_shared = 0;
    unsigned invcrt_rows = 32;
    bool tc_tail = false;  // TC17: per-pair tail on two streams
    unsigned crt2_SA = 0;
    std::size_t crt2_shared = 0;
    bool tc_chunk = false; // TC8: pair-at-a-time on two streams (L2-resident residues)
    cudaStream_t cs_[2] = {nullptr, nullptr};
    cudaEvent_t evf_ = nullptr, evj_[2] = {nullptr, nullptr};
    unsigned enc_KS = 0, enc_NT = 0;
    tc::CrtTc crt_cfg{};
    std::size_t crt_tc_shared = 0;
    bool tc_enc_ready = false, tc_crt_ready = false;
    bool tc_enc = false, tc_crt = false;

    // Per-stage timing (diagnostic). 8 events bracket the 7 pipeline stages.
    static constexpr int kStages = 7;
    cudaEvent_t ev_[kStages + 1] = {};
    float ms_[kStages] = {};
    bool profiling_ = false;

    explicit Device(const fherma::Point& p) : plan_(p), capacity_(p.B) {
        if (p.B == 0) throw std::runtime_error("gpu31: point B must be positive");
        if (plan_.K > GPU31_MAXK_C || plan_.L > GPU31_MAXL_C)
            throw std::runtime_error("gpu31: point exceeds constant table limits");
        try {
            // Constant-memory copy of the encode weights, padded to the stride
            // the kernel uses so the same row broadcasts to a whole warp.
            {
                std::vector<Weight> padded(std::size_t(GPU31_MAXK_C) * GPU31_MAXL_C);
                for (unsigned m = 0; m < plan_.K; ++m)
                    for (unsigned j = 0; j < plan_.L; ++j)
                        padded[std::size_t(m) * GPU31_MAXL_C + j] =
                            plan_.weights[std::size_t(m) * plan_.L + j];
                cudaError_t cs = cudaMemcpyToSymbol(c_weights, padded.data(),
                                                    padded.size() * sizeof(Weight));
                if (cs != cudaSuccess)
                    throw std::runtime_error("gpu31: constant weight upload failed");
            }
            upload(d_primes, plan_.primes);
            upload(d_weights, plan_.weights);
            upload(d_roots, plan_.roots);
            upload(d_inv_roots, plan_.inverse_roots);
            upload(d_twist, plan_.twist);
            upload(d_untwist, plan_.untwist);
            upload(d_reversed_twist, plan_.reversed_twist);
            upload(d_crt_primes, plan_.crt_primes);
            upload(d_crt_weights, plan_.crt_weights);
            upload(d_crt_q, plan_.crt_q);

            const std::size_t residues = std::size_t(capacity_) * plan_.K * plan_.N;
            d_ra = static_cast<u32*>(raw_alloc(residues * sizeof(u32)));
            d_rb = static_cast<u32*>(raw_alloc(residues * sizeof(u32)));
            d_rc = static_cast<u32*>(raw_alloc(residues * sizeof(u32)));

            crt_args_.n = plan_.N;
            crt_args_.limbs = plan_.L;
            crt_args_.bits = plan_.crt_bits;
            crt_args_.count = plan_.K;
            crt_args_.gap = plan_.crt_gap;
            crt_args_.primes = d_crt_primes;
            crt_args_.weights = d_crt_weights;
            crt_args_.q = d_crt_q;
            crt_shared_ = crt_tile_x *
                (2 * sizeof(u64) * plan_.L +
                 sizeof(u32) * (plan_.K + 1 + plan_.L + 2));

            int device = 0;
            cudaDeviceProp prop{};
            if (cudaGetDevice(&device) != cudaSuccess ||
                cudaGetDeviceProperties(&prop, device) != cudaSuccess)
                throw std::runtime_error("gpu31: cannot query device properties");
            if (crt_shared_ > prop.sharedMemPerBlock)
                throw std::runtime_error("gpu31: CRT tile exceeds shared memory capacity");
            build_tc(prop);
            if (cudaStreamCreateWithFlags(&cs_[0], cudaStreamNonBlocking) != cudaSuccess ||
                cudaStreamCreateWithFlags(&cs_[1], cudaStreamNonBlocking) != cudaSuccess ||
                cudaEventCreateWithFlags(&evf_, cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&evj_[0], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&evj_[1], cudaEventDisableTiming) != cudaSuccess) {
                cudaGetLastError();
                cs_[0] = cs_[1] = nullptr;
            }
            for (int i = 0; i <= kStages; ++i)
                if (cudaEventCreate(&ev_[i]) != cudaSuccess) break;
            profiling_ = (ev_[kStages] != nullptr);
        } catch (...) {
            release();
            throw;
        }
    }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    ~Device() { release(); }

    // All buffers (d_a/d_b/d_c) are owned by the caller.  Only the persistent
    // tables and the scratch above belong to this object.  Asynchronous: all
    // launches go to `stream`; the caller checks completion when consuming out.
    // Optional external NTT hook: when set, replaces transform<true/false>
    // in stages 3 and 5 (e.g. an optimized NTT implementation).
    cudaError_t (*ntt_hook_)(u32*, u32*, unsigned /*B*/, cudaStream_t, bool /*fwd*/, void*) = nullptr;
    void* ntt_hook_ctx_ = nullptr;
    void set_ntt_hook(cudaError_t (*h)(u32*, u32*, unsigned, cudaStream_t, bool, void*), void* ctx) {
        ntt_hook_ = h;
        ntt_hook_ctx_ = ctx;
    }

    cudaError_t run(const u32* d_a, const u32* d_b, u32* d_c, u32 B,
                    cudaStream_t stream = nullptr, bool skip_crt = false) {
        if (B == 0) return cudaSuccess;
        if (B > capacity_ || !d_a || !d_b || !d_c) return cudaErrorInvalidValue;
        const unsigned n = plan_.N, l = plan_.L, k = plan_.K, logN = plan_.logN;
        cudaError_t st = cudaSuccess;
        if (profiling_) cudaEventRecord(ev_[0], stream);

        if (tc_ntt && tc_enc && tc_crt && !skip_crt) {
            if (tc_chunk && cs_[0] && cs_[1]) {
                // TC8: pairs one at a time on two streams; one pair's residues
                // (3 x K x N x 4 B ~ 93 MB) stay in L2 between the stages.
                cudaEventRecord(evf_, stream);
                cudaStreamWaitEvent(cs_[0], evf_, 0);
                cudaStreamWaitEvent(cs_[1], evf_, 0);
                const std::size_t in_stride = std::size_t(n) * l;
                const std::size_t res_stride = std::size_t(k) * n;
                for (u32 j = 0; j < B; ++j) {
                    const unsigned sl = j & 1u;
                    st = tc_pipeline(d_a + j * in_stride, d_b + j * in_stride, d_c + j * in_stride,
                                     d_ra + sl * res_stride, d_rb + sl * res_stride,
                                     d_rc + sl * res_stride, 1, cs_[sl], false);
                    if (st != cudaSuccess) return st;
                }
                cudaEventRecord(evj_[0], cs_[0]);
                cudaEventRecord(evj_[1], cs_[1]);
                cudaStreamWaitEvent(stream, evj_[0], 0);
                cudaStreamWaitEvent(stream, evj_[1], 0);
                if (profiling_)
                    for (int e = 1; e <= kStages; ++e) cudaEventRecord(ev_[e], stream);
                return cudaGetLastError();
            }
            if (tc_tail && cs_[0] && cs_[1] && B > 1) {
                // TC17: encode batched over all pairs; the tail (fused_high,
                // inv_low, CRT) pair by pair on two streams so each pair's rc
                // (K x N x 4 B ~ 31 MB) stays in L2 between the three kernels.
                st = tc_pipeline(d_a, d_b, d_c, d_ra, d_rb, d_rc, B, stream, profiling_, 1);
                if (st != cudaSuccess) return st;
                cudaEventRecord(evf_, stream);
                cudaStreamWaitEvent(cs_[0], evf_, 0);
                cudaStreamWaitEvent(cs_[1], evf_, 0);
                const std::size_t in_stride = std::size_t(n) * l;
                const std::size_t res_stride = std::size_t(k) * n;
                for (u32 j = 0; j < B; ++j) {
                    const unsigned sl = j & 1u;
                    st = tc_pipeline(d_a + j * in_stride, d_b + j * in_stride, d_c + j * in_stride,
                                     d_ra + j * res_stride, d_rb + j * res_stride,
                                     d_rc + sl * res_stride, 1, cs_[sl], false, 2);
                    if (st != cudaSuccess) return st;
                }
                cudaEventRecord(evj_[0], cs_[0]);
                cudaEventRecord(evj_[1], cs_[1]);
                cudaStreamWaitEvent(stream, evj_[0], 0);
                cudaStreamWaitEvent(stream, evj_[1], 0);
                if (profiling_)
                    for (int e = 4; e <= kStages; ++e) cudaEventRecord(ev_[e], stream);
                return cudaGetLastError();
            }
            return tc_pipeline(d_a, d_b, d_c, d_ra, d_rb, d_rc, B, stream, profiling_);
        }

        if (tc_enc) {
            // 1+2. tensor-core encode reads bit-reversed rows directly.
            tc::encode_tc<<<dim3((n + tc::ENC_ROWS - 1) / tc::ENC_ROWS, 2, B), 256, 0, stream>>>(
                d_a, d_b, d_ra, d_rb, d_fenc, d_primes, d_weights, d_reversed_twist,
                n, l, k, logN, enc_KS, enc_NT);
            if ((st = cudaGetLastError()) != cudaSuccess) return st;
            if (profiling_) { cudaEventRecord(ev_[1], stream); cudaEventRecord(ev_[2], stream); }
        } else {
        // 1. bit-reverse coefficient positions in the pair's a/b arrays (in place)
        reindex<<<dim3((n + reindex_rows - 1) / reindex_rows, 1, B),
                  dim3(warp_size, reindex_rows), 0, stream>>>(
            const_cast<u32*>(d_a), const_cast<u32*>(d_b), n, l, logN);
        if ((st = cudaGetLastError()) != cudaSuccess) return st;
        if (profiling_) cudaEventRecord(ev_[1], stream);

        // 2. Shoup-reduce limbs, apply bit-reversed twist -> ra / rb
        const std::size_t enc_shared =
            2 * warp_size * (GPU31_MAXL_C + 1) * sizeof(u32);
        encode_tiled<<<dim3((n + warp_size - 1) / warp_size,
                            (k + group_primes - 1) / group_primes, B),
                       dim3(warp_size, group_primes), enc_shared, stream>>>(
            d_a, d_b, d_ra, d_rb, d_primes, d_weights, d_reversed_twist, n, l, k);
        if ((st = cudaGetLastError()) != cudaSuccess) return st;
        if (profiling_) cudaEventRecord(ev_[2], stream);
        }

        // 3. forward NTT of both operands
        if (ntt_hook_)
            st = ntt_hook_(d_ra, d_rb, B, stream, true, ntt_hook_ctx_);
        else
            st = transform<true>(d_ra, d_rb, d_roots, n, k, B, stream);
        if (st != cudaSuccess) return st;
        if (profiling_) cudaEventRecord(ev_[3], stream);

        // 4. pointwise product, written in bit-reversed order for the inverse
        multiply_reverse<<<dim3((n + 255) / 256, k, B), 256, 0, stream>>>(
            d_ra, d_rb, d_rc, d_primes, n, k, logN);
        if ((st = cudaGetLastError()) != cudaSuccess) return st;
        if (profiling_) cudaEventRecord(ev_[4], stream);

        // 5. inverse NTT of the product
        if (ntt_hook_)
            st = ntt_hook_(d_rc, nullptr, B, stream, false, ntt_hook_ctx_);
        else
            st = transform<false>(d_rc, nullptr, d_inv_roots, n, k, B, stream);
        if (st != cudaSuccess) return st;
        if (profiling_) cudaEventRecord(ev_[5], stream);

        // 6. untwist / normalize (fused into the tensor-core CRT when enabled)
        if (!(tc_crt && tc_fuse_decode && !skip_crt)) {
            decode<<<dim3((n + 255) / 256, k, B), 256, 0, stream>>>(
                d_rc, d_primes, d_untwist, n, k);
            if ((st = cudaGetLastError()) != cudaSuccess) return st;
        }
        if (profiling_) cudaEventRecord(ev_[6], stream);

        // 7. centered CRT -> canonical [pair][coefficient][limb] output in d_c.
        // When skip_crt is set, the caller runs an external CRT (e.g. tensor-core
        // MMA) on d_rc after this pipeline returns.
        if (!skip_crt && tc_crt) {
            tc::crt_tc<<<dim3((n + tc::CRT_ROWS - 1) / tc::CRT_ROWS, 1, B), 128,
                          crt_tc_shared, stream>>>(crt_args_, d_rc, d_c, d_fcrt, crt_cfg,
                                                   d_primes, tc_fuse_decode ? d_untw_inv : nullptr, 0u, d_corr_m);
            st = cudaGetLastError();
        } else if (!skip_crt) {
            reconstruct_kernel<<<dim3((n + crt_tile_x - 1) / crt_tile_x, 1, B),
                                 dim3(crt_tile_x, crt_tile_y), crt_shared_, stream>>>(
                crt_args_, d_rc, d_c);
            st = cudaGetLastError();
        }
        if (profiling_) cudaEventRecord(ev_[7], stream);
        return st;
    }

    // Accessors for the tensor-core CRT integration in solve.cu.
    const Plan& plan() const { return plan_; }
    const CrtArgs& crt_args() const { return crt_args_; }
    const u32* residues_device() const { return d_rc; }
    const Prime* primes_device() const { return d_primes; }
    const u32* roots_device() const { return d_roots; }
    const u32* inv_roots_device() const { return d_inv_roots; }

    // Run the reference (CUDA-core) CRT into `out` for `pairs` pairs.
    cudaError_t crt_reference(const u32* residues, u32* out, u32 pairs,
                              cudaStream_t stream) const {
        if (pairs == 0) return cudaSuccess;
        reconstruct_kernel<<<dim3((plan_.N + crt_tile_x - 1) / crt_tile_x, 1, pairs),
                             dim3(crt_tile_x, crt_tile_y), crt_shared_, stream>>>(
            crt_args_, residues, out);
        return cudaGetLastError();
    }

    // Read and print the stage times. Call after the stream has synchronised.
    void report_profile() const {
        if (!profiling_) return;
        static const char* names[kStages] = {
            "reindex", "encode", "ntt_fwd", "pointwise", "ntt_inv", "decode", "crt"};
        float sum = 0.f;
        for (int i = 0; i < kStages; ++i) {
            float t = 0.f;
            cudaEventElapsedTime(&t, ev_[i], ev_[i + 1]);
            sum += t;
        }
        std::fprintf(stderr, "gpu31 stages (ms):");
        for (int i = 0; i < kStages; ++i) {
            float t = 0.f;
            cudaEventElapsedTime(&t, ev_[i], ev_[i + 1]);
            std::fprintf(stderr, " %s %.3f", names[i], t);
        }
        std::fprintf(stderr, " | total %.3f\n", sum);
        {
            unsigned long long h[16] = {};
            if (cudaMemcpyFromSymbol(h, tc::g_tcprof, sizeof(h)) == cudaSuccess) {
                const char* nm[10] = {"crt.stage", "crt.aload", "crt.mma", "crt.carry", "crt.store",
                                      "fh.load", "fh.fwd", "fh.pw", "fh.inv", "fh.store"};
                double c = 0, fsum = 0;
                for (int i = 0; i < 5; ++i) c += double(h[i]);
                for (int i = 5; i < 10; ++i) fsum += double(h[i]);
                std::fprintf(stderr, "[PROF] crt Gcyc=%.2f:", c / 1e9);
                for (int i = 0; i < 5; ++i)
                    std::fprintf(stderr, " %s %.1f%%", nm[i], c > 0 ? 100.0 * double(h[i]) / c : 0.0);
                std::fprintf(stderr, " | fh Gcyc=%.2f:", fsum / 1e9);
                for (int i = 5; i < 10; ++i)
                    std::fprintf(stderr, " %s %.1f%%", nm[i], fsum > 0 ? 100.0 * double(h[i]) / fsum : 0.0);
                {
                    const char* en[6] = {"setup", "mma", "epi", "dit", "store", "aload"};
                    double es = 0;
                    for (int i = 10; i < 16; ++i) es += double(h[i]);
                    std::fprintf(stderr, " | efl Gcyc=%.2f:", es / 1e9);
                    for (int i = 10; i < 16; ++i)
                        std::fprintf(stderr, " %s %.1f%%", en[i - 10], es > 0 ? 100.0 * double(h[i]) / es : 0.0);
                }
                std::fprintf(stderr, "\n");
                unsigned long long z[16] = {};
                cudaMemcpyToSymbol(tc::g_tcprof, z, sizeof(z));
            } else {
                cudaGetLastError();
            }
        }
    }

    cudaError_t tc_pipeline(const u32* pa, const u32* pb, u32* pc, u32* pra, u32* prb, u32* prc,
                            u32 B, cudaStream_t stream, bool prof, int part = 0) {
        // part 0: whole pipeline, 1: encode (+fwd low) only, 2: tail only
        const unsigned n = plan_.N, l = plan_.L, k = plan_.K, logN = plan_.logN;
        cudaError_t st = cudaSuccess;
            const unsigned logH = logN - 9;
            // TC29: 256-row encode tiles + low stage 8 in fused_high (needs the
            // OC12 radix-16 load path in fused_high, i.e. logH >= 4)
            const bool ef256 = TC29_EF256 && TC22_MERGED && tc_efl && logH >= 4;
            if (part != 2) {
            if (tc_efl && ef256) {
                tc::encode_fwdlow256<<<dim3(n / tc::E2_ROWS, 2, B), 256, 0, stream>>>(
                    pa, pb, pra, prb, d_fenc, d_primes, d_weights,
                    d_pk_fwd, n, l, k, logN, enc_KS, enc_NT);
                if ((st = cudaGetLastError()) != cudaSuccess) return st;
                if (prof) { cudaEventRecord(ev_[1], stream); cudaEventRecord(ev_[2], stream);
                                  cudaEventRecord(ev_[3], stream); }
            } else if (tc_efl) {
                tc::encode_fwdlow<<<dim3(n / tc::EF_ROWS, 2, B), 512, 0, stream>>>(
                    pa, pb, pra, prb, d_fenc, d_primes, d_weights, TC22_MERGED ? nullptr : d_reversed_twist,
                    d_pk_fwd, n, l, k, logN, enc_KS, enc_NT);
                if ((st = cudaGetLastError()) != cudaSuccess) return st;
                if (prof) { cudaEventRecord(ev_[1], stream); cudaEventRecord(ev_[2], stream);
                                  cudaEventRecord(ev_[3], stream); }
            } else {
            tc::encode_tc<<<dim3((n + tc::ENC_ROWS - 1) / tc::ENC_ROWS, 2, B), 256, 0, stream>>>(
                pa, pb, pra, prb, d_fenc, d_primes, d_weights, TC22_MERGED ? nullptr : d_reversed_twist,
                n, l, k, logN, enc_KS, enc_NT);
            if ((st = cudaGetLastError()) != cudaSuccess) return st;
            if (prof) { cudaEventRecord(ev_[1], stream); cudaEventRecord(ev_[2], stream); }
            tc::fwd_low<<<dim3(n / tc::NT_TILE, k, B), 256, 0, stream>>>(
                pra, prb, d_primes, d_pk_fwd, n, k);
            if ((st = cudaGetLastError()) != cudaSuccess) return st;
            if (prof) cudaEventRecord(ev_[3], stream);
            }
            }
            if (part == 1) return cudaGetLastError();
            tc::fused_high<<<dim3(tc::NT_TILE / tc::NT_COLS, k, B), dim3(tc::NT_COLS, 8),
                             2 * (n / tc::NT_TILE) * (tc::NT_COLS + (TC31_S32 ? 0 : 1)) * sizeof(u32), stream>>>(
                pra, prb, prc, d_primes, d_twf, d_twi, n, k, logH, d_pk_fwd, ef256 ? 1u : 0u);
            if ((st = cudaGetLastError()) != cudaSuccess) return st;
            if (prof) cudaEventRecord(ev_[4], stream);
            if (tc_invcrt) {
                tc::CrtTc c2 = crt_cfg;
                c2.SA = crt2_SA;
                const unsigned R = invcrt_rows, P = tc::invcrt_P(k, R);
                cudaLaunchConfig_t cf = {};
                cudaLaunchAttribute at[1];
                at[0].id = cudaLaunchAttributeClusterDimension;
                at[0].val.clusterDim.x = tc::NT_TILE / R; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
                cf.gridDim = dim3(n / R, 1, B); cf.blockDim = dim3(R * 8, 1, 1);
                cf.dynamicSmemBytes = invcrt_shared; cf.stream = stream; cf.attrs = at; cf.numAttrs = 1;
                const uint2* fr = reinterpret_cast<const uint2*>(d_fcrt);
                st = (R == 32) ? cudaLaunchKernelEx(&cf, tc::inv_crt<32>, crt_args_, (const u32*)prc, pc, fr, c2,
                                                    (const u32*)d_pk_inv, logN, P)
                               : cudaLaunchKernelEx(&cf, tc::inv_crt<64>, crt_args_, (const u32*)prc, pc, fr, c2,
                                                    (const u32*)d_pk_inv, logN, P);
                if (st == cudaSuccess) st = cudaGetLastError();
                if (prof) for (int e = 5; e <= 7; ++e) cudaEventRecord(ev_[e], stream);
                return st;
            }
            tc::inv_low<<<dim3(n / tc::NT_TILE, k, B), 256, 0, stream>>>(
                prc, d_primes, d_pk_inv, n, k);
            if ((st = cudaGetLastError()) != cudaSuccess) return st;
            if (prof) { cudaEventRecord(ev_[5], stream); cudaEventRecord(ev_[6], stream); }
            if (tc_crt2) {
                tc::CrtTc c2 = crt_cfg;
                c2.SA = crt2_SA;
                tc::crt_tc2<<<dim3((n + tc::CRT2_ROWS - 1) / tc::CRT2_ROWS, 1, B), 128,
                              crt2_shared, stream>>>(crt_args_, prc, pc, d_fcrt, c2,
                                                     d_primes, d_untw_slot, logN, d_corr_m);
            } else
            tc::crt_tc<<<dim3((n + tc::CRT_ROWS - 1) / tc::CRT_ROWS, 1, B), 128,
                          crt_tc_shared, stream>>>(crt_args_, prc, pc, d_fcrt, crt_cfg,
                                                   d_primes, d_untw_slot, logN, d_corr_m);
            st = cudaGetLastError();
            if (prof) cudaEventRecord(ev_[7], stream);
            return st;
    }

private:
    void build_tc(const cudaDeviceProp& prop) {
        if (prop.major < 8) return;
        std::vector<u32> ps;
        for (const Prime& pr : plan_.primes) ps.push_back(pr.p);
        if (plan_.L <= 8 * tc::ENC_KS_MAX) {
            std::vector<u32> f = tc::bake_encode(ps, plan_.L, enc_KS, enc_NT);
            upload_words(d_fenc, f);
            tc_enc_ready = d_fenc != nullptr;
        }
        const unsigned pitch = plan_.K + 1;
        if (pitch <= 8 * tc::CRT_KS_MAX) {
            unsigned ks = 0, nt = 0;
            std::vector<u32> f = tc::bake_crt(plan_.crt_weights, plan_.L, pitch, ks, nt);
            crt_cfg.KS = ks;
            crt_cfg.NT = nt;
            crt_cfg.SA = tc::crt_sa_stride(ks);
            crt_cfg.SX = 2 * nt + 1;
            crt_cfg.DS = 2 * crt_cfg.SX;
            {
                std::vector<u32> cm(plan_.K);
                for (unsigned kk = 0; kk < plan_.K; ++kk)
                    cm[kk] = u32((u64(1) << 62) / plan_.primes[kk].p);
                upload_u32(d_corr_m, cm);
                if (plan_.K <= tc::CRT_KMAX_C) {
                    if (cudaMemcpyToSymbol(tc::c_crt_corr, cm.data(), cm.size() * sizeof(u32)) != cudaSuccess ||
                        cudaMemcpyToSymbol(tc::c_crt_prime, plan_.primes.data(), plan_.K * sizeof(Prime)) != cudaSuccess) {
                        cudaGetLastError();
                        return;
                    }
                }
                if (!d_corr_m) return;
            }
            crt_tc_shared = tc::crt_tc_smem(crt_cfg);
            {
                unsigned s2 = ks * 8 + 8;
                while (s2 % 32 != 4 && s2 % 32 != 28) ++s2;
                crt2_SA = s2;
                crt2_shared = tc::crt_tc2_smem(crt2_SA, crt_cfg.SX);
                // TC27: prefer 32 rows / cluster 16 (non-portable), else 64 / 8
                invcrt_shared = 0;
                {
                    const std::size_t s32 = tc::invcrt_smem(32, crt2_SA, crt_cfg.SX);
                    bool ok = cudaFuncSetAttribute(tc::inv_crt<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(s32)) == cudaSuccess &&
                              cudaFuncSetAttribute(tc::inv_crt<32>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess;
                    if (ok) {
                        cudaLaunchConfig_t cf = {};
                        cudaLaunchAttribute at[1];
                        at[0].id = cudaLaunchAttributeClusterDimension;
                        at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
                        cf.gridDim = dim3(plan_.N / 32, 1, 1); cf.blockDim = dim3(256, 1, 1);
                        cf.dynamicSmemBytes = s32; cf.attrs = at; cf.numAttrs = 1;
                        int ncl = 0;
                        ok = cudaOccupancyMaxActiveClusters(&ncl, tc::inv_crt<32>, &cf) == cudaSuccess && ncl > 0;
                    }
                    cudaGetLastError();
                    if (ok) { invcrt_rows = 32; invcrt_shared = s32; }
                    else {
                        const std::size_t s64 = tc::invcrt_smem(64, crt2_SA, crt_cfg.SX);
                        if (cudaFuncSetAttribute(tc::inv_crt<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(s64)) == cudaSuccess) {
                            invcrt_rows = 64; invcrt_shared = s64;
                        }
                        cudaGetLastError();
                    }
                }
            }
            if (crt_tc_shared > 48 * 1024) {
                if (cudaFuncSetAttribute(tc::crt_tc, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         int(crt_tc_shared)) != cudaSuccess) {
                    cudaGetLastError();
                    return;
                }
            }
            upload_words(d_fcrt, f);
            tc_crt_ready = d_fcrt != nullptr;
            // fused decode table: decode(r) = r*u*R^-2, term = decode(r)*inv
            // -> c = mont(u, inv) = u*inv*R^-1, and mont(r, c) = r*u*inv*R^-2.
            {
                const unsigned K = plan_.K, N = plan_.N, logN = plan_.logN;
                std::vector<u32> t(std::size_t(K) * N), ts(std::size_t(K) * N);
                std::vector<u32> slotc(K, 0u);
                for (unsigned kk = 0; kk < K; ++kk) {
                    const Prime& pr = plan_.primes[kk];
                    const u32 inv_m = plan_.crt_primes[kk].inv;
                    for (unsigned i = 0; i < N; ++i)
                        t[std::size_t(kk) * N + i] =
                            mont(plan_.untwist[std::size_t(kk) * N + i], inv_m, pr);
#if TC22_MERGED
                    // merged NTT leaves n * x * R in slot order: term = mont(r, n^-1 * inv)
                    const u32 ninv = host_powmod(N % pr.p, pr.p - 2, pr.p);
                    const u32 cst = u32(u64(ninv) * inv_m % pr.p);
                    for (unsigned i = 0; i < N; ++i) ts[std::size_t(kk) * N + i] = cst;
                    slotc[kk] = cst;
#else
                    for (unsigned i = 0; i < N; ++i)
                        ts[std::size_t(kk) * N + i] =
                            t[std::size_t(kk) * N + reverse_bits(i, logN)];
#endif
                }
                upload_u32(d_untw_inv, t);
#if TC22_MERGED
                if (K <= tc::CRT_KMAX_C &&
                    cudaMemcpyToSymbol(tc::c_crt_slotc, slotc.data(), K * sizeof(u32)) != cudaSuccess)
                    cudaGetLastError();
#endif
                if (N >= 1024 && N % tc::NT_TILE == 0 && N / tc::NT_TILE <= 128 && N / tc::NT_TILE >= 2) {
                    upload_u32(d_untw_slot, ts);
                    build_ntt_tables();
                }
            }
        }
        std::fprintf(stderr, "[TC] enc ready=%d KS=%u NT=%u | crt ready=%d KS=%u NT=%u SA=%u smem=%zu\n",
                     int(tc_enc_ready), enc_KS, enc_NT, int(tc_crt_ready), crt_cfg.KS, crt_cfg.NT,
                     crt_cfg.SA, crt_tc_shared);
    }
    static void upload_u32(u32*& dst, const std::vector<u32>& src) {
        void* p = nullptr;
        if (cudaMalloc(&p, src.size() * sizeof(u32)) != cudaSuccess) { cudaGetLastError(); return; }
        if (cudaMemcpy(p, src.data(), src.size() * sizeof(u32), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(p);
            cudaGetLastError();
            return;
        }
        dst = static_cast<u32*>(p);
    }
    void build_ntt_tables() {
        const unsigned K = plan_.K, N = plan_.N, T = tc::NT_TILE;
        const unsigned HC = N / T;
        unsigned logH = 0;
        while ((1u << logH) < HC) ++logH;
#if TC22_MERGED
        // merged-psi twiddles (tools/tc_emul/merged_emul.cpp):
        //   low  (half=2^m, j<half):  psi^(2^(n-1-m)(1+2j))           at (half-1)+j
        //   high (stage s):  A_s[low] = psi^(2^(n-10-s)(1+2low)),  B_s[w] = psi^(w 2^(n-s))
        // inverse = the same exponents of psi^-1.  psi^e (Montgomery) = twist[k][e].
        const unsigned logN_ = plan_.logN;
        std::vector<u32> pf(std::size_t(K) * TC_LT, 0u), pi(pf.size(), 0u);
        const std::size_t TWS = 7 * T + 127;
        std::vector<u32> tf(std::size_t(K) * TWS, 0u), ti(tf.size(), 0u);
        for (unsigned kk = 0; kk < K; ++kk) {
            const u32* tw = plan_.twist.data() + std::size_t(kk) * N;
            const u32 pp = plan_.primes[kk].p;
            auto fpsi = [&](std::size_t e) { return tw[e]; };
            auto ipsi = [&](std::size_t e) { return e == 0 ? tw[0] : u32(pp - tw[N - e]); };
            for (unsigned m = 0; m < 9; ++m) {
                const unsigned half = 1u << m;
                for (unsigned j = 0; j < half; ++j) {
                    const std::size_t e = (std::size_t(1) << (logN_ - 1 - m)) * (1 + 2 * j);
                    pf[std::size_t(kk) * TC_LT + (half - 1) + j] = fpsi(e);
                    pi[std::size_t(kk) * TC_LT + (half - 1) + j] = ipsi(e);
                }
            }
            for (unsigned st = 0; st < logH; ++st) {
                const unsigned H = 1u << st;
                for (unsigned low = 0; low < T; ++low) {
                    const std::size_t e = (std::size_t(1) << (logN_ - 10 - st)) * (1 + 2 * low);
                    tf[std::size_t(kk) * TWS + st * T + low] = fpsi(e);
                    ti[std::size_t(kk) * TWS + st * T + low] = ipsi(e);
                }
                for (unsigned w = 0; w < H; ++w) {
                    const std::size_t e = std::size_t(w) << (logN_ - st);
                    tf[std::size_t(kk) * TWS + 7 * T + (H - 1) + w] = fpsi(e);
                    ti[std::size_t(kk) * TWS + 7 * T + (H - 1) + w] = ipsi(e);
                }
            }
        }
#else
        std::vector<u32> pf(std::size_t(K) * (T / 2)), pi(pf.size());
        const std::size_t TWS = 7 * T + 127;
        std::vector<u32> tf(std::size_t(K) * TWS, 0u), ti(tf.size(), 0u);
        for (unsigned kk = 0; kk < K; ++kk) {
            const u32* r = plan_.roots.data() + std::size_t(kk) * (N / 2);
            const u32* ri = plan_.inverse_roots.data() + std::size_t(kk) * (N / 2);
            for (unsigned p = 0; p < T / 2; ++p) {
                pf[std::size_t(kk) * (T / 2) + p] = r[std::size_t(p) * HC];
                pi[std::size_t(kk) * (T / 2) + p] = ri[std::size_t(p) * HC];
            }
            for (unsigned st = 0; st < logH; ++st) {
                const unsigned H = 1u << st, d = logH - st - 1;
                for (unsigned low = 0; low < T; ++low) {
                    tf[std::size_t(kk) * TWS + st * T + low] = r[std::size_t(low) << d];
                    ti[std::size_t(kk) * TWS + st * T + low] = ri[std::size_t(low) << d];
                }
                for (unsigned w = 0; w < H; ++w) {
                    tf[std::size_t(kk) * TWS + 7 * T + (H - 1) + w] = r[std::size_t(w * T) << d];
                    ti[std::size_t(kk) * TWS + 7 * T + (H - 1) + w] = ri[std::size_t(w * T) << d];
                }
            }
        }
#endif
        upload_u32(d_pk_fwd, pf);
        upload_u32(d_pk_inv, pi);
        upload_u32(d_twf, tf);
        upload_u32(d_twi, ti);
        tc_ntt_ready = d_pk_fwd && d_pk_inv && d_twf && d_twi && d_untw_slot;
    }
    static void upload_words(uint2*& dst, const std::vector<u32>& src) {
        void* p = nullptr;
        if (cudaMalloc(&p, src.size() * sizeof(u32)) != cudaSuccess) { cudaGetLastError(); return; }
        if (cudaMemcpy(p, src.data(), src.size() * sizeof(u32), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(p);
            cudaGetLastError();
            return;
        }
        dst = static_cast<uint2*>(p);
    }
    static void* raw_alloc(std::size_t bytes) {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess)
            throw std::runtime_error(std::string("gpu31: cudaMalloc failed: ") +
                                     cudaGetErrorString(cudaGetLastError()));
        return p;
    }
    template <class T>
    static void upload(T*& dst, const std::vector<T>& src) {
        const std::size_t bytes = src.size() * sizeof(T);
        dst = static_cast<T*>(raw_alloc(bytes));
        if (cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error(std::string("gpu31: table upload failed: ") +
                                     cudaGetErrorString(cudaGetLastError()));
    }
    void release() noexcept {
        if (d_primes) cudaFree(d_primes);
        if (d_weights) cudaFree(d_weights);
        if (d_roots) cudaFree(d_roots);
        if (d_inv_roots) cudaFree(d_inv_roots);
        if (d_twist) cudaFree(d_twist);
        if (d_untwist) cudaFree(d_untwist);
        if (d_reversed_twist) cudaFree(d_reversed_twist);
        if (d_crt_primes) cudaFree(d_crt_primes);
        if (d_crt_weights) cudaFree(d_crt_weights);
        if (d_crt_q) cudaFree(d_crt_q);
        if (d_ra) cudaFree(d_ra);
        if (d_rb) cudaFree(d_rb);
        if (d_rc) cudaFree(d_rc);
        if (d_fenc) cudaFree(d_fenc);
        if (d_fcrt) cudaFree(d_fcrt);
        if (d_untw_inv) cudaFree(d_untw_inv);
        for (int q = 0; q < 2; ++q) {
            if (cs_[q]) cudaStreamDestroy(cs_[q]);
            if (evj_[q]) cudaEventDestroy(evj_[q]);
            cs_[q] = nullptr;
            evj_[q] = nullptr;
        }
        if (evf_) cudaEventDestroy(evf_);
        evf_ = nullptr;
        if (d_corr_m) cudaFree(d_corr_m);
        d_corr_m = nullptr;
        d_untw_inv = nullptr;
        for (u32** pp : {&d_pk_fwd, &d_pk_inv, &d_twf, &d_twi, &d_untw_slot}) {
            if (*pp) cudaFree(*pp);
            *pp = nullptr;
        }
        d_fenc = d_fcrt = nullptr;
        for (int i = 0; i <= kStages; ++i)
            if (ev_[i]) cudaEventDestroy(ev_[i]);
        d_primes = nullptr;
        d_weights = nullptr;
        d_roots = nullptr;
        d_inv_roots = nullptr;
        d_twist = nullptr;
        d_untwist = nullptr;
        d_reversed_twist = nullptr;
        d_crt_primes = nullptr;
        d_crt_weights = nullptr;
        d_crt_q = nullptr;
        d_ra = nullptr;
        d_rb = nullptr;
        d_rc = nullptr;
    }

    template <bool Two>
    cudaError_t transform(u32* a, u32* b, const u32* roots,
                          unsigned n, unsigned k, u32 batches, cudaStream_t stream) const {
        const unsigned tile = n < max_tile ? n : max_tile;
        constexpr unsigned arrays = Two ? 2 : 1;
        low_stages<Two><<<dim3(n / tile, k, batches), tile / 2,
                          arrays * tile * sizeof(u32), stream>>>(
            a, b, d_primes, roots, n, k, tile);
        cudaError_t st = cudaGetLastError();
        if (st != cudaSuccess || n == tile) return st;
        high_stages<Two><<<dim3(max_tile / low_columns, k, batches),
                           dim3(low_columns, high_workers),
                           arrays * (n / tile) * low_columns * sizeof(u32), stream>>>(
            a, b, d_primes, roots, n, k);
        return cudaGetLastError();
    }
};

}  // namespace gpu31

#undef GPU31_FHD
