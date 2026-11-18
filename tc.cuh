// tc.cuh — tensor-core (IMMA u8) encode and CRT for the gpu31 pipeline.
//
// Both conversions are integer GEMMs with byte-split operands:
//
//   encode:  V_k = sum_j byte_j(X) * w_{k,j},  w_{k,j} = 2^(8j+32) mod p_k
//            A[row=coef][j]      = byte j of the coefficient (raw limb memory)
//            B[n=(k,s)][j]       = byte s of w_{k,j}
//            V_k = sum_s S_{k,s} 2^(8s)   (< 2^50), then mod p_k
//
//   CRT:     X = sum_j term_j * W_j   (W_j = gpu31 crt_weights column j)
//            A[row=coef][(j,s)]  = byte s of term_j
//            B[n=d][(j,s)]       = byte (d-s) of W_j      (diagonal layout)
//            X = sum_d S_d 2^(8d), carried, then gpu31::reduce_sum.
//
// mma.sync.m16n8k32.row.col.s32.u8.u8.s32 fragment layout (PTX ISA), with
// g = lane>>2, t = lane&3:
//   A regs: a0 = (row g,   k t*4..+3)  a1 = (row g+8, k t*4..+3)
//           a2 = (row g,   k 16+t*4..) a3 = (row g+8, k 16+t*4..)
//   B regs: b0 = (k t*4..+3, col g)    b1 = (k 16+t*4.., col g)
//   C regs: c0,c1 = (row g, col 2t,2t+1)  c2,c3 = (row g+8, col 2t,2t+1)
// With rows stored as little-endian bytes, every A/B register is one aligned
// 32-bit word, so no packing happens on the device.  B is baked on the host in
// [ntile][kstep][lane] order (uint2 per lane): one coalesced LDG.64 per mma.
#pragma once

#include <cstdint>
#include <vector>

// TC10 (parallel branch): warp-shuffle for the first 5 DIT stages of
// encode_fwdlow.  Set to 0 for A/B against the original shared-only path.
#ifndef TC10_SHUF
#define TC10_SHUF 0   // OC2 measured slower: encode 24.4 -> 26.2 ms
#endif

namespace tc {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr unsigned ENC_KS_MAX = 7;    // L <= 56
constexpr unsigned CRT_KS_MAX = 15;   // K+1 <= 120
constexpr unsigned ENC_ROWS = 128;    // 4 warps x 32 rows
constexpr unsigned ENC_S = 68;        // smem words per row: >= 56, == 4 (mod 32)
constexpr unsigned CRT_ROWS = 64;     // 4 warps x 16 rows
constexpr unsigned CRT2_ROWS = 32;    // TC15: 4 warps split n-tiles
inline std::size_t crt_tc2_smem(unsigned SA, unsigned SX) {
    return ((std::size_t(CRT2_ROWS) * SA * 4 + 15) & ~std::size_t(15)) + std::size_t(CRT2_ROWS) * SX * 8;
}


// ---------------------------------------------------------------------------
// host baking
// ---------------------------------------------------------------------------
inline u32 pack_bytes(const unsigned char b[4]) {
    return u32(b[0]) | (u32(b[1]) << 8) | (u32(b[2]) << 16) | (u32(b[3]) << 24);
}

// primes: K values. Returns frag of NT*KS*32*2 words.
inline std::vector<u32> bake_encode(const std::vector<u32>& primes, unsigned L,
                                    unsigned& KS, unsigned& NT) {
    const unsigned K = unsigned(primes.size());
    KS = (L + 7) / 8;
    NT = (K * 4 + 7) / 8;
    const unsigned JB = 4 * L;   // bytes per coefficient
    std::vector<u32> w(std::size_t(K) * JB);
    for (unsigned k = 0; k < K; ++k) {
        const u64 p = primes[k];
        u64 f = (u64(1) << 32) % p;              // 2^32 mod p
        for (unsigned j = 0; j < JB; ++j) {
            w[std::size_t(k) * JB + j] = u32(f);
            f = (f << 8) % p;
        }
    }
    std::vector<u32> frag(std::size_t(NT) * KS * 32 * 2, 0);
    for (unsigned nt = 0; nt < NT; ++nt)
        for (unsigned ks = 0; ks < KS; ++ks)
            for (unsigned lane = 0; lane < 32; ++lane) {
                const unsigned g = lane >> 2, t = lane & 3;
                const unsigned n = nt * 8 + g, k = n / 4, s = n % 4;
                for (unsigned r = 0; r < 2; ++r) {
                    unsigned char b[4];
                    for (unsigned i = 0; i < 4; ++i) {
                        const unsigned j = ks * 32 + r * 16 + t * 4 + i;
                        b[i] = (k < K && j < JB)
                                   ? (unsigned char)((w[std::size_t(k) * JB + j] >> (8 * s)) & 255)
                                   : 0;
                    }
                    frag[((std::size_t(nt) * KS + ks) * 32 + lane) * 2 + r] = pack_bytes(b);
                }
            }
    return frag;
}

// weights: L x pitch (limb-major, gpu31::Plan::crt_weights), pitch = K+1.
inline std::vector<u32> bake_crt(const std::vector<u32>& weights, unsigned L, unsigned pitch,
                                 unsigned& KS, unsigned& NT) {
    KS = (pitch + 7) / 8;
    NT = (4 * L + 3 + 7) / 8;
    auto wbyte = [&](unsigned j, unsigned e) -> unsigned {
        const unsigned limb = e / 4, sh = 8 * (e % 4);
        return (weights[std::size_t(limb) * pitch + j] >> sh) & 255;
    };
    std::vector<u32> frag(std::size_t(NT) * KS * 32 * 2, 0);
    for (unsigned nt = 0; nt < NT; ++nt)
        for (unsigned ks = 0; ks < KS; ++ks)
            for (unsigned lane = 0; lane < 32; ++lane) {
                const unsigned g = lane >> 2, t = lane & 3;
                const unsigned d = nt * 8 + g;
                for (unsigned r = 0; r < 2; ++r) {
                    unsigned char b[4];
                    for (unsigned i = 0; i < 4; ++i) {
                        const unsigned kk = ks * 32 + r * 16 + t * 4 + i;
                        const unsigned j = kk / 4, s = kk % 4;
                        const int e = int(d) - int(s);
                        b[i] = (j < pitch && e >= 0 && unsigned(e) < 4 * L)
                                   ? (unsigned char)wbyte(j, unsigned(e))
                                   : 0;
                    }
                    frag[((std::size_t(nt) * KS + ks) * 32 + lane) * 2 + r] = pack_bytes(b);
                }
            }
    return frag;
}

inline unsigned crt_sa_stride(unsigned KS) {    // >= KS*8+4 words, == +-4 (mod 32)
    unsigned s = KS * 8 + 4;
    while (s % 32 != 4 && s % 32 != 28) ++s;
    return s;
}

#if defined(__CUDACC__)

// Phase profiling: thread (0,0) of every CTA adds clock64 deltas per phase.
// Slots: CRT 0 staging, 1 A-load, 2 mma+epilogue, 3 carry/reduce, 4 store;
//        fused_high 5 load, 6 fwd, 7 pointwise, 8 inverse, 9 store;
//        encode_fwdlow 10 setup, 11 mma, 12 epilogue, 13 DIT, 14 store, 15 A-load.
#ifndef TC19_PF
#define TC19_PF 1
#endif
#ifndef TC20_R8
#define TC20_R8 1
#endif
#ifndef TC16_R4
#define TC16_R4 1
#endif
#ifndef TC12_R4
#define TC12_R4 1
#endif

// OC8 (parallel branch): radix-8 high stages in fused_high - three DIT/DIF
// stages per shared round trip (s, s+1, s+2), plus radix-2 tail stages.
// CPU-validated bit-exact against the radix-2 reference (20 random trials).
#ifndef OC8_R8
#define OC8_R8 1
#endif

// OC9 (parallel branch): Shoup multiply with a hoisted twiddle constant in
// fused_high butterflies.  mu[p] = floor((2^32-p)*2^32/p) is uploaded per
// prime; w_shoup = w + umulhi(w,mu) is computed once per twiddle and reused
// by every butterfly that consumes it.  CPU-validated bit-exact (verified by
// verify_shoup_mul.py, 8 primes x 20000 trials + edge cases).
#ifndef OC9_SHOUP
#define OC9_SHOUP 1
#endif
#ifndef TC_PROF
#define TC_PROF 1
#endif
__device__ unsigned long long g_tcprof[16];
constexpr unsigned CRT_KMAX_C = 128;
__constant__ gpu31::Prime c_crt_prime[CRT_KMAX_C];
__constant__ u32 c_crt_corr[CRT_KMAX_C];
#define TCP_T(v) const long long v = (TC_PROF && threadIdx.x == 0 && threadIdx.y == 0) ? (long long)clock64() : 0
#define TCP_ADD(slot, a, b) do { if (TC_PROF && threadIdx.x == 0 && threadIdx.y == 0) \
    atomicAdd(&g_tcprof[slot], (unsigned long long)((b) - (a))); } while (0)

#if OC9_SHOUP
// One Shoup reduction with a pre-hoisted constant for twiddle w.
__device__ __forceinline__ u32 shoup_mul(u32 x, u32 w, u32 mu, const gpu31::Prime& p) {
    const u32 ws = w + __umulhi(w, mu);
    const u32 q = __umulhi(x, ws);
    const u64 r = u64(x) * w - u64(q) * p.p;   // 0 <= r < 2p (mu is floored)
    const u32 v = u32(r);
    return v >= p.p ? v - p.p : v;
}
__device__ __forceinline__ void shoup_butterfly(u32& a, u32& b, u32 w, u32 mu,
                                                const gpu31::Prime& p) {
    const u32 u = a, v = shoup_mul(b, w, mu, p);
    a = gpu31::add(u, v, p.p);
    b = gpu31::sub(u, v, p.p);
}
#endif

__device__ __forceinline__ void mma_u8(int (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3,
                                       u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// encode: raw (un-reindexed) coefficients -> twisted Montgomery residues,
// exactly what gpu31 reindex + encode_tiled produce.
// grid (ceil(n/128), 2 operands, B), block 128, smem ENC_ROWS*ENC_S*4.
// ---------------------------------------------------------------------------
constexpr unsigned ENC_NTB = 8;   // independent accumulator chains per warp
__global__ void __launch_bounds__(256, 2)
encode_tc(const u32* __restrict__ src_a, const u32* __restrict__ src_b,
          u32* __restrict__ ra, u32* __restrict__ rb,
          const uint2* __restrict__ frag, const gpu31::Prime* __restrict__ primes,
          const gpu31::Weight* __restrict__ weights, const u32* __restrict__ rtwist,
          unsigned n, unsigned l, unsigned K, unsigned logN, unsigned KS, unsigned NT) {
    __shared__ u32 sA[ENC_ROWS * ENC_S];
    const u32* src = blockIdx.y ? src_b : src_a;
    u32* dst = blockIdx.y ? rb : ra;
    const unsigned pair = blockIdx.z;
    const unsigned start = blockIdx.x * ENC_ROWS;
    const u64 base = u64(pair) * n * l;
    const unsigned W = KS * 8;

    for (unsigned idx = threadIdx.x; idx < ENC_ROWS * W; idx += blockDim.x) {
        const unsigned r = idx / W, c = idx - r * W, i = start + r;
        u32 v = 0;
        if (i < n && c < l) {
            const unsigned src_i = logN ? (__brev(i) >> (32 - logN)) : i;
            v = src[base + u64(src_i) * l + c];
        }
        sA[r * ENC_S + c] = v;
    }
    __syncthreads();

    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned rowbase = warp * 16;

    u32 a[ENC_KS_MAX][4];
#pragma unroll
    for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
        const u32* r0 = sA + (rowbase + g) * ENC_S + ks * 8 + t;
        const u32* r1 = r0 + 8 * ENC_S;
        const bool on = ks < KS;
        a[ks][0] = on ? r0[0] : 0;
        a[ks][1] = on ? r1[0] : 0;
        a[ks][2] = on ? r0[4] : 0;
        a[ks][3] = on ? r1[4] : 0;
    }

    const bool odd = t & 1;
    const unsigned sh = (t & 1) * 16;
    const unsigned i = start + rowbase + g + (odd ? 8 : 0);
    const bool rowok = i < n;
    const u64 rbase = u64(pair) * K * n;

    for (unsigned nt0 = 0; nt0 < NT; nt0 += ENC_NTB) {
        int acc[ENC_NTB][4];
        u32 tw[ENC_NTB];
#pragma unroll
        for (unsigned j = 0; j < ENC_NTB; ++j) {
            acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
            const unsigned k = (nt0 + j) * 2 + (t >> 1);
            tw[j] = (k < K && rowok) ? rtwist[u64(k) * n + i] : 0u;
        }
#pragma unroll
        for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
            if (ks < KS) {
                uint2 b[ENC_NTB];
#pragma unroll
                for (unsigned j = 0; j < ENC_NTB; ++j) {
                    const unsigned nt = nt0 + j;
                    if (nt < NT) b[j] = frag[(std::size_t(nt) * KS + ks) * 32 + lane];
                    else b[j] = make_uint2(0u, 0u);
                }
#pragma unroll
                for (unsigned j = 0; j < ENC_NTB; ++j)
                    if (nt0 + j < NT)
                        mma_u8(acc[j], a[ks][0], a[ks][1], a[ks][2], a[ks][3], b[j].x, b[j].y);
            }
        }
#pragma unroll
        for (unsigned j = 0; j < ENC_NTB; ++j) {
            const u64 pg = (u64(u32(acc[j][0])) << sh) + (u64(u32(acc[j][1])) << (sh + 8));
            const u64 pg8 = (u64(u32(acc[j][2])) << sh) + (u64(u32(acc[j][3])) << (sh + 8));
            const u64 send = odd ? pg : pg8;
            const u64 mine = odd ? pg8 : pg;
            const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
            const unsigned k = (nt0 + j) * 2 + (t >> 1);
            if (k < K && rowok) {
                const gpu31::Prime p = primes[k];
                const gpu31::Weight w0 = weights[std::size_t(k) * l];
                const u32 lo = gpu31::canonical_word(u32(tot), p.p);
                u32 hi = gpu31::lazy_product(u32(tot >> 32), w0, p.p);
                if (hi >= p.p) hi -= p.p;
                const u32 v = gpu31::add(lo, hi, p.p);
                dst[rbase + u64(k) * n + i] = gpu31::mont(v, tw[j], p);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CRT: plain residues [pair][k][n] -> canonical [pair][n][l] limbs, exactly as
// gpu31::reconstruct_kernel.  grid (ceil(n/64), 1, B), block 128.
// smem: sA (CRT_ROWS*SA u32, later reused as sX u64[CRT_ROWS*SX]) +
//       digits (CRT_ROWS*DS u32) + red (2*CRT_ROWS double).
// ---------------------------------------------------------------------------
struct CrtTc {
    unsigned KS, NT, SA, SX, DS;
};

inline std::size_t crt_tc_smem(const CrtTc& c) {
    std::size_t a = std::size_t(CRT_ROWS) * c.SA * 4;
    const std::size_t x = std::size_t(CRT_ROWS) * c.SX * 8;
    return a > x ? a : x;
}

constexpr unsigned CRT_NTB = 7;   // independent accumulator chains per warp
__global__ void __launch_bounds__(128, 3)
crt_tc(gpu31::CrtArgs pl, const u32* __restrict__ residues, u32* __restrict__ out,
       const uint2* __restrict__ frag, CrtTc cfg,
       const gpu31::Prime* __restrict__ primes, const u32* __restrict__ untw_inv,
       unsigned slot_logN, const u32* __restrict__ corr_m) {
    // slot_logN != 0: residues are in bit-reversed slot order (DIF inverse);
    // row `slot` holds coefficient rev(slot).  untw_inv is then slot-ordered.
    // Shared: one region, used as sA (u32, stride SA) -> sX (u64, stride SX)
    // -> digits (u32, in place over sX, stride 2*SX).
    extern __shared__ __align__(16) unsigned char smem_raw[];
    u32* sA = reinterpret_cast<u32*>(smem_raw);
    u64* sX = reinterpret_cast<u64*>(smem_raw);

    const unsigned n = pl.n, l = pl.limbs, K = pl.count;
    const unsigned pair = blockIdx.z, start = blockIdx.x * CRT_ROWS;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid & 63, jg = tid >> 6;
    const unsigned i = start + c;
    const bool active = i < n;
    const u64 rbase = u64(pair) * K * n;
    const unsigned W = cfg.KS * 8;

    TCP_T(t0);
    // 1. terms + fixed-point correction sum (term/p in 2^-32 units)
    u64 fsum = 0;
    if (untw_inv && K <= CRT_KMAX_C) {
        // TC11: batched loads (8 independent residue+untwist pairs in flight),
        // prime and correction constants from __constant__ (warp-uniform j).
#pragma unroll 1
        for (unsigned j0 = jg; j0 < K; j0 += 16) {
            u32 r[8], u[8];
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + 2 * q;
                r[q] = (j < K && active) ? residues[rbase + u64(j) * n + i] : 0u;
                u[q] = (j < K && active) ? untw_inv[u64(j) * n + i] : 0u;
            }
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + 2 * q;
                if (j < K) {
                    const u32 term = gpu31::mont(r[q], u[q], c_crt_prime[j]);
                    sA[c * cfg.SA + j] = term;
                    fsum += (u64(term) * c_crt_corr[j]) >> 30;
                }
            }
        }
    } else {
#pragma unroll 4
    for (unsigned j = jg; j < K; j += 2) {
        u32 term = 0;
        if (active) {
            const u32 r = residues[rbase + u64(j) * n + i];
            if (untw_inv)
                term = gpu31::mont(r, untw_inv[u64(j) * n + i], primes[j]);
            else
                term = gpu31::weight_residue(r, pl.primes[j]);
        }
        sA[c * cfg.SA + j] = term;
        fsum += (u64(term) * corr_m[j]) >> 30;
    }
    }
    for (unsigned j = K + 1 + jg; j < W; j += 2) sA[c * cfg.SA + j] = 0;
    // spare words W..W+3 of the row hold the two partial sums
    sA[c * cfg.SA + W + 2 * jg] = u32(fsum);
    sA[c * cfg.SA + W + 2 * jg + 1] = u32(fsum >> 32);
    __syncthreads();
    if (jg == 0) {
        const u32* rw = sA + c * cfg.SA + W;
        const u64 s = (u64(rw[0]) | (u64(rw[1]) << 32)) + (u64(rw[2]) | (u64(rw[3]) << 32));
        sA[c * cfg.SA + K] = u32((s + (u64(1) << 31)) >> 32);
    }
    __syncthreads();

    TCP_T(t1);
    // 2. A fragments to registers
    const unsigned warp = tid >> 5, lane = tid & 31;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned rowbase = warp * 16;
    u32 a[CRT_KS_MAX][4];
#pragma unroll
    for (unsigned ks = 0; ks < CRT_KS_MAX; ++ks) {
        const u32* r0 = sA + (rowbase + g) * cfg.SA + ks * 8 + t;
        const u32* r1 = r0 + 8 * cfg.SA;
        const bool on = ks < cfg.KS;
        a[ks][0] = on ? r0[0] : 0;
        a[ks][1] = on ? r1[0] : 0;
        a[ks][2] = on ? r0[4] : 0;
        a[ks][3] = on ? r1[4] : 0;
    }
    __syncthreads();   // sA is now overwritten by sX

    TCP_T(t2);
    // 3. GEMM over output byte positions, CRT_NTB chains in flight
    const bool odd = t & 1;
    const unsigned sh = (t & 1) * 16;
    const unsigned row = rowbase + g + (odd ? 8 : 0);
    for (unsigned nt0 = 0; nt0 < cfg.NT; nt0 += CRT_NTB) {
        int acc[CRT_NTB][4];
#pragma unroll
        for (unsigned j = 0; j < CRT_NTB; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0;
#pragma unroll
        for (unsigned ks = 0; ks < CRT_KS_MAX; ++ks) {
            if (ks < cfg.KS) {
                uint2 b[CRT_NTB];
#pragma unroll
                for (unsigned j = 0; j < CRT_NTB; ++j) {
                    const unsigned nt = nt0 + j;
                    if (nt < cfg.NT) b[j] = frag[(std::size_t(nt) * cfg.KS + ks) * 32 + lane];
                    else b[j] = make_uint2(0u, 0u);
                }
#pragma unroll
                for (unsigned j = 0; j < CRT_NTB; ++j)
                    if (nt0 + j < cfg.NT)
                        mma_u8(acc[j], a[ks][0], a[ks][1], a[ks][2], a[ks][3], b[j].x, b[j].y);
            }
        }
#pragma unroll
        for (unsigned j = 0; j < CRT_NTB; ++j) {
            if (nt0 + j < cfg.NT) {
                const u64 pg = (u64(u32(acc[j][0])) << sh) + (u64(u32(acc[j][1])) << (sh + 8));
                const u64 pg8 = (u64(u32(acc[j][2])) << sh) + (u64(u32(acc[j][3])) << (sh + 8));
                const u64 send = odd ? pg : pg8;
                const u64 mine = odd ? pg8 : pg;
                const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
                sX[row * cfg.SX + (nt0 + j) * 2 + (t >> 1)] = tot;
            }
        }
    }
    __syncthreads();

    TCP_T(t3);
    // 4. carry + reduce, one thread per coefficient, digits in place over sX
    const unsigned DS = 2 * cfg.SX;
    u32* digits = reinterpret_cast<u32*>(smem_raw);
    if (tid < CRT_ROWS) {
        u32* dg = digits + tid * DS;
        const u64* xs = sX + tid * cfg.SX;
        u64 carry = 0;
        const unsigned words = cfg.NT * 2;
        for (unsigned w = 0; w < words; ++w) {
            carry += xs[w];
            dg[w] = u32(carry);
            carry >>= 32;
        }
        for (unsigned w = words; w < l + 2; ++w) {
            dg[w] = u32(carry);
            carry >>= 32;
        }
        gpu31::reduce_sum<1>(dg, l, pl.bits, pl.gap, pl.q);
    }
    __syncthreads();

    TCP_T(t4);
    // 5. store the tile's rows
    const unsigned rows = (n - start < CRT_ROWS) ? (n - start) : CRT_ROWS;
    if (slot_logN == 0) {
        const u64 obase = u64(pair) * n * l + u64(start) * l;
        for (unsigned idx = tid; idx < rows * l; idx += blockDim.x) {
            const unsigned r = idx / l, cc = idx - r * l;
            out[obase + idx] = digits[r * DS + cc];
        }
    } else {
        const u64 obase = u64(pair) * n * l;
        for (unsigned idx = tid; idx < rows * l; idx += blockDim.x) {
            const unsigned r = idx / l, cc = idx - r * l;
            const unsigned pos = __brev(start + r) >> (32 - slot_logN);
            out[obase + u64(pos) * l + cc] = digits[r * DS + cc];
        }
    }
    TCP_T(t5);
    TCP_ADD(0, t0, t1); TCP_ADD(1, t1, t2); TCP_ADD(2, t2, t3);
    TCP_ADD(3, t3, t4); TCP_ADD(4, t4, t5);
}

// TC15 CRT variant: 32 rows/CTA, warps split n-tiles, B reused by 2 m-tiles.
__global__ void __launch_bounds__(128, 3)
crt_tc2(gpu31::CrtArgs pl, const u32* __restrict__ residues, u32* __restrict__ out,
       const uint2* __restrict__ frag, CrtTc cfg,
       const gpu31::Prime* __restrict__ primes, const u32* __restrict__ untw_inv,
       unsigned slot_logN, const u32* __restrict__ corr_m) {
    // slot_logN != 0: residues are in bit-reversed slot order (DIF inverse);
    // row `slot` holds coefficient rev(slot).  untw_inv is then slot-ordered.
    // Shared: one region, used as sA (u32, stride SA) -> sX (u64, stride SX)
    // -> digits (u32, in place over sX, stride 2*SX).
    extern __shared__ __align__(16) unsigned char smem_raw[];
    // TC15: 32 rows/CTA; sA stays live during the GEMM, sX is separate.
    u32* sA = reinterpret_cast<u32*>(smem_raw);
    u64* sX = reinterpret_cast<u64*>(smem_raw +
        ((std::size_t(CRT2_ROWS) * cfg.SA * 4 + 15) & ~std::size_t(15)));
    constexpr unsigned NG = 128 / CRT2_ROWS;   // staging groups per row

    const unsigned n = pl.n, l = pl.limbs, K = pl.count;
    const unsigned pair = blockIdx.z, start = blockIdx.x * CRT2_ROWS;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid & (CRT2_ROWS - 1), jg = tid / CRT2_ROWS;
    const unsigned i = start + c;
    const bool active = i < n;
    const u64 rbase = u64(pair) * K * n;
    const unsigned W = cfg.KS * 8;

    TCP_T(t0);
    // 1. terms + fixed-point correction sum (term/p in 2^-32 units)
    u64 fsum = 0;
    if (untw_inv && K <= CRT_KMAX_C) {
        // TC11: batched loads (8 independent residue+untwist pairs in flight),
        // prime and correction constants from __constant__ (warp-uniform j).
#pragma unroll 1
        for (unsigned j0 = jg; j0 < K; j0 += 8 * NG) {
            u32 r[8], u[8];
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + NG * q;
                r[q] = (j < K && active) ? residues[rbase + u64(j) * n + i] : 0u;
                u[q] = (j < K && active) ? untw_inv[u64(j) * n + i] : 0u;
            }
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + NG * q;
                if (j < K) {
                    const u32 term = gpu31::mont(r[q], u[q], c_crt_prime[j]);
                    sA[c * cfg.SA + j] = term;
                    fsum += (u64(term) * c_crt_corr[j]) >> 30;
                }
            }
        }
    } else {
#pragma unroll 4
    for (unsigned j = jg; j < K; j += NG) {
        u32 term = 0;
        if (active) {
            const u32 r = residues[rbase + u64(j) * n + i];
            if (untw_inv)
                term = gpu31::mont(r, untw_inv[u64(j) * n + i], primes[j]);
            else
                term = gpu31::weight_residue(r, pl.primes[j]);
        }
        sA[c * cfg.SA + j] = term;
        fsum += (u64(term) * corr_m[j]) >> 30;
    }
    }
    for (unsigned j = K + 1 + jg; j < W; j += NG) sA[c * cfg.SA + j] = 0;
    // spare words W..W+3 of the row hold the two partial sums
    sA[c * cfg.SA + W + 2 * jg] = u32(fsum);
    sA[c * cfg.SA + W + 2 * jg + 1] = u32(fsum >> 32);
    __syncthreads();
    if (jg == 0) {
        const u32* rw = sA + c * cfg.SA + W;
        u64 s = 0;
#pragma unroll
        for (unsigned q = 0; q < NG; ++q) s += u64(rw[2 * q]) | (u64(rw[2 * q + 1]) << 32);
        sA[c * cfg.SA + K] = u32((s + (u64(1) << 31)) >> 32);
    }
    __syncthreads();

    TCP_T(t1);
    // 2-3. warps split the n-tiles (nt = warp + 4u); each warp covers all
    // 32 rows (2 m-tiles) so one B fragment feeds two mma; A from shared.
    const unsigned warp = tid >> 5, lane = tid & 31;
    const unsigned g = lane >> 2, t = lane & 3;
    const bool odd = t & 1;
    const unsigned sh = (t & 1) * 16;
    TCP_T(t2);
    for (unsigned nt0 = warp; nt0 < cfg.NT; nt0 += 8) {
        const unsigned ntA = nt0, ntB = nt0 + 4;
        const bool hasB = ntB < cfg.NT;
        int acc[2][2][4];
#pragma unroll
        for (unsigned mt = 0; mt < 2; ++mt)
#pragma unroll
            for (unsigned q = 0; q < 2; ++q) acc[mt][q][0] = acc[mt][q][1] = acc[mt][q][2] = acc[mt][q][3] = 0;
#pragma unroll 3
        for (unsigned ks = 0; ks < cfg.KS; ++ks) {
            const uint2 b0 = frag[(std::size_t(ntA) * cfg.KS + ks) * 32 + lane];
            const uint2 b1 = hasB ? frag[(std::size_t(ntB) * cfg.KS + ks) * 32 + lane] : make_uint2(0u, 0u);
#pragma unroll
            for (unsigned mt = 0; mt < 2; ++mt) {
                const u32* r0 = sA + (mt * 16 + g) * cfg.SA + ks * 8 + t;
                const u32* r1 = r0 + 8 * cfg.SA;
                const u32 x0 = r0[0], x1 = r1[0], x2 = r0[4], x3 = r1[4];
                mma_u8(acc[mt][0], x0, x1, x2, x3, b0.x, b0.y);
                if (hasB) mma_u8(acc[mt][1], x0, x1, x2, x3, b1.x, b1.y);
            }
        }
#pragma unroll
        for (unsigned mt = 0; mt < 2; ++mt)
#pragma unroll
            for (unsigned q = 0; q < 2; ++q) {
                if (q == 1 && !hasB) continue;
                const unsigned nt = q ? ntB : ntA;
                const u64 pg = (u64(u32(acc[mt][q][0])) << sh) + (u64(u32(acc[mt][q][1])) << (sh + 8));
                const u64 pg8 = (u64(u32(acc[mt][q][2])) << sh) + (u64(u32(acc[mt][q][3])) << (sh + 8));
                const u64 send = odd ? pg : pg8;
                const u64 mine = odd ? pg8 : pg;
                const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
                const unsigned row = mt * 16 + g + (odd ? 8 : 0);
                sX[row * cfg.SX + nt * 2 + (t >> 1)] = tot;
            }
    }
    __syncthreads();

    TCP_T(t3);
    // 4. carry + reduce, one thread per coefficient, digits in place over sX
    const unsigned DS = 2 * cfg.SX;
    u32* digits = reinterpret_cast<u32*>(sX);
    if (tid < CRT2_ROWS) {
        u32* dg = digits + tid * DS;
        const u64* xs = sX + tid * cfg.SX;
        u64 carry = 0;
        const unsigned words = cfg.NT * 2;
        for (unsigned w = 0; w < words; ++w) {
            carry += xs[w];
            dg[w] = u32(carry);
            carry >>= 32;
        }
        for (unsigned w = words; w < l + 2; ++w) {
            dg[w] = u32(carry);
            carry >>= 32;
        }
        gpu31::reduce_sum<1>(dg, l, pl.bits, pl.gap, pl.q);
    }
    __syncthreads();

    TCP_T(t4);
    // 5. store the tile's rows
    const unsigned rows = (n - start < CRT2_ROWS) ? (n - start) : CRT2_ROWS;
    if (slot_logN == 0) {
        const u64 obase = u64(pair) * n * l + u64(start) * l;
        for (unsigned idx = tid; idx < rows * l; idx += blockDim.x) {
            const unsigned r = idx / l, cc = idx - r * l;
            out[obase + idx] = digits[r * DS + cc];
        }
    } else {
        const u64 obase = u64(pair) * n * l;
        for (unsigned idx = tid; idx < rows * l; idx += blockDim.x) {
            const unsigned r = idx / l, cc = idx - r * l;
            const unsigned pos = __brev(start + r) >> (32 - slot_logN);
            out[obase + u64(pos) * l + cc] = digits[r * DS + cc];
        }
    }
    TCP_T(t5);
    TCP_ADD(0, t0, t1); TCP_ADD(1, t1, t2); TCP_ADD(2, t2, t3);
    TCP_ADD(3, t3, t4); TCP_ADD(4, t4, t5);
}


// ---------------------------------------------------------------------------
// Fused NTT pipeline (TC3).  Forward: gpu31's DIT (bit-reversed twisted input
// from encode_tc -> low stages in 512-tiles -> high stages across tiles, natural
// order out).  The high forward stages, the pointwise product and the high
// stages of a Gentleman-Sande DIF inverse run in ONE kernel per column group,
// then the DIF low stages run in 512-tiles; the result is in bit-reversed slot
// order, which crt_tc (slot_logN) consumes directly.  No reindex, no
// scattered pointwise store, no separate decode pass.
// ---------------------------------------------------------------------------
constexpr unsigned NT_TILE = 512;
constexpr unsigned NT_COLS = 32;
__device__ __forceinline__ unsigned padw(unsigned x) { return x + (x >> 5); }

// forward DIT low stages over both operands (identical to ntt_fast low).
__global__ void __launch_bounds__(256)
fwd_low(u32* __restrict__ a, u32* __restrict__ b, const gpu31::Prime* __restrict__ primes,
        const u32* __restrict__ packed, unsigned n, unsigned K) {
    __shared__ u32 sa[NT_TILE + NT_TILE / 32], sb[NT_TILE + NT_TILE / 32], sr[NT_TILE / 2];
    const unsigned t = threadIdx.x;
    const u64 base = (u64(blockIdx.z) * K + blockIdx.y) * n + u64(blockIdx.x) * NT_TILE;
    const gpu31::Prime p = primes[blockIdx.y];
    sr[t] = packed[u64(blockIdx.y) * (NT_TILE / 2) + t];
    sa[padw(t)] = a[base + t];
    sa[padw(t + 256)] = a[base + t + 256];
    sb[padw(t)] = b[base + t];
    sb[padw(t + 256)] = b[base + t + 256];
    __syncthreads();
#pragma unroll
    for (unsigned m = 0; m < 9; ++m) {
        const unsigned half = 1u << m, j = t & (half - 1);
        const unsigned first = 2 * (t - j) + j, second = first + half;
        const u32 w = sr[j << (8 - m)];
        gpu31::butterfly(sa[padw(first)], sa[padw(second)], w, p);
        gpu31::butterfly(sb[padw(first)], sb[padw(second)], w, p);
        __syncthreads();
    }
    a[base + t] = sa[padw(t)];
    a[base + t + 256] = sa[padw(t + 256)];
    b[base + t] = sb[padw(t)];
    b[base + t + 256] = sb[padw(t + 256)];
}

// forward DIT high stages (a, b) + pointwise + inverse DIF high stages -> c.
// grid (512/NT_COLS, K, B), block (NT_COLS, 8), smem 2*HC*(NT_COLS+1)*4.
// tw tables: [k][(HC-1)*512], stage s at offset 512*(2^s-1), [within][low].
__global__ void __launch_bounds__(256)
fused_high(const u32* __restrict__ ra, const u32* __restrict__ rb, u32* __restrict__ rc,
           const gpu31::Prime* __restrict__ primes, const u32* __restrict__ twf,
           const u32* __restrict__ twi, unsigned n, unsigned K, unsigned logH, const u32* __restrict__ d_mu) {
    // twf/twi: per prime [7][512] "low" factors followed by [127] "within"
    // factors: w = mont(A[s][low], B[(2^s-1)+within]) = root[(within*512+low)<<d].
    extern __shared__ u32 sm[];
    constexpr unsigned S = NT_COLS + 1;
    const unsigned HC = 1u << logH;
    u32* sa = sm;
    u32* sb = sm + HC * S;
    const unsigned lane = threadIdx.x, rw = threadIdx.y;
    const unsigned low = blockIdx.x * NT_COLS + lane;
    const unsigned k = blockIdx.y;
    const u64 plane = (u64(blockIdx.z) * K + k) * n;
    const gpu31::Prime p = primes[k];
#if OC9_SHOUP
    const u32 mu = d_mu[k];          // Shoup constant for this prime
#define OC9_BF(a, b, w) shoup_butterfly((a), (b), (w), mu, p)
#define OC9_MUL(x, w) shoup_mul((x), (w), mu, p)
#else
#define OC9_BF(a, b, w) gpu31::butterfly((a), (b), (w), p)
#define OC9_MUL(x, w) gpu31::mont((x), (w), p)
#endif
    constexpr unsigned TWS = 7 * NT_TILE + 127;
    const u32* tfB = twf + u64(k) * TWS + 7 * NT_TILE;
    const u32* tiB = twi + u64(k) * TWS + 7 * NT_TILE;
    u32 waf[7], wai[7];
#pragma unroll
    for (unsigned q = 0; q < 7; ++q) {
        waf[q] = q < logH ? twf[u64(k) * TWS + q * NT_TILE + low] : 0u;
        wai[q] = q < logH ? twi[u64(k) * TWS + q * NT_TILE + low] : 0u;
    }

    TCP_T(f0);
    for (unsigned h = rw; h < HC; h += 8) {
        sa[h * S + lane] = ra[plane + u64(h) * NT_TILE + low];
        sb[h * S + lane] = rb[plane + u64(h) * NT_TILE + low];
    }
    __syncthreads();
    TCP_T(f1);
#if OC8_R8
    // OC8: three DIT stages per shared round trip.  Group rows are
    // r0 + i*H (i=0..7, H=2^s).  Stage s pairs (0,1),(2,3),(4,5),(6,7);
    // stage s+1 pairs (0,2),(1,3),(4,6),(5,7); stage s+2 pairs (0,4)..(3,7).
#pragma unroll
    for (unsigned s = 0; s + 2 < 7; s += 3) {
        if (s + 2 >= logH) break;
        const unsigned H = 1u << s;
        for (unsigned g = rw; g < HC / 8; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 8 + w;
            unsigned ix[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) ix[i] = (r0 + i * H) * S + lane;
            const u32 t1 = gpu31::mont(waf[s], tfB[(H - 1) + w], p);
            const u32 t2a = gpu31::mont(waf[s + 1], tfB[(2 * H - 1) + w], p);
            const u32 t2b = gpu31::mont(waf[s + 1], tfB[(2 * H - 1) + w + H], p);
            u32 t4[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                t4[i] = gpu31::mont(waf[s + 2], tfB[(4 * H - 1) + w + i * H], p);
            u32 av[8], bv[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) { av[i] = sa[ix[i]]; bv[i] = sb[ix[i]]; }
#pragma unroll
            for (unsigned i = 0; i < 8; i += 2) {
                OC9_BF(av[i], av[i + 1], t1);
                OC9_BF(bv[i], bv[i + 1], t1);
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned i = (j < 2) ? j : j + 2;   // 0,1,4,5
                const u32 tw = (i & 1) ? t2b : t2a;
                OC9_BF(av[i], av[i + 2], tw);
                OC9_BF(bv[i], bv[i + 2], tw);
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                OC9_BF(av[i], av[i + 4], t4[i]);
                OC9_BF(bv[i], bv[i + 4], t4[i]);
            }
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) { sa[ix[i]] = av[i]; sb[ix[i]] = bv[i]; }
        }
        __syncthreads();
    }
    // radix-2 tail for the remaining stages (logH % 3 of them)
#pragma unroll
    for (unsigned k = 0; k < 3; ++k) {
        if (k >= logH % 3) break;
        const unsigned s = logH - (logH % 3) + k, H = 1u << s;
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = gpu31::mont(waf[s], tfB[(H - 1) + within], p);
            OC9_BF(sa[first * S + lane], sa[second * S + lane], w);
            OC9_BF(sb[first * S + lane], sb[second * S + lane], w);
        }
        __syncthreads();
    }
#elif TC12_R4
    // TC12: radix-4 = two DIT stages (s, s+1) per shared round trip.  A group
    // is rows r, r+H, r+2H, r+3H (r = blk*4H + w): stage s pairs (0,1),(2,3)
    // with within w; stage s+1 pairs (0,2) within w and (1,3) within w+H.
#pragma unroll
    for (unsigned s = 0; s + 1 < 7; s += 2) {
        if (s + 1 >= logH) break;
        const unsigned H = 1u << s;
        for (unsigned g = rw; g < HC / 4; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 4 + w;
            const unsigned i0 = r0 * S + lane, i1 = (r0 + H) * S + lane,
                           i2 = (r0 + 2 * H) * S + lane, i3 = (r0 + 3 * H) * S + lane;
            const u32 t1 = gpu31::mont(waf[s], tfB[(H - 1) + w], p);
            const u32 t2 = gpu31::mont(waf[s + 1], tfB[(2 * H - 1) + w], p);
            const u32 t3 = gpu31::mont(waf[s + 1], tfB[(2 * H - 1) + w + H], p);
            u32 a0 = sa[i0], a1 = sa[i1], a2 = sa[i2], a3 = sa[i3];
            u32 b0 = sb[i0], b1 = sb[i1], b2 = sb[i2], b3 = sb[i3];
            OC9_BF(a0, a1, t1); OC9_BF(a2, a3, t1);
            OC9_BF(b0, b1, t1); OC9_BF(b2, b3, t1);
            OC9_BF(a0, a2, t2); OC9_BF(a1, a3, t3);
            OC9_BF(b0, b2, t2); OC9_BF(b1, b3, t3);
            sa[i0] = a0; sa[i1] = a1; sa[i2] = a2; sa[i3] = a3;
            sb[i0] = b0; sb[i1] = b1; sb[i2] = b2; sb[i3] = b3;
        }
        __syncthreads();
    }
    if (logH & 1) {
        const unsigned s = logH - 1, H = 1u << s;
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 wa = (s == 6 ? waf[6] : (s == 4 ? waf[4] : (s == 2 ? waf[2] : waf[0])));
            const u32 w = gpu31::mont(wa, tfB[(H - 1) + within], p);
            OC9_BF(sa[first * S + lane], sa[second * S + lane], w);
            OC9_BF(sb[first * S + lane], sb[second * S + lane], w);
        }
        __syncthreads();
    }
#else
#pragma unroll
    for (unsigned s = 0; s < 7; ++s) {
        if (s >= logH) break;
        const unsigned H = 1u << s;
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = gpu31::mont(waf[s], tfB[(H - 1) + within], p);
            OC9_BF(sa[first * S + lane], sa[second * S + lane], w);
            OC9_BF(sb[first * S + lane], sb[second * S + lane], w);
        }
        __syncthreads();
    }
#endif
    TCP_T(f2);
    for (unsigned h = rw; h < HC; h += 8)
        sa[h * S + lane] = gpu31::mont(sa[h * S + lane], sb[h * S + lane], p);
    __syncthreads();
    TCP_T(f3);
#if OC8_R8
    // radix-2 DIF tail first (highest stages, reverse order)
#pragma unroll
    for (unsigned k = 0; k < 3; ++k) {
        if (k >= logH % 3) break;
        const unsigned s = logH - 1 - k, H = 1u << s;
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = gpu31::mont(wai[s], tiB[(H - 1) + within], p);
            const u32 u = sa[first * S + lane], v = sa[second * S + lane];
            sa[first * S + lane] = gpu31::add(u, v, p.p);
            sa[second * S + lane] = OC9_MUL(gpu31::sub(u, v, p.p), w);
        }
        __syncthreads();
    }
    // radix-8 DIF: stages s+2, s+1, s (descending groups)
#pragma unroll
    for (int s = int((logH / 3) * 3) - 3; s >= 0; s -= 3) {
        const unsigned H = 1u << s;
        for (unsigned g = rw; g < HC / 8; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 8 + w;
            unsigned ix[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) ix[i] = (r0 + i * H) * S + lane;
            u32 t4[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                t4[i] = gpu31::mont(wai[s + 2], tiB[(4 * H - 1) + w + i * H], p);
            const u32 t2a = gpu31::mont(wai[s + 1], tiB[(2 * H - 1) + w], p);
            const u32 t2b = gpu31::mont(wai[s + 1], tiB[(2 * H - 1) + w + H], p);
            const u32 t1 = gpu31::mont(wai[s], tiB[(H - 1) + w], p);
            u32 av[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) av[i] = sa[ix[i]];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const u32 u = av[i], v = av[i + 4];
                av[i] = gpu31::add(u, v, p.p);
                av[i + 4] = OC9_MUL(gpu31::sub(u, v, p.p), t4[i]);
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned i = (j < 2) ? j : j + 2;   // 0,1,4,5
                const u32 tw = (i & 1) ? t2b : t2a;
                const u32 u = av[i], v = av[i + 2];
                av[i] = gpu31::add(u, v, p.p);
                av[i + 2] = OC9_MUL(gpu31::sub(u, v, p.p), tw);
            }
#pragma unroll
            for (unsigned i = 0; i < 8; i += 2) {
                const u32 u = av[i], v = av[i + 1];
                av[i] = gpu31::add(u, v, p.p);
                av[i + 1] = OC9_MUL(gpu31::sub(u, v, p.p), t1);
            }
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) sa[ix[i]] = av[i];
        }
        __syncthreads();
    }
#elif TC12_R4
    if (logH & 1) {
        const unsigned s = logH - 1, H = 1u << s;
        const u32 wa = (s == 6 ? wai[6] : (s == 4 ? wai[4] : (s == 2 ? wai[2] : wai[0])));
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = gpu31::mont(wa, tiB[(H - 1) + within], p);
            const u32 u = sa[first * S + lane], v = sa[second * S + lane];
            sa[first * S + lane] = gpu31::add(u, v, p.p);
            sa[second * S + lane] = OC9_MUL(gpu31::sub(u, v, p.p), w);
        }
        __syncthreads();
    }
    // DIF pairs (s+1, s) from the top: stage s+1 on (0,2),(1,3), then s on (0,1),(2,3).
#pragma unroll
    for (int s = 4; s >= 0; s -= 2) {
        if (unsigned(s) + 2 > logH - (logH & 1)) continue;
        const unsigned H = 1u << s;
        for (unsigned g = rw; g < HC / 4; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 4 + w;
            const unsigned i0 = r0 * S + lane, i1 = (r0 + H) * S + lane,
                           i2 = (r0 + 2 * H) * S + lane, i3 = (r0 + 3 * H) * S + lane;
            const u32 t2 = gpu31::mont(wai[s + 1], tiB[(2 * H - 1) + w], p);
            const u32 t3 = gpu31::mont(wai[s + 1], tiB[(2 * H - 1) + w + H], p);
            const u32 t1 = gpu31::mont(wai[s], tiB[(H - 1) + w], p);
            u32 x0 = sa[i0], x1 = sa[i1], x2 = sa[i2], x3 = sa[i3];
            u32 u = x0, v = x2;
            x0 = gpu31::add(u, v, p.p); x2 = OC9_MUL(gpu31::sub(u, v, p.p), t2);
            u = x1; v = x3;
            x1 = gpu31::add(u, v, p.p); x3 = OC9_MUL(gpu31::sub(u, v, p.p), t3);
            u = x0; v = x1;
            x0 = gpu31::add(u, v, p.p); x1 = OC9_MUL(gpu31::sub(u, v, p.p), t1);
            u = x2; v = x3;
            x2 = gpu31::add(u, v, p.p); x3 = OC9_MUL(gpu31::sub(u, v, p.p), t1);
            sa[i0] = x0; sa[i1] = x1; sa[i2] = x2; sa[i3] = x3;
        }
        __syncthreads();
    }
#else
#pragma unroll
    for (int s = 6; s >= 0; --s) {
        if (unsigned(s) >= logH) continue;
        const unsigned H = 1u << s;
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = gpu31::mont(wai[s], tiB[(H - 1) + within], p);
            const u32 u = sa[first * S + lane], v = sa[second * S + lane];
            sa[first * S + lane] = gpu31::add(u, v, p.p);
            sa[second * S + lane] = OC9_MUL(gpu31::sub(u, v, p.p), w);
        }
        __syncthreads();
    }
#endif
    TCP_T(f4);
    for (unsigned h = rw; h < HC; h += 8)
        rc[plane + u64(h) * NT_TILE + low] = sa[h * S + lane];
    TCP_T(f5);
    TCP_ADD(5, f0, f1); TCP_ADD(6, f1, f2); TCP_ADD(7, f2, f3);
    TCP_ADD(8, f3, f4); TCP_ADD(9, f4, f5);
}

#undef OC9_BF
#undef OC9_MUL

// inverse DIF low stages in 512-tiles (output: bit-reversed slot order).
__global__ void __launch_bounds__(256)
inv_low(u32* __restrict__ c, const gpu31::Prime* __restrict__ primes,
        const u32* __restrict__ packed, unsigned n, unsigned K) {
    __shared__ u32 sa[NT_TILE + NT_TILE / 32], sr[NT_TILE / 2];
    const unsigned t = threadIdx.x;
    const u64 base = (u64(blockIdx.z) * K + blockIdx.y) * n + u64(blockIdx.x) * NT_TILE;
    const gpu31::Prime p = primes[blockIdx.y];
    sr[t] = packed[u64(blockIdx.y) * (NT_TILE / 2) + t];
    sa[padw(t)] = c[base + t];
    sa[padw(t + 256)] = c[base + t + 256];
    __syncthreads();
    // TC9: stages m=8..5 are cross-warp -> shared.  Stages m=4..0 pair
    // positions t and t^half inside one warp, so keep both of a thread's
    // positions (t and t+256) in registers and exchange with shfl.  This
    // removes 5 barriers and 10 shared round-trips per block.
#pragma unroll
    for (int m = 8; m >= 5; --m) {
        const unsigned half = 1u << m, j = t & (half - 1);
        const unsigned first = 2 * (t - j) + j, second = first + half;
        const u32 w = sr[j << (8 - m)];
        const u32 u = sa[padw(first)], v = sa[padw(second)];
        sa[padw(first)] = gpu31::add(u, v, p.p);
        sa[padw(second)] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
        __syncthreads();
    }
    u32 a0 = sa[padw(t)], a1 = sa[padw(t + 256)];
#pragma unroll
    for (int m = 4; m >= 0; --m) {
        const unsigned half = 1u << m;
        const unsigned j = t & (half - 1);
        const u32 w = sr[j << (8 - m)];
        const u32 v0 = __shfl_xor_sync(0xffffffffu, a0, half);
        const u32 v1 = __shfl_xor_sync(0xffffffffu, a1, half);
        if (t & half) {
            a0 = gpu31::mont(gpu31::sub(v0, a0, p.p), w, p);
            a1 = gpu31::mont(gpu31::sub(v1, a1, p.p), w, p);
        } else {
            a0 = gpu31::add(a0, v0, p.p);
            a1 = gpu31::add(a1, v1, p.p);
        }
    }
    c[base + t] = a0;
    c[base + t + 256] = a1;
}


// ---------------------------------------------------------------------------
// TC5: encode + forward DIT low stages in one kernel.  A CTA owns 512
// consecutive output positions (one NTT low tile) of one operand and loops
// over groups of 8 primes: tensor-core encode -> twisted residues into shared
// -> 9 DIT stages in shared -> one coalesced 2 KB store per prime.  The
// separate fwd_low pass (read+write of ra and rb) disappears.
// grid (n/512, 2, B), block 512 (16 warps x 32 rows), static smem ~25 KB.
// ---------------------------------------------------------------------------
constexpr unsigned EF_ROWS = 512;
constexpr unsigned EF_NTB = 4;                 // n-tiles per group = 8 primes
constexpr unsigned EF_PR = EF_NTB * 2;
constexpr unsigned EF_PT = NT_TILE + NT_TILE / 32;
__global__ void __launch_bounds__(512, 1)
encode_fwdlow(const u32* __restrict__ src_a, const u32* __restrict__ src_b,
              u32* __restrict__ ra, u32* __restrict__ rb,
              const uint2* __restrict__ frag, const gpu31::Prime* __restrict__ primes,
              const gpu31::Weight* __restrict__ weights, const u32* __restrict__ rtwist,
              const u32* __restrict__ packed, unsigned n, unsigned l, unsigned K,
              unsigned logN, unsigned KS, unsigned NT) {
    __shared__ u32 sT[EF_PR][EF_PT];
    __shared__ u32 sR[EF_PR][NT_TILE / 2];
    __shared__ uint2 sB[EF_NTB * ENC_KS_MAX * 32];
    const u32* src = blockIdx.y ? src_b : src_a;
    u32* dst = blockIdx.y ? rb : ra;
    const unsigned pair = blockIdx.z;
    const unsigned start = blockIdx.x * EF_ROWS;
    const u64 base = u64(pair) * n * l;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5, lane = tid & 31;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned rowbase = warp * 32;

    TCP_T(e0);
    long long pe[5] = {0, 0, 0, 0, 0};
    // A fragments straight from global (rows are bit-reversed coefficients)
    u32 a[2][ENC_KS_MAX][4];
#pragma unroll
    for (unsigned mt = 0; mt < 2; ++mt) {
        const unsigned i0 = start + rowbase + mt * 16 + g, i1 = i0 + 8;
        const u32* r0 = src + base + u64(__brev(i0) >> (32 - logN)) * l;
        const u32* r1 = src + base + u64(__brev(i1) >> (32 - logN)) * l;
#pragma unroll
        for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
            const unsigned w0 = ks * 8 + t, w1 = w0 + 4;
            const bool on = ks < KS;
            a[mt][ks][0] = (on && w0 < l) ? r0[w0] : 0u;
            a[mt][ks][1] = (on && w0 < l) ? r1[w0] : 0u;
            a[mt][ks][2] = (on && w1 < l) ? r0[w1] : 0u;
            a[mt][ks][3] = (on && w1 < l) ? r1[w1] : 0u;
        }
    }

    const bool odd = t & 1;
    const unsigned sh = (t & 1) * 16;
    const u64 rbase = u64(pair) * K * n;

    TCP_T(e1);
    TCP_ADD(15, e0, e1);
#if TC19_PF
    // TC19: the next group's twiddles and B fragments are fetched into
    // registers while the current group runs its DIT; setup is then only
    // shared stores.  sR: 2048 words = 4 per thread; sB: <= 1024 uint2 = 2.
    constexpr unsigned PF_R = EF_PR * (NT_TILE / 2) / 512, PF_B = 2;
    u32 pfR[PF_R];
    uint2 pfB[PF_B];
    auto prefetch = [&](unsigned g0n) {
#pragma unroll
        for (unsigned u = 0; u < PF_R; ++u) {
            const unsigned idx = tid + u * 512;
            const unsigned q = idx / (NT_TILE / 2), r = idx % (NT_TILE / 2);
            const unsigned kk = g0n * 2 + q;
            pfR[u] = kk < K ? packed[u64(kk) * (NT_TILE / 2) + r] : 0u;
        }
#pragma unroll
        for (unsigned u = 0; u < PF_B; ++u) {
            const unsigned idx = tid + u * 512;
            uint2 v = make_uint2(0u, 0u);
            if (idx < EF_NTB * KS * 32) {
                const unsigned j = idx / (KS * 32), rem = idx - j * (KS * 32);
                const unsigned nt = g0n + j;
                if (nt < NT) v = frag[std::size_t(nt) * KS * 32 + rem];
            }
            pfB[u] = v;
        }
    };
    prefetch(0);
#endif
    for (unsigned nt0 = 0; nt0 < NT; nt0 += EF_NTB) {
        TCP_T(g0);
#if TC19_PF
#pragma unroll
        for (unsigned u = 0; u < PF_R; ++u) {
            const unsigned idx = tid + u * 512;
            sR[idx / (NT_TILE / 2)][idx % (NT_TILE / 2)] = pfR[u];
        }
#pragma unroll
        for (unsigned u = 0; u < PF_B; ++u) {
            const unsigned idx = tid + u * 512;
            if (idx < EF_NTB * KS * 32) sB[idx] = pfB[u];
        }
#else
        // twiddles of this group's primes
        for (unsigned idx = tid; idx < EF_PR * (NT_TILE / 2); idx += blockDim.x) {
            const unsigned q = idx / (NT_TILE / 2), r = idx % (NT_TILE / 2);
            const unsigned kk = nt0 * 2 + q;
            sR[q][r] = kk < K ? packed[u64(kk) * (NT_TILE / 2) + r] : 0u;
        }
        // this group's B fragments, shared by all 16 warps
        for (unsigned idx = tid; idx < EF_NTB * KS * 32; idx += blockDim.x) {
            const unsigned j = idx / (KS * 32), rem = idx - j * (KS * 32);
            const unsigned nt = nt0 + j;
            sB[idx] = nt < NT ? frag[std::size_t(nt) * KS * 32 + rem] : make_uint2(0u, 0u);
        }
#endif
        __syncthreads();
        TCP_T(g1);
        int acc[2][EF_NTB][4];
        u32 tw[2][EF_NTB];
#pragma unroll
        for (unsigned mt = 0; mt < 2; ++mt)
#pragma unroll
            for (unsigned j = 0; j < EF_NTB; ++j) {
                acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0;
                const unsigned k = (nt0 + j) * 2 + (t >> 1);
                const unsigned i = start + rowbase + mt * 16 + g + (odd ? 8 : 0);
                tw[mt][j] = k < K ? rtwist[u64(k) * n + i] : 0u;
            }
#pragma unroll
        for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
            if (ks < KS) {
                uint2 b[EF_NTB];
#pragma unroll
                for (unsigned j = 0; j < EF_NTB; ++j) {
                    const unsigned nt = nt0 + j;
                    b[j] = sB[(j * KS + ks) * 32 + lane];
                    (void)nt;
                }
#pragma unroll
                for (unsigned j = 0; j < EF_NTB; ++j)
                    if (nt0 + j < NT) {
                        mma_u8(acc[0][j], a[0][ks][0], a[0][ks][1], a[0][ks][2], a[0][ks][3], b[j].x, b[j].y);
                        mma_u8(acc[1][j], a[1][ks][0], a[1][ks][1], a[1][ks][2], a[1][ks][3], b[j].x, b[j].y);
                    }
            }
        }
        TCP_T(g2);
#pragma unroll
        for (unsigned j = 0; j < EF_NTB; ++j) {
            const unsigned k = (nt0 + j) * 2 + (t >> 1);
            gpu31::Prime p{};
            gpu31::Weight w0{};
            if (k < K) { p = primes[k]; w0 = weights[std::size_t(k) * l]; }
#pragma unroll
            for (unsigned mt = 0; mt < 2; ++mt) {
                const u64 pg = (u64(u32(acc[mt][j][0])) << sh) + (u64(u32(acc[mt][j][1])) << (sh + 8));
                const u64 pg8 = (u64(u32(acc[mt][j][2])) << sh) + (u64(u32(acc[mt][j][3])) << (sh + 8));
                const u64 send = odd ? pg : pg8;
                const u64 mine = odd ? pg8 : pg;
                const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
                u32 val = 0;
                if (k < K) {
                    const u32 lo = gpu31::canonical_word(u32(tot), p.p);
                    u32 hi = gpu31::lazy_product(u32(tot >> 32), w0, p.p);
                    if (hi >= p.p) hi -= p.p;
                    val = gpu31::mont(gpu31::add(lo, hi, p.p), tw[mt][j], p);
                }
                const unsigned rl = rowbase + mt * 16 + g + (odd ? 8 : 0);
                sT[j * 2 + (t >> 1)][padw(rl)] = val;
            }
        }
        __syncthreads();
        TCP_T(g3);

#if TC19_PF
        if (nt0 + EF_NTB < NT) prefetch(nt0 + EF_NTB);
#endif
        // 9 DIT stages on 8 primes: 8*256 butterflies, 4 per thread per stage
        gpu31::Prime pq[4];
#pragma unroll
        for (unsigned rep = 0; rep < 4; ++rep) {
            const unsigned q = (tid >> 8) + rep * 2;
            const unsigned kk = nt0 * 2 + q;
            pq[rep] = primes[kk < K ? kk : 0];
        }
#if TC10_SHUF
        // TC10: DIT stages m=0..4 pair positions bt/bt+half inside one warp.
        // Keep the 4 primes x 2 positions in registers and exchange with shfl;
        // stages m=5..8 stay in shared (cross-warp).  Bit-exact CPU-checked.
        const unsigned bt = tid & 255, jg = tid >> 8;
        u32 v0[4], v1[4];
#pragma unroll
        for (unsigned rep = 0; rep < 4; ++rep) {
            const unsigned q = jg + rep * 2;
            v0[rep] = sT[q][padw(bt)];
            v1[rep] = sT[q][padw(bt + 256)];
        }
#pragma unroll
        for (unsigned m = 0; m < 5; ++m) {
            const unsigned half = 1u << m;
            const unsigned j = bt & (half - 1);
#pragma unroll
            for (unsigned rep = 0; rep < 4; ++rep) {
                const unsigned q = jg + rep * 2;
                const u32 w = sR[q][j << (8 - m)];
                const u32 x0 = __shfl_xor_sync(0xffffffffu, v0[rep], half);
                const u32 x1 = __shfl_xor_sync(0xffffffffu, v1[rep], half);
                if (bt & half) {
                    v0[rep] = gpu31::sub(x0, gpu31::mont(v0[rep], w, pq[rep]), pq[rep].p);
                    v1[rep] = gpu31::sub(x1, gpu31::mont(v1[rep], w, pq[rep]), pq[rep].p);
                } else {
                    v0[rep] = gpu31::add(v0[rep], gpu31::mont(x0, w, pq[rep]), pq[rep].p);
                    v1[rep] = gpu31::add(v1[rep], gpu31::mont(x1, w, pq[rep]), pq[rep].p);
                }
            }
        }
#pragma unroll
        for (unsigned rep = 0; rep < 4; ++rep) {
            const unsigned q = jg + rep * 2;
            sT[q][padw(bt)] = v0[rep];
            sT[q][padw(bt + 256)] = v1[rep];
        }
        __syncthreads();
#pragma unroll
        for (unsigned m = 5; m < 9; ++m) {
            const unsigned half = 1u << m;
#pragma unroll
            for (unsigned rep = 0; rep < 4; ++rep) {
                const unsigned q = jg + rep * 2, bt = tid & 255;
                const unsigned j = bt & (half - 1);
                const unsigned first = 2 * (bt - j) + j, second = first + half;
                gpu31::butterfly(sT[q][padw(first)], sT[q][padw(second)], sR[q][j << (8 - m)], pq[rep]);
            }
            __syncthreads();
        }
#elif TC20_R8
        // TC20: radix-8 DIT - stage triples (0,1,2),(3,4,5),(6,7,8): 3 shared
        // passes.  8 primes x 64 groups = 512 groups, one per thread.  Group =
        // positions x0 + e*H (e=0..7, x0 = blk*8H + w); stage m+s pairs
        // (e, e+2^s) for bit s of e clear, twiddle index j = w + (e mod 2^s)*H.
        {
            const unsigned q = tid >> 6, gi = tid & 63;
            const unsigned kk = nt0 * 2 + q;
            const gpu31::Prime pr8 = primes[kk < K ? kk : 0];
            u32* row = sT[q];
            const u32* tw8 = sR[q];
#pragma unroll
            for (unsigned m = 0; m < 9; m += 3) {
                const unsigned H = 1u << m;
                const unsigned w = gi & (H - 1);
                const unsigned x0 = (gi - w) * 8 + w;
                u32 v[8];
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) v[e] = row[padw(x0 + e * H)];
#pragma unroll
                for (unsigned st = 0; st < 3; ++st) {
                    const unsigned span = 1u << st;
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e) {
                        if (e & span) continue;
                        const unsigned j = w + (e & (span - 1)) * H;
                        const u32 tw = tw8[j << (8 - (m + st))];
                        gpu31::butterfly(v[e], v[e + span], tw, pr8);
                    }
                }
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) row[padw(x0 + e * H)] = v[e];
                __syncthreads();
            }
        }
#elif TC16_R4
        // TC16: radix-4 DIT - stage pairs (0,1),(2,3),(4,5),(6,7) then 8.
        // Group = positions x, x+H, x+2H, x+3H (x = blk*4H + w): stage m on
        // (0,1),(2,3) with j=w, stage m+1 on (0,2) j=w and (1,3) j=w+H.
        // 8 primes x 128 groups = 1024 groups, 2 per thread.
        gpu31::Prime pr4[2];
#pragma unroll
        for (unsigned rep = 0; rep < 2; ++rep) {
            const unsigned kk = nt0 * 2 + (tid >> 7) + rep * 4;
            pr4[rep] = primes[kk < K ? kk : 0];
        }
#pragma unroll
        for (unsigned m = 0; m < 8; m += 2) {
            const unsigned H = 1u << m;
#pragma unroll
            for (unsigned rep = 0; rep < 2; ++rep) {
                const unsigned q = (tid >> 7) + rep * 4, gi = tid & 127;
                const unsigned w = gi & (H - 1);
                const unsigned x0 = (gi - w) * 4 + w;
                u32* row = sT[q];
                const unsigned i0 = padw(x0), i1 = padw(x0 + H), i2 = padw(x0 + 2 * H), i3 = padw(x0 + 3 * H);
                u32 v0 = row[i0], v1 = row[i1], v2 = row[i2], v3 = row[i3];
                const u32 t1 = sR[q][w << (8 - m)];
                const u32 t2 = sR[q][w << (7 - m)];
                const u32 t3 = sR[q][(w + H) << (7 - m)];
                gpu31::butterfly(v0, v1, t1, pr4[rep]);
                gpu31::butterfly(v2, v3, t1, pr4[rep]);
                gpu31::butterfly(v0, v2, t2, pr4[rep]);
                gpu31::butterfly(v1, v3, t3, pr4[rep]);
                row[i0] = v0; row[i1] = v1; row[i2] = v2; row[i3] = v3;
            }
            __syncthreads();
        }
        {
            const unsigned m = 8, half = 256;
#pragma unroll
            for (unsigned rep = 0; rep < 4; ++rep) {
                const unsigned q = (tid >> 8) + rep * 2, bt = tid & 255;
                const unsigned j = bt & (half - 1);
                const unsigned first = 2 * (bt - j) + j, second = first + half;
                gpu31::butterfly(sT[q][padw(first)], sT[q][padw(second)], sR[q][j << (8 - m)], pq[rep]);
            }
            __syncthreads();
        }
#else
#pragma unroll
        for (unsigned m = 0; m < 9; ++m) {
            const unsigned half = 1u << m;
#pragma unroll
            for (unsigned rep = 0; rep < 4; ++rep) {
                const unsigned q = (tid >> 8) + rep * 2, bt = tid & 255;
                const unsigned j = bt & (half - 1);
                const unsigned first = 2 * (bt - j) + j, second = first + half;
                gpu31::butterfly(sT[q][padw(first)], sT[q][padw(second)], sR[q][j << (8 - m)], pq[rep]);
            }
            __syncthreads();
        }
#endif
        TCP_T(g4);
        for (unsigned idx = tid; idx < EF_PR * NT_TILE; idx += blockDim.x) {
            const unsigned q = idx / NT_TILE, pos = idx % NT_TILE;
            const unsigned kk = nt0 * 2 + q;
            if (kk < K) dst[rbase + u64(kk) * n + start + pos] = sT[q][padw(pos)];
        }
        __syncthreads();
        TCP_T(g5);
        pe[0] += g1 - g0; pe[1] += g2 - g1; pe[2] += g3 - g2; pe[3] += g4 - g3; pe[4] += g5 - g4;
    }
    TCP_ADD(10, 0, pe[0]); TCP_ADD(11, 0, pe[1]); TCP_ADD(12, 0, pe[2]);
    TCP_ADD(13, 0, pe[3]); TCP_ADD(14, 0, pe[4]);
}

#endif  // __CUDACC__

}  // namespace tc
