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

namespace tc {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr unsigned ENC_KS_MAX = 7;    // L <= 56
constexpr unsigned CRT_KS_MAX = 15;   // K+1 <= 120
constexpr unsigned ENC_ROWS = 128;    // 4 warps x 32 rows
constexpr unsigned ENC_S = 68;        // smem words per row: >= 56, == 4 (mod 32)
constexpr unsigned CRT_ROWS = 64;     // 4 warps x 16 rows

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

inline unsigned crt_sa_stride(unsigned KS) {    // >= KS*8 words, == 4 (mod 32)
    unsigned s = KS * 8;
    while (s % 32 != 4) ++s;
    return s;
}

#if defined(__CUDACC__)

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
__global__ void __launch_bounds__(128)
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
    const unsigned rowbase = warp * 32;

    u32 a[2][ENC_KS_MAX][4];
#pragma unroll
    for (unsigned mt = 0; mt < 2; ++mt)
#pragma unroll
        for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
            const u32* r0 = sA + (rowbase + mt * 16 + g) * ENC_S + ks * 8 + t;
            const u32* r1 = r0 + 8 * ENC_S;
            const bool on = ks < KS;
            a[mt][ks][0] = on ? r0[0] : 0;
            a[mt][ks][1] = on ? r1[0] : 0;
            a[mt][ks][2] = on ? r0[4] : 0;
            a[mt][ks][3] = on ? r1[4] : 0;
        }

    const u64 rbase = u64(pair) * K * n;
    for (unsigned nt = 0; nt < NT; ++nt) {
        int acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        const uint2* fp = frag + std::size_t(nt) * KS * 32 + lane;
#pragma unroll
        for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
            if (ks < KS) {
                const uint2 b = fp[ks * 32];
                mma_u8(acc[0], a[0][ks][0], a[0][ks][1], a[0][ks][2], a[0][ks][3], b.x, b.y);
                mma_u8(acc[1], a[1][ks][0], a[1][ks][1], a[1][ks][2], a[1][ks][3], b.x, b.y);
            }
        }
        const unsigned k = nt * 2 + (t >> 1);
        const unsigned sh = (t & 1) * 16;
        const bool odd = t & 1;
        gpu31::Prime p{};
        gpu31::Weight w0{};
        if (k < K) { p = primes[k]; w0 = weights[std::size_t(k) * l]; }
#pragma unroll
        for (unsigned mt = 0; mt < 2; ++mt) {
            const u64 pg = (u64(u32(acc[mt][0])) << sh) + (u64(u32(acc[mt][1])) << (sh + 8));
            const u64 pg8 = (u64(u32(acc[mt][2])) << sh) + (u64(u32(acc[mt][3])) << (sh + 8));
            const u64 send = odd ? pg : pg8;
            const u64 mine = odd ? pg8 : pg;
            const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
            const unsigned i = start + rowbase + mt * 16 + g + (odd ? 8 : 0);
            if (k < K && i < n) {
                const u32 lo = gpu31::canonical_word(u32(tot), p.p);
                u32 hi = gpu31::lazy_product(u32(tot >> 32), w0, p.p);
                if (hi >= p.p) hi -= p.p;
                const u32 v = gpu31::add(lo, hi, p.p);
                dst[rbase + u64(k) * n + i] = gpu31::mont(v, rtwist[u64(k) * n + i], p);
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
    if (x > a) a = x;
    a = (a + 15) & ~std::size_t(15);
    return a + std::size_t(CRT_ROWS) * c.DS * 4 + 2 * CRT_ROWS * sizeof(double) + 16;
}

__global__ void __launch_bounds__(128)
crt_tc(gpu31::CrtArgs pl, const u32* __restrict__ residues, u32* __restrict__ out,
       const uint2* __restrict__ frag, CrtTc cfg,
       const gpu31::Prime* __restrict__ primes, const u32* __restrict__ untw_inv,
       unsigned slot_logN) {
    // slot_logN != 0: residues are in bit-reversed slot order (DIF inverse);
    // row `slot` holds coefficient rev(slot).  untw_inv is then slot-ordered.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::size_t abytes = std::size_t(CRT_ROWS) * cfg.SA * 4;
    {
        const std::size_t x = std::size_t(CRT_ROWS) * cfg.SX * 8;
        if (x > abytes) abytes = x;
        abytes = (abytes + 15) & ~std::size_t(15);
    }
    u32* sA = reinterpret_cast<u32*>(smem_raw);
    u64* sX = reinterpret_cast<u64*>(smem_raw);
    u32* digits = reinterpret_cast<u32*>(smem_raw + abytes);
    double* red = reinterpret_cast<double*>(smem_raw + abytes + std::size_t(CRT_ROWS) * cfg.DS * 4 +
                                            (16 - (std::size_t(CRT_ROWS) * cfg.DS * 4) % 16) % 16);

    const unsigned n = pl.n, l = pl.limbs, K = pl.count;
    const unsigned pair = blockIdx.z, start = blockIdx.x * CRT_ROWS;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid & 63, jg = tid >> 6;
    const unsigned i = start + c;
    const bool active = i < n;
    const u64 rbase = u64(pair) * K * n;
    const unsigned W = cfg.KS * 8;

    // 1. terms, correction sum
    double dsum = 0;
    for (unsigned j = jg; j < K; j += 2) {
        const gpu31::PrimeInfo pi = pl.primes[j];
        u32 term = 0;
        if (active) {
            const u32 r = residues[rbase + u64(j) * n + i];
            if (untw_inv)   // fused decode: one Montgomery product by untwist*inv
                term = gpu31::mont(r, untw_inv[u64(j) * n + i], primes[j]);
            else
                term = gpu31::weight_residue(r, pi);
        }
        sA[c * cfg.SA + j] = term;
        dsum += double(term) * pi.reciprocal;
    }
    red[jg * CRT_ROWS + c] = dsum;
    for (unsigned j = K + 1 + jg; j < W; j += 2) sA[c * cfg.SA + j] = 0;
    __syncthreads();
    if (jg == 0) sA[c * cfg.SA + K] = u32(red[c] + red[CRT_ROWS + c] + 0.5);
    __syncthreads();

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

    // 3. GEMM over output byte positions, fold into 32-bit-word partial sums
    const bool odd = t & 1;
    const unsigned sh = (t & 1) * 16;
    for (unsigned nt = 0; nt < cfg.NT; ++nt) {
        int acc[4] = {0, 0, 0, 0};
        const uint2* fp = frag + std::size_t(nt) * cfg.KS * 32 + lane;
#pragma unroll
        for (unsigned ks = 0; ks < CRT_KS_MAX; ++ks) {
            if (ks < cfg.KS) {
                const uint2 b = fp[ks * 32];
                mma_u8(acc, a[ks][0], a[ks][1], a[ks][2], a[ks][3], b.x, b.y);
            }
        }
        const u64 pg = (u64(u32(acc[0])) << sh) + (u64(u32(acc[1])) << (sh + 8));
        const u64 pg8 = (u64(u32(acc[2])) << sh) + (u64(u32(acc[3])) << (sh + 8));
        const u64 send = odd ? pg : pg8;
        const u64 mine = odd ? pg8 : pg;
        const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
        const unsigned row = rowbase + g + (odd ? 8 : 0);
        const unsigned word = nt * 2 + (t >> 1);
        sX[row * cfg.SX + word] = tot;
    }
    __syncthreads();

    // 4. carry + reduce, one thread per coefficient
    if (tid < CRT_ROWS) {
        u32* dg = digits + tid * cfg.DS;
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

    // 5. coalesced store of the tile's rows
    const unsigned rows = (n - start < CRT_ROWS) ? (n - start) : CRT_ROWS;
    if (slot_logN == 0) {
        const u64 obase = u64(pair) * n * l + u64(start) * l;
        for (unsigned idx = tid; idx < rows * l; idx += blockDim.x) {
            const unsigned r = idx / l, cc = idx - r * l;
            out[obase + idx] = digits[r * cfg.DS + cc];
        }
    } else {
        const u64 obase = u64(pair) * n * l;
        for (unsigned idx = tid; idx < rows * l; idx += blockDim.x) {
            const unsigned r = idx / l, cc = idx - r * l;
            const unsigned pos = __brev(start + r) >> (32 - slot_logN);
            out[obase + u64(pos) * l + cc] = digits[r * cfg.DS + cc];
        }
    }
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
           const u32* __restrict__ twi, unsigned n, unsigned K, unsigned logH) {
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
    const u32* tf = twf + u64(k) * (HC - 1) * NT_TILE;
    const u32* ti = twi + u64(k) * (HC - 1) * NT_TILE;

    for (unsigned h = rw; h < HC; h += 8) {
        sa[h * S + lane] = ra[plane + u64(h) * NT_TILE + low];
        sb[h * S + lane] = rb[plane + u64(h) * NT_TILE + low];
    }
    __syncthreads();
    for (unsigned s = 0; s < logH; ++s) {
        const unsigned H = 1u << s, off = NT_TILE * (H - 1);
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = tf[off + within * NT_TILE + low];
            gpu31::butterfly(sa[first * S + lane], sa[second * S + lane], w, p);
            gpu31::butterfly(sb[first * S + lane], sb[second * S + lane], w, p);
        }
        __syncthreads();
    }
    for (unsigned h = rw; h < HC; h += 8)
        sa[h * S + lane] = gpu31::mont(sa[h * S + lane], sb[h * S + lane], p);
    __syncthreads();
    for (unsigned s = logH; s-- > 0;) {
        const unsigned H = 1u << s, off = NT_TILE * (H - 1);
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = ti[off + within * NT_TILE + low];
            const u32 u = sa[first * S + lane], v = sa[second * S + lane];
            sa[first * S + lane] = gpu31::add(u, v, p.p);
            sa[second * S + lane] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
        }
        __syncthreads();
    }
    for (unsigned h = rw; h < HC; h += 8)
        rc[plane + u64(h) * NT_TILE + low] = sa[h * S + lane];
}

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
#pragma unroll
    for (int m = 8; m >= 0; --m) {
        const unsigned half = 1u << m, j = t & (half - 1);
        const unsigned first = 2 * (t - j) + j, second = first + half;
        const u32 w = sr[j << (8 - m)];
        const u32 u = sa[padw(first)], v = sa[padw(second)];
        sa[padw(first)] = gpu31::add(u, v, p.p);
        sa[padw(second)] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
        __syncthreads();
    }
    c[base + t] = sa[padw(t)];
    c[base + t + 256] = sa[padw(t + 256)];
}

#endif  // __CUDACC__

}  // namespace tc
