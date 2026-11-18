// crt_mma_fast.cuh — tensor-core CRT with PRE-BAKED byte-split fragments.
//
// Why this file exists: crt_mma_diag.cuh proved the mma.sync CRT is correct
// (1.06 ms on 2 pairs) but took 62 ms at B=128 because ~121 scalar
// shift/mask/OR instructions surrounded every single mma.sync.  The tensor
// cores executed <4% of the warp instructions.
//
// The fix here is data layout, not arithmetic:
//   * B operand (CRT weights, fixed) is byte-split ON THE HOST into u32 words
//     that are already the m16n8k32 B fragments: one LDG.32 per fragment,
//     zero shift/mask/OR in the k-loop.
//   * A operand (Shoup terms, per-tile) is written into shared memory as 4
//     byte-plane u32 words, also already MMA fragments.
// The k-loop body is then 4 A loads + 2 B loads + 1 mma.sync (~7 instructions
// instead of ~121).  The reassembly and carry/reduce are kept verbatim from
// crt_mma_diag.cuh (they are proven correct); the stride-32 bank conflict is
// fixed by using an odd stride (33).
//
// Namespace `crt_mma_fast`; API mirrors crt_mma::CrtMmaPlan so integration into
// solve.cu is a two-line swap.  Pure-CPU selftests at the bottom validate the
// packing, the fragment mapping and the byte-split reassembly against the
// reference sum_column/carry_and_reduce before any GPU run.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "fherma.h"

namespace crt_mma_fast {

using u32 = std::uint32_t;
using u64 = std::uint64_t;

#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
#define CRTMF_FHD __host__ __device__
#else
#define CRTMF_FHD
#endif

static constexpr unsigned MF_MAXL      = 128;
static constexpr unsigned MF_MAXPITCH  = 128;   // pitch=119 at the ranked point
static constexpr unsigned MF_M_COEF    = 32;                // coefs / CTA
static constexpr unsigned MF_WARPS     = MF_M_COEF / 4;     // 8
static constexpr unsigned MF_THREADS   = MF_WARPS * 32;     // 256
static constexpr unsigned MF_TS        = MF_M_COEF + 1;     // 33, ODD
static constexpr unsigned MF_ACC_WORDS = MF_WARPS * 2u * 4u * 4u * 4u;
static constexpr unsigned MF_JG        = (MF_MAXPITCH + 3) / 4;  // 128 j-groups

struct PrimeInfo { u32 p, inv, inv_shoup; double reciprocal; };
struct CrtArgs {
    unsigned n, limbs, bits, count;
    u32 gap;
    const PrimeInfo* primes;
    const u32* weights;   // unused by the fast path (B is pre-baked)
    const u32* q;
};

// ---- reference reduction helpers (verbatim from gpu31 / crt_mma_diag) ------
CRTMF_FHD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    const u32 quot = __umulhi(r, p.inv_shoup);
#else
    const u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    const u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}
template <unsigned Stride>
CRTMF_FHD inline u32 shifted_word(const u32* q, unsigned limbs, unsigned pos, unsigned shift) {
    const unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    const unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}
template <unsigned Stride>
CRTMF_FHD inline bool subtract_if_ge(u32* a, unsigned size, const u32* q,
                                     unsigned limbs, unsigned shift) {
    for (unsigned j = size; j-- > 0;) {
        const u32 b = shifted_word<1>(q, limbs, j, shift);
        if (a[j * Stride] < b) return false;
        if (a[j * Stride] > b) break;
    }
    u64 borrow = 0;
    for (unsigned j = 0; j < size; ++j) {
        const u64 b = u64(shifted_word<1>(q, limbs, j, shift)) + borrow;
        const u32 x = a[j * Stride];
        a[j * Stride] = x - u32(b);
        borrow = u64(x) < b;
    }
    return true;
}
template <unsigned Stride>
CRTMF_FHD inline u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}
template <unsigned Stride>
CRTMF_FHD inline void reduce_sum(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
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
CRTMF_FHD inline void carry_and_reduce(u32* digits, const u64* low, const u64* high,
                                       unsigned l, unsigned bits, u32 gap, const u32* q,
                                       unsigned low_stride) {
    u64 carry = 0, prior_high = 0;
    for (unsigned i = 0; i < l; ++i) {
        carry += low[i * low_stride] + prior_high;
        digits[i] = u32(carry);
        carry >>= 32;
        prior_high = high[i * low_stride];
    }
    carry += prior_high;
    digits[l] = u32(carry);
    digits[l + 1] = u32(carry >> 32);
    reduce_sum<1>(digits, l, bits, gap, q);
}
struct U128 { u64 lo, hi; };
CRTMF_FHD inline void u128_add_shift(U128& x, u64 v, unsigned sh) {
    const u64 add_lo = (sh < 64) ? (v << sh) : 0ull;
    const u64 add_hi = (sh == 0) ? 0ull : ((sh < 64) ? (v >> (64 - sh)) : 0ull);
    const u64 nl = x.lo + add_lo;
    x.hi += add_hi + (nl < x.lo ? 1ull : 0ull);
    x.lo = nl;
}

// ---- host packing ---------------------------------------------------------
// WB[i][cw][cb][tg] : u32 whose 4 bytes are byte `cw` of
//   weights[i*pitch + (cb*32 + tg*4 + bb)], bb=0..3.
// jgroup AP[c][jg][m] : u32 whose 4 bytes are byte `c` of
//   terms[jg*4+bb][m], bb=0..3.
inline std::vector<u32> bake_wb(const std::vector<u32>& w, unsigned L, unsigned pitch) {
    const unsigned chunks = (pitch + 31) / 32;
    std::vector<u32> wb(std::size_t(L) * 4 * chunks * 8, 0u);
    for (unsigned i = 0; i < L; ++i)
        for (unsigned cb = 0; cb < chunks; ++cb)
            for (unsigned tg = 0; tg < 8; ++tg)
                for (unsigned cw = 0; cw < 4; ++cw) {
                    u32 packed = 0;
                    for (unsigned bb = 0; bb < 4; ++bb) {
                        const unsigned j = cb * 32 + tg * 4 + bb;
                        const u32 v = (j < pitch) ? w[std::size_t(i) * pitch + j] : 0u;
                        packed |= ((v >> (8u * cw)) & 0xFFu) << (8u * bb);
                    }
                    wb[(((std::size_t(i) * 4 + cw) * chunks + cb) * 8 + tg)] = packed;
                }
    return wb;
}

// ---- device kernel --------------------------------------------------------
#if defined(__CUDACC__) && !defined(PGPU_HOST_CHECK)
template <int MODE>
__global__ void crt_mma_fast_kernel(CrtArgs plan, const u32* residues,
                                    const u32* d_wb, u32* out, u32 batches) {
    (void)MODE;
    const unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    const unsigned chunks = (pitch + 31) / 32;
    const unsigned jgroups = (pitch + 3) / 4;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned g = lane >> 2, t = lane & 3u;
    const unsigned coef_tile = blockIdx.x * MF_M_COEF;
    const unsigned pair = blockIdx.z;
    const unsigned m_base = 4u * warp + (g >> 2);
    const unsigned m_hi = m_base + 2u;

    extern __shared__ unsigned char smem[];
    u32* terms = reinterpret_cast<u32*>(smem);                       // pitch * 33
    u32* ap = reinterpret_cast<u32*>(terms + std::size_t(pitch) * MF_TS);  // 4 * jgroups * 32
    u64* low = reinterpret_cast<u64*>(ap + 4ull * MF_JG * MF_M_COEF);
    u64* high = low + std::size_t(l) * MF_TS;
    int* acc = reinterpret_cast<int*>(high + std::size_t(l) * MF_TS);

    // step 1: Shoup terms, written BOTH as u32 (for FP64 correction) and as
    // four A byte planes in MMA-fragment order.  One thread per (jgroup, m)
    // handles all four j = jgroup*4+bb, so the byte-plane write is race-free.
    for (unsigned jg = threadIdx.x / MF_M_COEF; jg < jgroups; jg += MF_THREADS / MF_M_COEF) {
        const unsigned m = threadIdx.x % MF_M_COEF;
        const unsigned coef = coef_tile + m;
        u32 plane[4] = {0, 0, 0, 0};
        #pragma unroll
        for (unsigned bb = 0; bb < 4; ++bb) {
            const unsigned j = jg * 4 + bb;
            const u32 r = (coef < plan.n && j < k)
                        ? residues[u64(pair) * k * plan.n + u64(j) * plan.n + coef] : 0u;
            const u32 val = (j < pitch) ? weight_residue(r, plan.primes[j]) : 0u;
            terms[std::size_t(j) * MF_TS + m] = val;
            #pragma unroll
            for (unsigned c = 0; c < 4; ++c)
                plane[c] |= ((val >> (8u * c)) & 0xFFu) << (8u * bb);
        }
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c)
            ap[(std::size_t(c) * MF_JG + jg) * MF_M_COEF + m] = plane[c];
    }
    __syncthreads();

    // step 2: FP64 correction (verbatim), also injected into the A planes at
    // jgroup = k/4, group k%4.
    if (coef_tile < plan.n && threadIdx.x < MF_M_COEF) {
        const unsigned m = threadIdx.x;
        double sum = 0;
        for (unsigned j = 0; j < k; ++j)
            sum += double(terms[std::size_t(j) * MF_TS + m]) * plan.primes[j].reciprocal;
        const u32 val = u32(sum + 0.5);
        const unsigned jg = k >> 2, bb = k & 3u;
        terms[std::size_t(k) * MF_TS + m] = val;
        // Patch byte `bb` of the correction jgroup word with byte c of val.
        #pragma unroll
        for (unsigned c = 0; c < 4; ++c) {
            u32& word = ap[(std::size_t(c) * MF_JG + jg) * MF_M_COEF + m];
            const u32 bytev = (val >> (8u * c)) & 0xFFu;
            word |= bytev << (8u * bb);
        }
    }
    __syncthreads();

    // step 3: tensor-core byte-split product.  A fragments come from shared
    // A-planes, B fragments from the pre-baked host table (one LDG.32 each).
    int* accw = acc + warp * (2u * 4u * 4u * 4u);
    const unsigned cw = g & 3u;
    for (unsigned ip = 0; ip < (l + 1u) / 2u; ++ip) {
        int c0 = 0, c1v = 0, c2v = 0, c3v = 0;
        for (unsigned kb = 0; kb < pitch; kb += 32u) {
            const unsigned cb = kb >> 5;
            const unsigned jg0 = cb * 4 + t, jg1 = jg0 + 4;
            const u32 a0 = ap[(std::size_t(cw) * MF_JG + jg0) * MF_M_COEF + m_base];
            const u32 a1 = ap[(std::size_t(cw) * MF_JG + jg0) * MF_M_COEF + m_hi];
            const u32 a2 = ap[(std::size_t(cw) * MF_JG + jg1) * MF_M_COEF + m_base];
            const u32 a3 = ap[(std::size_t(cw) * MF_JG + jg1) * MF_M_COEF + m_hi];
            const unsigned limb0 = 2u * ip + (g >> 2);
            u32 b0 = 0, b1 = 0;
            if (limb0 < l) {
                const u32* wb = d_wb + (((std::size_t(limb0) * 4 + cw) * chunks + cb) * 8);
                b0 = wb[t];
                b1 = wb[t + 4];
            }
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+r"(c0), "+r"(c1v), "+r"(c2v), "+r"(c3v)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
        // stash (verbatim mapping from crt_mma_diag.cuh)
        const unsigned ab = t >> 1, cw0 = (t & 1u) << 1;
        const unsigned cg = g >> 2, ct = g & 3u;
        const unsigned idx0 = ((ab * 4u + cg) * 4u + ct) * 4u + cw0;
        const unsigned idx1 = ((ab * 4u + cg + 2u) * 4u + ct) * 4u + cw0;
        accw[idx0 + 0] = c0; accw[idx0 + 1] = c1v;
        accw[idx1 + 0] = c2v; accw[idx1 + 1] = c3v;
        __syncwarp();
        if (lane < 8u) {
            const unsigned co = lane >> 1, lb = lane & 1u;
            U128 X{0ull, 0ull};
            #pragma unroll
            for (unsigned ct2 = 0; ct2 < 4u; ++ct2)
                #pragma unroll
                for (unsigned cw2 = 0; cw2 < 4u; ++cw2)
                    u128_add_shift(X, u64(u32(accw[((lb * 4u + co) * 4u + ct2) * 4u + cw2])),
                                   8u * (ct2 + cw2));
            const unsigned i = 2u * ip + lb;
            if (i < l) {
                const unsigned mslot = 4u * warp + co;
                low[i * MF_TS + mslot] = u64(u32(X.lo));
                high[i * MF_TS + mslot] = (X.lo >> 32) | (X.hi << 32);
            }
        }
        __syncwarp();
    }
    __syncthreads();

    // step 4: reference carry + reduce (verbatim, contiguous digits)
    if (threadIdx.x < MF_M_COEF && coef_tile + threadIdx.x < plan.n) {
        const unsigned m = threadIdx.x;
        u32 digits[MF_MAXL + 2];
        carry_and_reduce(digits, low + m, high + m, l, plan.bits, plan.gap, plan.q, MF_TS);
        const unsigned coef = coef_tile + m;
        const u64 dst = u64(pair) * plan.n * l + u64(coef) * l;
        if (dst + l <= u64(batches) * plan.n * l)
            for (unsigned i = 0; i < l; ++i) out[dst + i] = digits[i];
    }
}
#endif  // __CUDACC__ && !PGPU_HOST_CHECK

// ---- shared-bytes calculator ----------------------------------------------
inline std::size_t shared_bytes(unsigned l, unsigned pitch) {
    return std::size_t(pitch) * MF_TS * sizeof(u32)
         + 4ull * MF_JG * MF_M_COEF * sizeof(u32)
         + 2ull * std::size_t(l) * MF_TS * sizeof(u64)
         + MF_ACC_WORDS * sizeof(int);
}

// ---- host plan + launchers -------------------------------------------------
struct CrtMmaFastPlan {
    unsigned L = 0, pitch = 0;
    std::vector<u32> wb_host;
    u32* d_wb = nullptr;
    bool ok = false;

    CrtMmaFastPlan() = default;
    CrtMmaFastPlan(const std::vector<u32>& w, unsigned L_, unsigned pitch_)
        : L(L_), pitch(pitch_) {
        if (L == 0 || pitch == 0 || L > MF_MAXL || pitch > MF_MAXPITCH)
            throw std::runtime_error("crt_mma_fast: L/pitch outside compiled limits");
        wb_host = bake_wb(w, L, pitch);
#ifdef __CUDACC__
        if (cudaMalloc(reinterpret_cast<void**>(&d_wb), wb_host.size() * sizeof(u32)) != cudaSuccess)
            throw std::runtime_error("crt_mma_fast: cudaMalloc failed");
        if (cudaMemcpy(d_wb, wb_host.data(), wb_host.size() * sizeof(u32),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_wb); d_wb = nullptr;
            throw std::runtime_error("crt_mma_fast: upload failed");
        }
        ok = true;
#endif
    }
    CrtMmaFastPlan(const CrtMmaFastPlan&) = delete;
    CrtMmaFastPlan& operator=(const CrtMmaFastPlan&) = delete;
    ~CrtMmaFastPlan() {
#ifdef __CUDACC__
        if (d_wb) cudaFree(d_wb);
#endif
        d_wb = nullptr;
    }

    template <class Args>
    cudaError_t run(const u32* residues, u32* out, const Args& a, u32 B,
                    cudaStream_t stream = nullptr) const {
#ifdef __CUDACC__
        if (!ok || !residues || !out || B == 0) return cudaErrorInvalidValue;
        if (a.limbs != L || a.count + 1 != pitch) return cudaErrorInvalidValue;
        CrtArgs ca{a.n, a.limbs, a.bits, a.count, a.gap,
                   reinterpret_cast<const PrimeInfo*>(a.primes), nullptr, a.q};
        const std::size_t smem = shared_bytes(L, pitch);
        cudaFuncSetAttribute(crt_mma_fast_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
        crt_mma_fast_kernel<0><<<dim3((ca.n + MF_M_COEF - 1) / MF_M_COEF, 1, B),
                                 MF_THREADS, smem, stream>>>(
            ca, residues, d_wb, out, B);
        return cudaGetLastError();
#else
        (void)residues; (void)out; (void)a; (void)B; (void)stream;
        return cudaSuccess;
#endif
    }
};

// ---- pure-CPU selftests ----------------------------------------------------
inline bool bake_wb_selftest(unsigned iterations = 4096, u64 seed = 0x1234abcdull) {
    auto next = [&seed]() -> u64 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    for (unsigned it = 0; it < iterations; ++it) {
        const unsigned L = 1 + (next() % 8), pitch = 4 * (1 + (next() % 4));
        std::vector<u32> w(std::size_t(L) * pitch);
        for (auto& v : w) v = u32(next());
        const auto wb = bake_wb(w, L, pitch);
        const unsigned chunks = (pitch + 31) / 32;
        for (unsigned i = 0; i < L; ++i)
            for (unsigned j = 0; j < pitch; ++j) {
                u32 recon = 0;
                const unsigned cb = j / 32, rem = j % 32, tg = rem / 4, bb = rem % 4;
                for (unsigned c = 0; c < 4; ++c) {
                    const u32 word = wb[(((std::size_t(i) * 4 + c) * chunks + cb) * 8 + tg)];
                    recon |= ((word >> (8u * bb)) & 0xFFu) << (8u * c);
                }
                if (recon != w[std::size_t(i) * pitch + j]) return false;
            }
    }
    return true;
}

// End-to-end tile check: pack A planes exactly as the kernel does, bake B,
// form the 16 byte-pair sums, reassemble and reduce, compare against the
// reference sum_column/carry_and_reduce.  This catches any packing/layout bug
// without a GPU.
inline bool tile_selftest(unsigned iterations = 64, u64 seed = 0xfeed1234ull) {
    auto next = [&seed]() -> u64 { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    const unsigned CO = 4, pitch = 8, L = 8;
    const unsigned chunks = (pitch + 31) / 32, jgroups = (pitch + 3) / 4;
    for (unsigned it = 0; it < iterations; ++it) {
        u32 terms[8][4] = {}, weights[8][8] = {};
        for (unsigned j = 0; j < pitch; ++j)
            for (unsigned m = 0; m < CO; ++m) terms[j][m] = u32(next());
        for (unsigned i = 0; i < L; ++i)
            for (unsigned j = 0; j < pitch; ++j) weights[i][j] = u32(next());
        std::vector<u32> wflat(std::size_t(L) * pitch);
        for (unsigned i = 0; i < L; ++i)
            for (unsigned j = 0; j < pitch; ++j) wflat[i * pitch + j] = weights[i][j];
        const auto wb = bake_wb(wflat, L, pitch);
        u32 ap[4][MF_JG][CO] = {};
        for (unsigned jg = 0; jg < jgroups; ++jg)
            for (unsigned m = 0; m < CO; ++m) {
                u32 plane[4] = {0, 0, 0, 0};
                for (unsigned bb = 0; bb < 4; ++bb) {
                    const u32 v = terms[jg * 4 + bb][m];
                    for (unsigned c = 0; c < 4; ++c)
                        plane[c] |= ((v >> (8u * c)) & 0xFFu) << (8u * bb);
                }
                for (unsigned c = 0; c < 4; ++c) ap[c][jg][m] = plane[c];
            }
        for (unsigned m = 0; m < CO; ++m) {
            u64 lo[L] = {}, hi[L] = {}, rlo[L] = {}, rhi[L] = {};
            for (unsigned i = 0; i < L; ++i) {
                u64 S[4][4] = {};
                for (unsigned j = 0; j < pitch; ++j) {
                    const unsigned jg = j >> 2, bb = j & 3u;
                    const u32 t = terms[j][m];
                    for (unsigned ct = 0; ct < 4; ++ct) {
                        const u32 ab = (ap[ct][jg][m] >> (8u * bb)) & 0xFFu;
                        if (ab != ((t >> (8u * ct)) & 0xFFu)) return false;
                        for (unsigned cw = 0; cw < 4; ++cw) {
                            const unsigned cb = j >> 5, tg = (j & 31u) >> 2;
                            const u32 word = wb[((i * 4 + cw) * chunks + cb) * 8 + tg];
                            const u32 bv = (word >> (8u * (j & 3u))) & 0xFFu;
                            if (bv != ((weights[i][j] >> (8u * cw)) & 0xFFu)) return false;
                            S[ct][cw] += u64(ab) * bv;
                        }
                    }
                }
                U128 X{0ull, 0ull};
                for (unsigned ct = 0; ct < 4; ++ct)
                    for (unsigned cw = 0; cw < 4; ++cw)
                        u128_add_shift(X, S[ct][cw], 8u * (ct + cw));
                lo[i] = u64(u32(X.lo));
                hi[i] = (X.lo >> 32) | (X.hi << 32);
                u64 l2 = 0, h2 = 0;
                for (unsigned j = 0; j < pitch; ++j) {
                    const u64 p = u64(terms[j][m]) * weights[i][j];
                    l2 += u32(p); h2 += p >> 32;
                }
                if (lo[i] + (hi[i] << 32) != l2 + (h2 << 32)) return false;
                rlo[i] = l2; rhi[i] = h2;
            }
            u32 got[10] = {}, want[10] = {}, q[8] = {};
            carry_and_reduce(got, lo, hi, L, 256u, 0u, q, 1);
            carry_and_reduce(want, rlo, rhi, L, 256u, 0u, q, 1);
            for (unsigned i = 0; i < L; ++i) if (got[i] != want[i]) return false;
        }
    }
    return true;
}

inline bool reassembly_selftest(unsigned iterations = 4096, u64 seed = 0xa5a5f00dull) {
    auto next = [&seed]() -> u64 { seed ^= seed >> 12; seed ^= seed << 25; seed ^= seed >> 27; return seed * 0x2545F4914F6CDD1Dull; };
    for (unsigned it = 0; it < iterations; ++it) {
        const u32 a = u32(next()), b = u32(next());
        U128 X{0ull, 0ull};
        for (unsigned ct = 0; ct < 4; ++ct)
            for (unsigned cw = 0; cw < 4; ++cw)
                u128_add_shift(X, u64(((a >> (8u * ct)) & 0xFFu) * ((b >> (8u * cw)) & 0xFFu)),
                               8u * (ct + cw));
        if (X.hi != 0 || X.lo != u64(a) * b) return false;
    }
    return true;
}

}  // namespace crt_mma_fast

#undef CRTMF_FHD
