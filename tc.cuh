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
#ifndef TC22_MERGED
#define TC22_MERGED 1   // merged-psi negacyclic NTT: no twist / untwist tables
#endif
// low-stage twiddle table: TC22 stores per stage (half-1)+j (511 values);
// the classic table stores omega^(j*N/(2half)) at j<<(8-m) (256 values).
#define TC_LT (TC22_MERGED ? 512u : 256u)
#define TWI(m, j) (TC22_MERGED ? ((1u << (m)) - 1u + (j)) : ((j) << (8 - (m))))
#ifndef TC20_R8
#define TC20_R8 1
#endif
// TC28: encode_fwdlow runs 1 CTA/SM (register-bound), so the bit-reversed A
// gather (~24% of the kernel) is never overlapped.  Each CTA prefetches into
// L2 the A rows of the CTA one wave ahead (linear id + #SMs); its own gather
// then hits L2 instead of DRAM.  No registers or shared used.
#ifndef TC28_PFA
#define TC28_PFA 1
#endif
// OC28 (parallel branch, agent A): conflict-free sT row stride in
// encode_fwdlow256's epilogue.  With the original stride E2_PT = 264 (== 8 mod
// 32) a warp writing q=0 and q=1 rows hits each shared bank twice (padw(rl+8)
// == padw(rl)+8), costing one extra shared transaction on every epilogue
// write.  Padding rows to 272 (== 16 mod 32) makes padw(rl+8) == padw(rl)+17
// -- an odd bank offset -- so the two primes' rows land in disjoint banks.
// Row contents / padw(pos) / DIT / store indices are unchanged; only the
// spacing between rows changes.  Flag 0 restores the original stride.
#ifndef OC28_EPICONF
#define OC28_EPICONF 1
#endif
#ifndef OC29_DITSHFL
#define OC29_DITSHFL 0
#endif
// OC29 (parallel branch, agent A): DIT entirely in registers.  One warp owns
// one prime; the 256-point row is held as 8 values per lane at position
// p = lane + 32*e.  Stages 0..4 are lane-xor shfl (partner p^(1<<s)), stages
// 5..7 are an in-thread radix-8 over e (positions differ by 32, 64, 128).
// sT is only the epilogue->DIT handoff (one barrier) and, at the end, the
// DIT->store coalesced write; no shared round trips or barriers between DIT
// stages.  Flag 0 restores the original 3-pass / 3-barrier shared DIT.
#ifndef TC16_R4
#define TC16_R4 1
#endif
#ifndef TC12_R4
#define TC12_R4 1
#endif
// OC26 (parallel branch): carry propagation in crt_tc2 phase 4.  The current
// path runs the NT*2-word carry chain on one thread per row (32 of 128 active);
// this variant splits each row's chain into four contiguous segments handled
// by four threads and has one owner thread propagate the four segment carries
// through a separate digit buffer in the dead sA region.  Flag 0 is a bit-exact
// copy of the original path.  No change to shared size (sA is dead here) and
// no inline asm / shfl, so the OC23 ptxas and OC13 mask pitfalls do not apply.
#ifndef OC26_CARRY
#define OC26_CARRY 1
#endif

// OC27 (agent B): 64 rows per CTA in the crt_tc2 slot (kernel crt_tc2_r64).
// Each B fragment feeds 4 m-tiles instead of 2, halving the per-run global/L2
// B traffic (~28 GB -> ~14 GB: NT*KS*32*8 B per CTA, CTA count 262144 ->
// 131072).  Shared budget (combat KS=15, NT=28): sA keeps the crt_tc2 stride
// SA=132 for all 64 rows (33792 B; %32==4 keeps the u32 A loads conflict-free
// and every row base 16 B aligned); sX holds one half of the n-tiles at a
// time (64*SXH*8 = 14848 B, SXH=29 odd; u64 columns 0..27 = the half's 14
// n-tiles, column 28 unused pad).  The GEMM runs twice (half 0: nt = warp+4u
// and warp+14+4u, u=0..2; half 1: nt = warp+28+4u, u=0..2); each pass ends
// with the OC26 4-thread carry over its columns (two 32-row waves), digits
// reduced and stored to global out inside the carry phase.  The u64 carry
// between halves plus the carry scratch live in the dead sA region:
// cinv 512 B + cout 1024 B + digit window 32*58*4 = 7424 B = 8960 B <= the
// 33792 B sA region.  Total 48640 B <= 49152 -> 2 CTA/SM without
// cudaFuncSetAttribute.  Flag 0 keeps the exact crt_tc2 path.
#ifndef OC27_ROWS64
#define OC27_ROWS64 0
#endif
#if OC27_ROWS64 && !OC26_CARRY
#error "OC27_ROWS64 requires OC26_CARRY=1 (r64 carry phase builds on it)"
#endif
constexpr unsigned CRT2_R64_ROWS = 64;
// u64 columns per sX half: odd (odd u64 row stride keeps the 2-way bank
// split of the epilogue stores), >= NT_HALF*2 where NT_HALF = (NT+1)/2.
inline unsigned crt2_sxh(unsigned NT) { return ((NT + 1) / 2) * 2 + 1; }
inline std::size_t crt_tc2_r64_smem(unsigned SA, unsigned SXH) {
    return ((std::size_t(CRT2_R64_ROWS) * SA * 4 + 15) & ~std::size_t(15)) +
           std::size_t(CRT2_R64_ROWS) * SXH * 8;
}
// Host-side eligibility: the r64 carry split needs w0 = 2*NTH <= l (half 0
// stores low digits only) and the dead-sA scratch must fit:
// cinv 512 + cout 1024 + digit window 32*DSH*4 bytes <= 64*SA*4.
inline bool crt2_r64_ok(unsigned KS, unsigned NT, unsigned SA, unsigned l) {
    const unsigned SXH = crt2_sxh(NT);
    const unsigned NTH = (SXH - 1) / 2;
    if (2 * NTH > l) return false;              // half-1 window V fits l2
    if (2 * SXH < l + 2) return false;          // shifted window covers l2
    if (KS > CRT_KS_MAX || NT > 2 * NTH) return false;
    const std::size_t need = 512 + 1024 + std::size_t(32) * (2 * SXH) * 4;
    return need <= std::size_t(CRT2_R64_ROWS) * SA * 4 &&
           crt_tc2_r64_smem(SA, SXH) <= 48 * 1024;
}

// OC34 (agent B): cp.async double-buffered residue staging in crt_tc2_r64.
// The staging phase (~34% of CRT) reads residues synchronously, then mounts
// them; TC30 showed the same overlap trick in encode is worth -1.4 ms.  Here
// the residues of prime batch b+1 are copied by cp.async (4 B per thread,
// zero-fill src-size for out-of-range primes/rows) into a second shared
// buffer while batch b's preweight/Montgomery + sA stores run.  Only the
// TC11 constant path with OC31 tile-major + OC32 preweight + merged slot
// order is pipelined (the combat config); every other config keeps the
// exact synchronous staging.
//
// Shared budget (exact): r64 base = sA 33792 + sX 14848 = 48640 B; headroom
// to 49152 is only 512 B, so a full second residue buffer cannot be added.
// Instead the two staging buffers are ALIASED ONTO THE DEAD sX REGION during
// staging (sX is first written by the GEMM epilogue, which starts after
// staging's final wait_group + __syncthreads):
//   buf[2][CPB][64] u32 = 2 * CPB * 64 * 4 B;  CPB=16 -> 8192 B <= 14848 B.
// ZERO extra shared bytes: total stays 48640 B <= 49152, 2 CTA/SM.
// Batch layout: batch b covers primes [b*CPB, (b+1)*CPB); thread tid copies
// prime b*CPB + tid/64, row tid%64 (one 4 B cp.async per thread per batch;
// 128 threads = 2 primes x 64 rows per issue round, CPB/2 = 8 rounds).
// K=114 -> 8 batches (last has 2 valid primes, zero-filled).
#ifndef OC34_CPA
#define OC34_CPA 1
#endif
constexpr unsigned OC34_CPB = 16;   // primes per cp.async batch/buffer
__host__ __device__ inline std::size_t oc34_buf_bytes() {
    return std::size_t(2) * OC34_CPB * CRT2_R64_ROWS * 4;   // 8192 at CPB=16
}
// eligible when the aliased buffers fit the sX region.  The CONFIG clauses
// (OC31_TILE_RC/OC32_PREWEIGHT/TC22_MERGED flags, ntiles, slot_logN, K) are
// checked at the call site inside the kernel — those macros live in the
// device section below, so this host helper only does the size math.
__host__ __device__ inline bool oc34_ok(unsigned SXH, unsigned K, unsigned ntiles, unsigned slot_logN) {
    return ntiles != 0 && slot_logN != 0 && K <= 128 &&   // 128 = CRT_KMAX_C
           oc34_buf_bytes() <= std::size_t(CRT2_R64_ROWS) * SXH * 8;
}

// OC8 (parallel branch): radix-8 high stages in fused_high - three DIT/DIF
// stages per shared round trip (s, s+1, s+2), plus radix-2 tail stages.
// CPU-validated bit-exact against the radix-2 reference (20 random trials).
#ifndef OC8_R8
#define OC8_R8 1
#endif

// OC10 (parallel branch): radix-16 high stages in fused_high - four DIT/DIF
// stages per shared round trip.  For logH=7 this turns 3 shared round trips
// per direction (8+8+2 in OC8) into 2 (16+8), cutting shared traffic by 1/3
// and twiddle Montgomery products from 7 per 8 butterflies to 15 per 32.
// The trailing stages (logH % 4) use the existing radix-8 / radix-2 paths.
// CPU-validated bit-exact against the naive negacyclic product for
// N = 2^11..2^14 (odd and even logH) by tools/tc_emul/radix16_emul.cpp.
#ifndef OC10_R16
#define OC10_R16 1
#endif

// OC11 (parallel branch): fuse the pointwise product into the last radix-8
// DIT pass and the first radix-8 DIF pass of fused_high.  After the final DIT
// butterfly the 8 group values of both operands are already in registers, so
// the product and the whole inverse radix-8 group can be done there too.  This
// removes two shared round trips and two barriers (7 -> 5 for logH=7 with
// OC10_R16).  Enabled only when the stage partition has a radix-8 group at
// the top (n8>0) and no radix-2 tail (s8==logH), i.e. the combat config
// logH=7; other configs use the unchanged path.  Flag OC11_FUSE=0 restores it.
// CPU-validated bit-exact by tools/tc_emul/fused_pw_emul.cpp.
#ifndef OC11_FUSE
#define OC11_FUSE 1
#endif

// OC12 (parallel branch): keep the boundary data in registers.  The first
// radix-16 DIT group reads ra/rb straight from global (no shared write + read
// + barrier), and the last radix-16 DIF group writes rc straight to global
// (no shared write + read).  With OC10+OC11 this leaves only two shared round
// trips in fused_high instead of seven in the original OC8 version.  Active
// only when n16>0; flag OC12_IO=0 restores the original load/store loops.
// CPU-validated bit-exact by tools/tc_emul/io_fused_emul.cpp.
// TC29: encode works on 256-row tiles (8 low DIT stages) with 256 threads,
// 2 CTAs/SM, so one CTA's barriers/DIT overlap the other's tensor-core MMA.
// The 9th low stage (m=8, pairs low j and j+256) moves into fused_high:
// lanes 0-15 take columns c, lanes 16-31 columns c+256 and the butterfly is
// one shfl_xor(16) right after the global load.  Runtime-gated (ef256 arg).
// TC30: cp.async double buffer for sR/sB in encode_fwdlow256.
// TC31: fused_high shared row stride 33 -> 32.  Every access is row*S+lane
// (one row per warp), so stride 32 is already conflict-free; smem 33.8 ->
// 32.8 KB per CTA lets 3 CTAs fit per SM instead of 2.
#ifndef TC31_S32
#define TC31_S32 1
#endif
// TC32: primes/weights of the group by cp.async into shared (encode256).
// TC33: fused_high __launch_bounds__(256,3): cap registers at 85 so the 3
// CTAs/SM that TC31's 32.8 KB smem allows are not blocked by registers.
#ifndef TC33_FH3
#define TC33_FH3 1
#endif
#ifndef TC32_PW
#define TC32_PW 1
#endif
#ifndef TC30_CPA
#define TC30_CPA 1
#endif
#ifndef TC29_EF256
#define TC29_EF256 1
#endif
// OC31: inv_low writes its residues in 512-tile-major order into the (dead)
// pra buffer; crt_tc2 staging then reads 114 primes 2 KB apart instead of
// 256 KB apart.  Same bytes, fewer TLB/L2 page touches.  0 = exact old path.
#ifndef OC31_TILE_RC
#define OC31_TILE_RC 1
#endif
// OC32: apply the per-prime CRT constant (n^-1*inv) inside inv_low, after
// the inverse NTT.  Montgomery-by-constant commutes with the linear inverse
// NTT, so crt_tc2 can use the residue as the term directly and drops one
// Montgomery per (prime, coefficient).  inv_low is DRAM-bound, so the extra
// arithmetic there is nearly free.  0 = exact old path.
#ifndef OC32_PREWEIGHT
#define OC32_PREWEIGHT 1
#endif
#ifndef OC12_IO
#define OC12_IO 1
#endif
#ifndef TC_PROF
#define TC_PROF 1
#endif
__device__ unsigned long long g_tcprof[16];
constexpr unsigned CRT_KMAX_C = 128;
__constant__ gpu31::Prime c_crt_prime[CRT_KMAX_C];
__constant__ u32 c_crt_corr[CRT_KMAX_C];
__constant__ u32 c_crt_slotc[CRT_KMAX_C];   // TC23: merged-NTT per-prime n^-1*inv
#define TCP_T(v) const long long v = (TC_PROF && threadIdx.x == 0 && threadIdx.y == 0) ? (long long)clock64() : 0
#define TCP_ADD(slot, a, b) do { if (TC_PROF && threadIdx.x == 0 && threadIdx.y == 0) \
    atomicAdd(&g_tcprof[slot], (unsigned long long)((b) - (a))); } while (0)

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
            tw[j] = (rtwist && k < K && rowok) ? rtwist[u64(k) * n + i] : 0u;
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
                dst[rbase + u64(k) * n + i] = rtwist ? gpu31::mont(v, tw[j], p) : v;
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
                u[q] = (TC22_MERGED && slot_logN) ? 0u : ((j < K && active) ? untw_inv[u64(j) * n + i] : 0u);
            }
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + 2 * q;
                if (j < K) {
                    const u32 term = gpu31::mont(r[q], (TC22_MERGED && slot_logN) ? c_crt_slotc[j] : u[q], c_crt_prime[j]);
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

// OC26: parallel carry propagation helper (device-only).  The four cooperate
// threads of a row each scan one contiguous segment of u64 addends into a
// separate digit buffer (in the dead sA region); a row owner then propagates
// the four segment carries.  cout[] holds the u64 outgoing carry of each
// segment; bit 63 is set when a segment contains a huge (>= 2^48) addend,
// in which case the owner replays the whole chain exactly (u64 wrap in the
// local running sum would otherwise lose high bits).  All barriers are
// uniform: every one of the 128 threads reaches both __syncthreads.
#if OC26_CARRY
__device__ __forceinline__ void oc26_carry_row(
        const u64* __restrict__ xs, u32* __restrict__ dg, u64* __restrict__ cout,
        unsigned words, unsigned l2) {
    const unsigned seg = threadIdx.x & 3;           // segment 0..3
    const unsigned row = threadIdx.x >> 2;          // owner row 0..31
    const unsigned base = words >> 2, rem = words & 3;
    const unsigned a0 = seg * base + (seg < rem ? seg : rem);
    const unsigned len = base + (seg < rem ? 1u : 0u);
    u64 run = 0, big = 0;
    for (unsigned j = 0; j < len; ++j) {
        const u64 x = xs[a0 + j];
        const u64 t = run + x;
        dg[a0 + j] = u32(t);
        run = t >> 32;
        big |= x >> 48;
    }
    cout[row * 4 + seg] = run | (big ? (u64(1) << 63) : u64(0));
    __syncthreads();
    if (seg != 0) return;                          // seg 0 owns its row
    u64 cin = 0;
    bool replay = false;
    for (unsigned s = 0; s < 4; ++s) replay |= (cout[row * 4 + s] >> 63) != 0;
    if (replay) {
        // Rare exact fallback: extreme u64 addends where the local "carry"
        // recurrence is not a small integer.  Serial but correct; length is
        // only `words` (56 in the production config).
        u64 carry = 0;
        for (unsigned w = 0; w < words; ++w) {
            carry += xs[w];
            dg[w] = u32(carry);
            carry >>= 32;
        }
        cin = carry;
    } else {
        for (unsigned s = 0; s < 4; ++s) {
            const unsigned a = s * base + (s < rem ? s : rem);
            u64 delta = cin;
            if (delta) {
                for (unsigned j = 0; j < base + (s < rem ? 1u : 0u) && delta; ++j) {
                    const u64 sum = u64(dg[a + j]) + u32(delta);
                    dg[a + j] = u32(sum);
                    delta = (delta >> 32) + (sum >> 32);
                }
            }
            cin = cout[row * 4 + s] + delta;
        }
    }
    for (unsigned w = words; w < l2; ++w) {
        dg[w] = u32(cin);
        cin >>= 32;
    }
}
#endif // OC26_CARRY

#if OC27_ROWS64
// OC27: the OC26 carry over 64 rows for one n-tile half, in two 32-row
// waves (128 threads = 4 per row).  Segment logic is oc26_carry_row's:
// parallel local scans into a digit window, owner propagates the running
// delta with early exit, exact serial replay when a segment saw an extreme
// addend (>= 2^48).  All barriers unconditional.
//
// Seam arithmetic.  Full addend stream = A (columns 0..w0-1) then B
// (columns w0..w0+w1-1); cin0 = sumA >> 32*w0 is half 0's outgoing u64
// carry.  The single chain's digit at global column w0+j equals
//   u32( localB_j + (cin0 >> 32*j) )   with upward carry propagation,
// where localB is half 1's zero-carry chain: folding the little-endian
// u64 cin0 into half 1's digit window at offset 0 (multi-word ripple,
// early exit) reproduces the continuation exactly, for any w0.
//
// Store split — half 0 keeps the LOW digits only, half 1 carries the seam:
//  - half 0: tail dg[words..l2) = 0 (NOT cin0), reduce_sum<1>, store l
//    words.  With w0 = words <= l (combat 28 <= 55) the window's nonzero
//    part is exactly L_A = sumA mod 2^(32*w0) < 2^896 << q, so the reduce
//    is a no-op and out = L_A exactly (no mod-q fold of cin0 happens).
//    cinv[row] is WRITTEN here before half 1 ever reads it (barrier-
//    separated), so no zero-init pass is needed.
//  - half 1: tail = local overflow carry, fold cin0 at offset 0 (ripple):
//    dg[0..l2) = V = sumB + cin0 (needs words+2 <= l2, i.e. words <= l —
//    host guard).  Shift V UP by w0 columns inside the DSH-word window
//    (top index w0+words+1 <= DSH-1, nothing dropped), add out = L_A with
//    ripple: dg[0..l2) = L_A + V*2^(32*w0) mod 2^(32*l2) = total mod
//    2^(32*l2) — bit-identical to the single chain's window (words >= l2
//    are discarded there too).  One reduce_sum<1> canonicalises; store.
// Exactness: L_A + (sumB + cin0)*2^(32*w0) = sumA + sumB = total, since
// sumA = L_A + cin0*2^(32*w0).  No intermediate mod-q reduce of V (it
// would still be correct mod q, but the direct window equality is
// stronger and cheaper).  Configs with words > l are rejected by the
// host (crt2_r64_ok).  Each out row is touched only by this CTA's owner
// thread, in order, barrier-separated.
//
// Layout of the dead sA region during this phase (u64 units):
//   cinv[0..63]    handover carries (persists across halves)
//   cout[0..127]   per-wave segment carries + big flags (tid-indexed)
//   dg[32][DSH]    per-wave digit window, DSH = 2*SXH u32 words per row
__device__ __forceinline__ void oc27_carry_half(
        const u64* __restrict__ xs, unsigned SXH, u64* __restrict__ cinv,
        u64* __restrict__ cout, u32* __restrict__ dgwin, unsigned DSH,
        u32* __restrict__ out, u64 obase,
        unsigned rows, unsigned l, unsigned words, unsigned slot_logN,
        unsigned start, unsigned tid, unsigned half, unsigned w0,
        unsigned bits, u32 gap, const u32* __restrict__ q) {
    const unsigned seg = tid & 3;
    const unsigned rw = tid >> 2;                    // row within the wave
    const unsigned base = words >> 2, rem = words & 3;
    const unsigned a0 = seg * base + (seg < rem ? seg : rem);
    const unsigned len = base + (seg < rem ? 1u : 0u);
    const unsigned l2 = l + 2;
    for (unsigned wave = 0; wave < 2; ++wave) {
        const unsigned row = wave * 32 + rw;         // 0..63
        const bool live = row < rows;
        u32* dg = dgwin + rw * DSH;
        u64 run = 0, big = 0;
        if (live) {
            const u64* x = xs + row * SXH;
            for (unsigned j = 0; j < len; ++j) {
                const u64 v = x[a0 + j];
                const u64 t = run + v;
                dg[a0 + j] = u32(t);
                run = t >> 32;
                big |= v >> 48;
            }
        }
        cout[tid] = run | (big ? (u64(1) << 63) : u64(0));
        __syncthreads();
        if (live && seg == 0) {
            const u64* x = xs + row * SXH;
            u64 cin = 0;
            bool replay = false;
            for (unsigned s = 0; s < 4; ++s)
                replay |= (cout[rw * 4 + s] >> 63) != 0;
            if (replay) {
                u64 carry = 0;
                for (unsigned w = 0; w < words; ++w) {
                    carry += x[w];
                    dg[w] = u32(carry);
                    carry >>= 32;
                }
                cin = carry;
            } else {
                for (unsigned s = 0; s < 4; ++s) {
                    const unsigned a = s * base + (s < rem ? s : rem);
                    u64 delta = cin;
                    if (delta) {
                        for (unsigned j = 0; j < base + (s < rem ? 1u : 0u) && delta; ++j) {
                            const u64 sum = u64(dg[a + j]) + u32(delta);
                            dg[a + j] = u32(sum);
                            delta = (delta >> 32) + (sum >> 32);
                        }
                    }
                    cin = cout[rw * 4 + s] + delta;
                }
            }
            u64 ob;
            if (slot_logN == 0) {
                ob = obase + u64(row) * l;
            } else {
                const unsigned pos = __brev(start + row) >> (32 - slot_logN);
                ob = obase + u64(pos) * l;
            }
            if (half == 0) {
                cinv[row] = cin;                     // handover to half 1
                // LOW digits only: zero the tail so cin0 is not folded mod
                // q here (half 1's shifted add carries it exactly once).
                for (unsigned w = words; w < l2; ++w) dg[w] = 0;
                gpu31::reduce_sum<1>(dg, l, bits, gap, q);
                for (unsigned cc = 0; cc < l; ++cc) out[ob + cc] = dg[cc];
            } else {
                // tail = local overflow carry, then fold cin0 at offset 0:
                // dg[0..l2) becomes V = sumB + cin0 (word count <= words+2
                // <= l+2 by the host guard words <= l).
                u64 tail = cin;
                for (unsigned w = words; w < l2; ++w) {
                    dg[w] = u32(tail);
                    tail >>= 32;
                }
                u64 add = cinv[row];
                for (unsigned w = 0; w < l2 && add; ++w) {
                    const u64 sum = u64(dg[w]) + u32(add);
                    dg[w] = u32(sum);
                    add = (add >> 32) + (sum >> 32);
                }
                // shift V UP by w0 columns in place (high to low, DSH-word
                // window; V's top word index <= words+1, so w0+words+1 <=
                // 2*words+1 = DSH-1: nothing dropped).  Words >= l2 of the
                // old window are stale — the shift overwrites every slot.
                for (unsigned j = DSH; j-- > w0;)
                    dg[j] = (j - w0 < l2) ? dg[j - w0] : 0u;
                for (unsigned j = 0; j < w0; ++j) dg[j] = 0;
                // add out (= L_A, half 0's low digits) with ripple
                u64 carry = 0;
                for (unsigned cc = 0; cc < l; ++cc) {
                    const u64 v = u64(dg[cc]) + u64(out[ob + cc]) + carry;
                    dg[cc] = u32(v);
                    carry = v >> 32;
                }
                for (unsigned w = l; w < DSH && carry; ++w) {
                    const u64 v = u64(dg[w]) + carry;
                    dg[w] = u32(v);
                    carry = v >> 32;
                }
                // dg[0..l2) now equals the single chain's 57-word window
                // (total mod 2^(32*l2)); words >= l2 are ignored by the
                // reduce, exactly like crt_tc2's discarded high carry.
                gpu31::reduce_sum<1>(dg, l, bits, gap, q);
                for (unsigned cc = 0; cc < l; ++cc) out[ob + cc] = dg[cc];
            }
        }
        __syncthreads();
    }
}
#endif // OC27_ROWS64

// TC15 CRT variant: 32 rows/CTA, warps split n-tiles, B reused by 2 m-tiles.
__global__ void __launch_bounds__(128, 3)
crt_tc2(gpu31::CrtArgs pl, const u32* __restrict__ residues, u32* __restrict__ out,
       const uint2* __restrict__ frag, CrtTc cfg,
       const gpu31::Prime* __restrict__ primes, const u32* __restrict__ untw_inv,
       unsigned slot_logN, const u32* __restrict__ corr_m, unsigned ntiles = 0) {
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
    // OC31: tile-major rc written by inv_low into the dead pra buffer.
    // CRT2_ROWS (32) divides NT_TILE (512), so a CTA never straddles tiles.
    const u64 rbase2 = (OC31_TILE_RC && ntiles)
        ? (u64(pair) * ntiles + (start >> 9)) * K : 0ull;
    auto RC = [&](unsigned j, unsigned coeff) -> u32 {
        return (OC31_TILE_RC && ntiles)
            ? residues[(rbase2 + j) * 512u + (coeff & 511u)]
            : residues[rbase + u64(j) * n + coeff];
    };
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
                r[q] = (j < K && active) ? RC(j, i) : 0u;
                u[q] = (TC22_MERGED && slot_logN) ? 0u : ((j < K && active) ? untw_inv[u64(j) * n + i] : 0u);
            }
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + NG * q;
                if (j < K) {
                    const u32 term = (OC32_PREWEIGHT && TC22_MERGED && slot_logN)
                        ? r[q]
                        : gpu31::mont(r[q], (TC22_MERGED && slot_logN) ? c_crt_slotc[j] : u[q], c_crt_prime[j]);
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
            const u32 r = RC(j, i);
            if (OC32_PREWEIGHT && TC22_MERGED && slot_logN && untw_inv)
                term = r;
            else if (untw_inv)
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
    // 4. carry + reduce.  OC26_CARRY=0: exact original path (one thread per
    // row, digits in place over sX).  OC26_CARRY=1: four threads per row,
    // separate digit buffer in the dead sA region, owner propagates carries.
    const unsigned DS = 2 * cfg.SX;
#if OC26_CARRY
    u32* digits = reinterpret_cast<u32*>(sA);   // separate from xs (no race)
    {
        const unsigned words = cfg.NT * 2;
        // sA is 32*SA*4 bytes; digits need 32*DS*4 = 32*2*SX*4 bytes.  With
        // SA >= 2*SX in all configs (SA=124, SX=57), this fits; cout[] lives
        // in the tail of the same sA region beyond the digit buffer.
        u64* cout = reinterpret_cast<u64*>(digits + CRT2_ROWS * DS);
        const unsigned row = tid >> 2;
        oc26_carry_row(sX + row * cfg.SX, digits + row * DS, cout,
                       words, l + 2);
        // oc26_carry_row contains the only __syncthreads of this phase; it is
        // reached by all 128 threads, so the barrier is uniform.
        if (row < CRT2_ROWS && (tid & 3) == 0) {   // one reducer per row
            u32* dg = digits + row * DS;
            gpu31::reduce_sum<1>(dg, l, pl.bits, pl.gap, pl.q);
        }
    }
    __syncthreads();
#else
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
#endif

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

#if OC27_ROWS64
// OC27 CRT variant: 64 rows/CTA, warps split n-tiles, B reused by 4 m-tiles.
// Two n-half passes share the same sX (SXH u64 columns); carry+reduce+store
// run per half via oc27_carry_half (two 32-row waves; the digit window never
// exceeds 32 rows).  launch_bounds(128,2): 2 CTA/SM is the smem ceiling
// anyway (48640 B each at KS=15/NT=28); the 256-reg cap leaves acc[4][2][4]
// = 32 regs + staging 16 + working set far below the limit.  All
// __syncthreads are unconditional.  Profiling: slots keep their crt_tc2
// meaning (0 staging, 1 setup, 2 mma+epilogue summed over both halves,
// 3 carry+reduce+store summed over both halves, 4 unused: the store moved
// into slot 3).
__global__ void __launch_bounds__(128, 2)
crt_tc2_r64(gpu31::CrtArgs pl, const u32* __restrict__ residues, u32* __restrict__ out,
       const uint2* __restrict__ frag, CrtTc cfg,
       const gpu31::Prime* __restrict__ primes, const u32* __restrict__ untw_inv,
       unsigned slot_logN, const u32* __restrict__ corr_m, unsigned ntiles = 0) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    u32* sA = reinterpret_cast<u32*>(smem_raw);
    const std::size_t abytes =
        (std::size_t(CRT2_R64_ROWS) * cfg.SA * 4 + 15) & ~std::size_t(15);
    u64* sX = reinterpret_cast<u64*>(smem_raw + abytes);
    constexpr unsigned NG = 128 / CRT2_R64_ROWS;   // = 2 staging groups/row

    const unsigned n = pl.n, l = pl.limbs, K = pl.count;
    const unsigned pair = blockIdx.z, start = blockIdx.x * CRT2_R64_ROWS;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid & (CRT2_R64_ROWS - 1), jg = tid / CRT2_R64_ROWS;
    const unsigned i = start + c;
    const bool active = i < n;
    const u64 rbase = u64(pair) * K * n;
    // OC31: tile-major rc written by inv_low into the dead pra buffer.
    // CRT2_R64_ROWS (64) divides NT_TILE (512), so a CTA never straddles
    // tiles — same RC lambda as crt_tc2.
    const u64 rbase2 = (OC31_TILE_RC && ntiles)
        ? (u64(pair) * ntiles + (start >> 9)) * K : 0ull;
    auto RC = [&](unsigned j, unsigned coeff) -> u32 {
        return (OC31_TILE_RC && ntiles)
            ? residues[(rbase2 + j) * 512u + (coeff & 511u)]
            : residues[rbase + u64(j) * n + coeff];
    };
    const unsigned W = cfg.KS * 8;
    const unsigned SXH = cfg.SX;    // host passes crt2_sxh(NT) in cfg.SX
    const unsigned NTH = (SXH - 1) / 2;            // n-tiles per half (14)
    const unsigned UP = (NTH + 3) / 4;             // u iterations per warp (4)

    TCP_T(t0);
    // 1. staging — identical to crt_tc2 (TC11 batched path + OC31 RC +
    // OC32 preweight), NG=2: each thread handles K/2 primes; warp lanes =
    // 32 consecutive rows.
    u64 fsum = 0;
#if OC34_CPA
    // OC34: cp.async double-buffered residue staging for the combat config
    // (constant primes + tile-major rc + preweight + merged slots).  The
    // buffers alias the dead sX region (see oc34_ok); batch b+1's residues
    // fly in while batch b is mounted into sA.  term = residue (preweight),
    // so the compute per batch is just the fsum multiply-add + sA stores.
    if (OC31_TILE_RC && OC32_PREWEIGHT && TC22_MERGED &&
        oc34_ok(SXH, K, ntiles, slot_logN) && untw_inv) {
        u32* rbuf = reinterpret_cast<u32*>(sX);      // [2][CPB][64]
        const unsigned NB = (K + OC34_CPB - 1) / OC34_CPB;
        const unsigned cq = tid >> 6, cr = tid & 63; // prime-in-pair, row
        const bool ract = (start + cr) < n;
        auto issue = [&](unsigned b, unsigned buf) {
            u32* dst = rbuf + buf * (OC34_CPB * CRT2_R64_ROWS);
#pragma unroll
            for (unsigned p = 0; p < OC34_CPB / 2; ++p) {
                const unsigned j = b * OC34_CPB + p * 2 + cq;
                const bool ok = j < K && ract;
                const u32* gp = residues + (ok ? (rbase2 + j) * 512u + ((start + cr) & 511u) : 0);
                const unsigned sa = unsigned(__cvta_generic_to_shared(dst + (p * 2 + cq) * CRT2_R64_ROWS + cr));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;"
                             :: "r"(sa), "l"(gp), "r"(ok ? 4 : 0) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        };
        issue(0, 0);
        for (unsigned b = 0; b < NB; ++b) {
            const unsigned buf = b & 1;
            if (b + 1 < NB) issue(b + 1, buf ^ 1);
            // wait until only the just-issued next batch is in flight
            if (b + 1 < NB) asm volatile("cp.async.wait_group 1;" ::: "memory");
            else            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            const u32* src = rbuf + buf * (OC34_CPB * CRT2_R64_ROWS);
            // each thread mounts its NG=2 prime streams for row c
#pragma unroll 1
            for (unsigned p = 0; p < OC34_CPB; ++p) {
                const unsigned j = b * OC34_CPB + p;
                if (j < K && j % NG == jg) {
                    const u32 term = src[p * CRT2_R64_ROWS + c];
                    sA[c * cfg.SA + j] = term;
                    fsum += (u64(term) * c_crt_corr[j]) >> 30;
                }
            }
            __syncthreads();   // all reads of buf done before it is reused
        }
    } else
#endif
    if (untw_inv && K <= CRT_KMAX_C) {
#pragma unroll 1
        for (unsigned j0 = jg; j0 < K; j0 += 8 * NG) {
            u32 r[8], u[8];
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + NG * q;
                r[q] = (j < K && active) ? RC(j, i) : 0u;
                u[q] = (TC22_MERGED && slot_logN) ? 0u : ((j < K && active) ? untw_inv[u64(j) * n + i] : 0u);
            }
#pragma unroll
            for (unsigned q = 0; q < 8; ++q) {
                const unsigned j = j0 + NG * q;
                if (j < K) {
                    const u32 term = (OC32_PREWEIGHT && TC22_MERGED && slot_logN)
                        ? r[q]
                        : gpu31::mont(r[q], (TC22_MERGED && slot_logN) ? c_crt_slotc[j] : u[q], c_crt_prime[j]);
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
            const u32 r = RC(j, i);
            if (OC32_PREWEIGHT && TC22_MERGED && slot_logN && untw_inv)
                term = r;
            else if (untw_inv)
                term = gpu31::mont(r, untw_inv[u64(j) * n + i], primes[j]);
            else
                term = gpu31::weight_residue(r, pl.primes[j]);
        }
        sA[c * cfg.SA + j] = term;
        fsum += (u64(term) * corr_m[j]) >> 30;
    }
    }
    for (unsigned j = K + 1 + jg; j < W; j += NG) sA[c * cfg.SA + j] = 0;
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
    const unsigned warp = tid >> 5, lane = tid & 31;
    const unsigned g = lane >> 2, t = lane & 3;
    const bool odd = t & 1;
    const unsigned sh = (t & 1) * 16;
    // dead-sA scratch for the carry phase (layout: see oc27_carry_half)
    u64* cinv = reinterpret_cast<u64*>(sA);          // 64 u64 = 512 B
    u64* cout = cinv + CRT2_R64_ROWS;                // 128 u64 = 1024 B
    u32* dgwin = reinterpret_cast<u32*>(cout + 128); // 32 rows * DSH u32
    const unsigned DSH = 2 * SXH;
    TCP_T(t2);
    const unsigned rows = (n - start < CRT2_R64_ROWS) ? (n - start) : CRT2_R64_ROWS;
    const u64 obase = u64(pair) * n * l;
    long long mma_cyc = 0, carry_cyc = 0;
    for (unsigned half = 0; half < 2; ++half) {
        const long long tm0 = (TC_PROF && tid == 0) ? (long long)clock64() : 0;
        // 2-3. GEMM over this half's n-tiles.  Warp w covers ntA = w+4u and
        // ntB = ntA+8 (u = 0..UP-1), filtered to the half's [0,NTH) range:
        // the union over warps and both streams is every tile exactly once
        // (NTH=14: warp0 {0,4,8,12}, warp1 {1,5,9,13}, warp2 {2,6,10},
        // warp3 {3,7,11}).
#pragma unroll 1
        for (unsigned u = 0; u < UP; ++u) {
            const unsigned ntA = half * NTH + warp + 4 * u;
            const unsigned ntB = ntA + 8;
            const bool hasA = (warp + 4 * u) < NTH && ntA < cfg.NT;
            const bool hasB = (warp + 4 * u + 8) < NTH && ntB < cfg.NT;
            if (!hasA && !hasB) continue;             // warp-uniform
            int acc[4][2][4];
#pragma unroll
            for (unsigned mt = 0; mt < 4; ++mt)
#pragma unroll
                for (unsigned q = 0; q < 2; ++q) acc[mt][q][0] = acc[mt][q][1] = acc[mt][q][2] = acc[mt][q][3] = 0;
#pragma unroll 3
            for (unsigned ks = 0; ks < cfg.KS; ++ks) {
                const uint2 b0 = hasA ? frag[(std::size_t(ntA) * cfg.KS + ks) * 32 + lane] : make_uint2(0u, 0u);
                const uint2 b1 = hasB ? frag[(std::size_t(ntB) * cfg.KS + ks) * 32 + lane] : make_uint2(0u, 0u);
#pragma unroll
                for (unsigned mt = 0; mt < 4; ++mt) {
                    const u32* r0 = sA + (mt * 16 + g) * cfg.SA + ks * 8 + t;
                    const u32* r1 = r0 + 8 * cfg.SA;
                    const u32 x0 = r0[0], x1 = r1[0], x2 = r0[4], x3 = r1[4];
                    if (hasA) mma_u8(acc[mt][0], x0, x1, x2, x3, b0.x, b0.y);
                    if (hasB) mma_u8(acc[mt][1], x0, x1, x2, x3, b1.x, b1.y);
                }
            }
#pragma unroll
            for (unsigned mt = 0; mt < 4; ++mt)
#pragma unroll
                for (unsigned q = 0; q < 2; ++q) {
                    const unsigned nt = q ? ntB : ntA;
                    if (!(q ? hasB : hasA)) continue;
                    const u64 pg = (u64(u32(acc[mt][q][0])) << sh) + (u64(u32(acc[mt][q][1])) << (sh + 8));
                    const u64 pg8 = (u64(u32(acc[mt][q][2])) << sh) + (u64(u32(acc[mt][q][3])) << (sh + 8));
                    const u64 send = odd ? pg : pg8;
                    const u64 mine = odd ? pg8 : pg;
                    const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
                    const unsigned row = mt * 16 + g + (odd ? 8 : 0);
                    sX[row * SXH + (nt - half * NTH) * 2 + (t >> 1)] = tot;
                }
        }
        __syncthreads();
        const long long tm1 = (TC_PROF && tid == 0) ? (long long)clock64() : 0;
        // 4. carry + reduce + store for this half (two 32-row waves).
        oc27_carry_half(sX, SXH, cinv, cout, dgwin, DSH, out, obase,
                        rows, l, NTH * 2, slot_logN, start, tid,
                        half, half * NTH * 2,
                        pl.bits, pl.gap, pl.q);
        const long long tm2 = (TC_PROF && tid == 0) ? (long long)clock64() : 0;
        mma_cyc += tm1 - tm0; carry_cyc += tm2 - tm1;
    }
    TCP_ADD(0, t0, t1); TCP_ADD(1, t1, t2);
    if (TC_PROF && tid == 0) {
        atomicAdd(&g_tcprof[2], (unsigned long long)mma_cyc);
        atomicAdd(&g_tcprof[3], (unsigned long long)carry_cyc);
    }
}
#endif // OC27_ROWS64


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
    __shared__ u32 sa[NT_TILE + NT_TILE / 32], sb[NT_TILE + NT_TILE / 32], sr[TC_LT];
    const unsigned t = threadIdx.x;
    const u64 base = (u64(blockIdx.z) * K + blockIdx.y) * n + u64(blockIdx.x) * NT_TILE;
    const gpu31::Prime p = primes[blockIdx.y];
    for (unsigned x = t; x < TC_LT; x += 256) sr[x] = packed[u64(blockIdx.y) * TC_LT + x];
    sa[padw(t)] = a[base + t];
    sa[padw(t + 256)] = a[base + t + 256];
    sb[padw(t)] = b[base + t];
    sb[padw(t + 256)] = b[base + t + 256];
    __syncthreads();
#pragma unroll
    for (unsigned m = 0; m < 9; ++m) {
        const unsigned half = 1u << m, j = t & (half - 1);
        const unsigned first = 2 * (t - j) + j, second = first + half;
        const u32 w = sr[TWI(m, j)];
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
__global__ void __launch_bounds__(256, TC33_FH3 ? 3 : 1)
fused_high(const u32* __restrict__ ra, const u32* __restrict__ rb, u32* __restrict__ rc,
           const gpu31::Prime* __restrict__ primes, const u32* __restrict__ twf,
           const u32* __restrict__ twi, unsigned n, unsigned K, unsigned logH,
           const u32* __restrict__ pkf = nullptr, unsigned ef256 = 0) {
    // twf/twi: per prime [7][512] "low" factors followed by [127] "within"
    // factors: w = mont(A[s][low], B[(2^s-1)+within]) = root[(within*512+low)<<d].
    extern __shared__ u32 sm[];
    constexpr unsigned S = TC31_S32 ? NT_COLS : NT_COLS + 1;   // all accesses are row*S+lane
    const unsigned HC = 1u << logH;
#if OC10_R16
    // Stage partition: n16 radix-16 groups (4 stages), n8 radix-8 groups
    // (3 stages) and n2 radix-2 stages, all covering [0, logH).
    const unsigned n16 = logH >> 2, rs = n16 << 2;
    const unsigned n8 = (logH - rs) / 3, s8 = rs + n8 * 3;
#if OC11_FUSE
    const bool oc11_fuse = (n8 > 0 && s8 == logH);
#endif
#endif
    u32* sa = sm;
    u32* sb = sm + HC * S;
    const unsigned lane = threadIdx.x, rw = threadIdx.y;
    const unsigned low = ef256 ? (blockIdx.x * 16 + (lane & 15) + ((lane & 16) << 4))
                               : blockIdx.x * NT_COLS + lane;
    // TC29: twiddle of low stage 8 for pair (j, j+256), j = low & 255
    const u32 w8 = ef256 ? pkf[u64(blockIdx.y) * TC_LT + TWI(8, low & 255)] : 0u;
    const unsigned k = blockIdx.y;
    const u64 plane = (u64(blockIdx.z) * K + k) * n;
    const gpu31::Prime p = primes[k];
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
#if OC10_R16 && OC12_IO
    if (n16 == 0) {
#endif
        for (unsigned h = rw; h < HC; h += 8) {
            sa[h * S + lane] = ra[plane + u64(h) * NT_TILE + low];
            sb[h * S + lane] = rb[plane + u64(h) * NT_TILE + low];
        }
        __syncthreads();
#if OC10_R16 && OC12_IO
    }
    // OC12: with n16>0 the first radix-16 DIT group loads ra/rb directly
    // into its register arrays, so no shared write/read/barrier is needed.
#endif
    TCP_T(f1);
#if OC10_R16
    // OC10: four DIT stages per shared round trip.  A group is rows
    // r0 + i*H (i=0..15, H=2^s).  Stage s pairs (i,i+1) with within w;
    // s+1 pairs (i,i+2) with within w+(i&1)*H; s+2 pairs (i,i+4) with
    // within w+(i&3)*H; s+3 pairs (i,i+8) with within w+(i&7)*H.
#pragma unroll
    for (unsigned q = 0; q < 4; ++q) {
        if (q >= n16) break;
        const unsigned s = q << 2, H = 1u << s;
        for (unsigned g = rw; g < HC / 16; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 16 + w;
            unsigned ix[16];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) ix[i] = (r0 + i * H) * S + lane;
            const u32 t1 = gpu31::mont(waf[s], tfB[(H - 1) + w], p);
            u32 t2[2], t4[4], t8[8];
#pragma unroll
            for (unsigned k = 0; k < 2; ++k) t2[k] = gpu31::mont(waf[s+1], tfB[(2 * H - 1) + w + k * H], p);
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) t4[k] = gpu31::mont(waf[s+2], tfB[(4 * H - 1) + w + k * H], p);
#pragma unroll
            for (unsigned k = 0; k < 8; ++k) t8[k] = gpu31::mont(waf[s+3], tfB[(8 * H - 1) + w + k * H], p);
            u32 av[16], bv[16];
#if OC10_R16 && OC12_IO
            if (q == 0) {
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) {
                    const unsigned row = r0 + i * H;
                    av[i] = ra[plane + u64(row) * NT_TILE + low];
                    bv[i] = rb[plane + u64(row) * NT_TILE + low];
                }
                if (ef256) {
                    // low stage 8 (DIT): lane j gets u + w*v, lane j+256 gets u - w*v
                    const bool up = (lane & 16) != 0;
#pragma unroll
                    for (unsigned i = 0; i < 16; ++i) {
                        const u32 xa = __shfl_xor_sync(0xffffffffu, av[i], 16);
                        const u32 xb = __shfl_xor_sync(0xffffffffu, bv[i], 16);
                        if (up) {
                            av[i] = gpu31::sub(xa, gpu31::mont(av[i], w8, p), p.p);
                            bv[i] = gpu31::sub(xb, gpu31::mont(bv[i], w8, p), p.p);
                        } else {
                            av[i] = gpu31::add(av[i], gpu31::mont(xa, w8, p), p.p);
                            bv[i] = gpu31::add(bv[i], gpu31::mont(xb, w8, p), p.p);
                        }
                    }
                }
            } else
#endif
            {
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) { av[i] = sa[ix[i]]; bv[i] = sb[ix[i]]; }
            }
#pragma unroll
            for (unsigned i = 0; i < 16; i += 2) {
                gpu31::butterfly(av[i], av[i+1], t1, p);
                gpu31::butterfly(bv[i], bv[i+1], t1, p);
            }
#pragma unroll
            for (unsigned k = 0; k < 2; ++k) {
#pragma unroll
                for (unsigned i = k; i < 16; i += 4) {
                    gpu31::butterfly(av[i], av[i+2], t2[k], p);
                    gpu31::butterfly(bv[i], bv[i+2], t2[k], p);
                }
            }
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) {
#pragma unroll
                for (unsigned i = k; i < 16; i += 8) {
                    gpu31::butterfly(av[i], av[i+4], t4[k], p);
                    gpu31::butterfly(bv[i], bv[i+4], t4[k], p);
                }
            }
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                gpu31::butterfly(av[i], av[i+8], t8[i], p);
                gpu31::butterfly(bv[i], bv[i+8], t8[i], p);
            }
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) { sa[ix[i]] = av[i]; sb[ix[i]] = bv[i]; }
        }
        __syncthreads();
    }
    // radix-8 for the next min(3, logH-rs) stages, then radix-2 tail.
#pragma unroll
    for (unsigned q = 0; q < 2; ++q) {
        if (q >= n8) break;
        const unsigned s = rs + q * 3, H = 1u << s;
        for (unsigned g = rw; g < HC / 8; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 8 + w;
            unsigned ix[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) ix[i] = (r0 + i * H) * S + lane;
            const u32 t1 = gpu31::mont(waf[s], tfB[(H - 1) + w], p);
            const u32 t2a = gpu31::mont(waf[s+1], tfB[(2 * H - 1) + w], p);
            const u32 t2b = gpu31::mont(waf[s+1], tfB[(2 * H - 1) + w + H], p);
            u32 t4[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                t4[i] = gpu31::mont(waf[s+2], tfB[(4 * H - 1) + w + i * H], p);
            u32 av[8], bv[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) { av[i] = sa[ix[i]]; bv[i] = sb[ix[i]]; }
#pragma unroll
            for (unsigned i = 0; i < 8; i += 2) {
                gpu31::butterfly(av[i], av[i+1], t1, p);
                gpu31::butterfly(bv[i], bv[i+1], t1, p);
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned i = (j < 2) ? j : j + 2;   // 0,1,4,5
                const u32 tw = (i & 1) ? t2b : t2a;
                gpu31::butterfly(av[i], av[i+2], tw, p);
                gpu31::butterfly(bv[i], bv[i+2], tw, p);
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                gpu31::butterfly(av[i], av[i+4], t4[i], p);
                gpu31::butterfly(bv[i], bv[i+4], t4[i], p);
            }
#if OC10_R16 && OC11_FUSE
            if (oc11_fuse && q + 1 == n8) {
                // OC11: 8 final DIT values of a and b are in registers.  Do the
                // pointwise product here and the whole inverse radix-8 group
                // (stages s+2,s+1,s) before the single shared write.  This is
                // the exact inverse of the three DIT butterflies above, so the
                // group's data never needs the separate pointwise/DIF passes.
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) av[i] = gpu31::mont(av[i], bv[i], p);
                u32 q4[4];
#pragma unroll
                for (unsigned i = 0; i < 4; ++i)
                    q4[i] = gpu31::mont(wai[s+2], tiB[(4 * H - 1) + w + i * H], p);
                const u32 q2a = gpu31::mont(wai[s+1], tiB[(2 * H - 1) + w], p);
                const u32 q2b = gpu31::mont(wai[s+1], tiB[(2 * H - 1) + w + H], p);
                const u32 q1 = gpu31::mont(wai[s], tiB[(H - 1) + w], p);
#pragma unroll
                for (unsigned i = 0; i < 4; ++i) {
                    const u32 u = av[i], v = av[i+4];
                    av[i] = gpu31::add(u, v, p.p);
                    av[i+4] = gpu31::mont(gpu31::sub(u, v, p.p), q4[i], p);
                }
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    const unsigned i = (j < 2) ? j : j + 2;   // 0,1,4,5
                    const u32 tw = (i & 1) ? q2b : q2a;
                    const u32 u = av[i], v = av[i+2];
                    av[i] = gpu31::add(u, v, p.p);
                    av[i+2] = gpu31::mont(gpu31::sub(u, v, p.p), tw, p);
                }
#pragma unroll
                for (unsigned i = 0; i < 8; i += 2) {
                    const u32 u = av[i], v = av[i+1];
                    av[i] = gpu31::add(u, v, p.p);
                    av[i+1] = gpu31::mont(gpu31::sub(u, v, p.p), q1, p);
                }
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) sa[ix[i]] = av[i];
            } else
#endif
            {
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) { sa[ix[i]] = av[i]; sb[ix[i]] = bv[i]; }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned k = 0; k < 3; ++k) {
        if (k >= logH - s8) break;
        const unsigned s = s8 + k, H = 1u << s;
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = gpu31::mont(waf[s], tfB[(H - 1) + within], p);
            gpu31::butterfly(sa[first * S + lane], sa[second * S + lane], w, p);
            gpu31::butterfly(sb[first * S + lane], sb[second * S + lane], w, p);
        }
        __syncthreads();
    }
#elif OC8_R8
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
                gpu31::butterfly(av[i], av[i + 1], t1, p);
                gpu31::butterfly(bv[i], bv[i + 1], t1, p);
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned i = (j < 2) ? j : j + 2;   // 0,1,4,5
                const u32 tw = (i & 1) ? t2b : t2a;
                gpu31::butterfly(av[i], av[i + 2], tw, p);
                gpu31::butterfly(bv[i], bv[i + 2], tw, p);
            }
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                gpu31::butterfly(av[i], av[i + 4], t4[i], p);
                gpu31::butterfly(bv[i], bv[i + 4], t4[i], p);
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
            gpu31::butterfly(sa[first * S + lane], sa[second * S + lane], w, p);
            gpu31::butterfly(sb[first * S + lane], sb[second * S + lane], w, p);
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
            gpu31::butterfly(a0, a1, t1, p); gpu31::butterfly(a2, a3, t1, p);
            gpu31::butterfly(b0, b1, t1, p); gpu31::butterfly(b2, b3, t1, p);
            gpu31::butterfly(a0, a2, t2, p); gpu31::butterfly(a1, a3, t3, p);
            gpu31::butterfly(b0, b2, t2, p); gpu31::butterfly(b1, b3, t3, p);
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
            gpu31::butterfly(sa[first * S + lane], sa[second * S + lane], w, p);
            gpu31::butterfly(sb[first * S + lane], sb[second * S + lane], w, p);
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
            gpu31::butterfly(sa[first * S + lane], sa[second * S + lane], w, p);
            gpu31::butterfly(sb[first * S + lane], sb[second * S + lane], w, p);
        }
        __syncthreads();
    }
#endif
    TCP_T(f2);
#if OC10_R16 && OC11_FUSE
    if (!oc11_fuse)
#endif
    {
        for (unsigned h = rw; h < HC; h += 8)
            sa[h * S + lane] = gpu31::mont(sa[h * S + lane], sb[h * S + lane], p);
        __syncthreads();
    }
    TCP_T(f3);
#if OC10_R16
    // OC10 inverse: radix-2 tail (highest stages) first, then radix-8 and
    // radix-16 groups in descending stage order.
#pragma unroll
    for (unsigned k = 0; k < 3; ++k) {
        if (k >= logH - s8) break;
        const unsigned s = logH - 1 - k, H = 1u << s;
        for (unsigned h = rw; h < HC / 2; h += 8) {
            const unsigned within = h & (H - 1);
            const unsigned first = 2 * (h - within) + within, second = first + H;
            const u32 w = gpu31::mont(wai[s], tiB[(H - 1) + within], p);
            const u32 u = sa[first * S + lane], v = sa[second * S + lane];
            sa[first * S + lane] = gpu31::add(u, v, p.p);
            sa[second * S + lane] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned q = 0; q < 2; ++q) {
#if OC10_R16 && OC11_FUSE
        if (oc11_fuse && q == 0) continue;   // already done in the fused DIT pass
#endif
        if (q >= n8) break;
        const unsigned s = s8 - 3 * (q + 1), H = 1u << s;
        for (unsigned g = rw; g < HC / 8; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 8 + w;
            unsigned ix[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) ix[i] = (r0 + i * H) * S + lane;
            u32 t4[4];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i)
                t4[i] = gpu31::mont(wai[s+2], tiB[(4 * H - 1) + w + i * H], p);
            const u32 t2a = gpu31::mont(wai[s+1], tiB[(2 * H - 1) + w], p);
            const u32 t2b = gpu31::mont(wai[s+1], tiB[(2 * H - 1) + w + H], p);
            const u32 t1 = gpu31::mont(wai[s], tiB[(H - 1) + w], p);
            u32 av[8];
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) av[i] = sa[ix[i]];
#pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                const u32 u = av[i], v = av[i+4];
                av[i] = gpu31::add(u, v, p.p);
                av[i+4] = gpu31::mont(gpu31::sub(u, v, p.p), t4[i], p);
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned i = (j < 2) ? j : j + 2;   // 0,1,4,5
                const u32 tw = (i & 1) ? t2b : t2a;
                const u32 u = av[i], v = av[i+2];
                av[i] = gpu31::add(u, v, p.p);
                av[i+2] = gpu31::mont(gpu31::sub(u, v, p.p), tw, p);
            }
#pragma unroll
            for (unsigned i = 0; i < 8; i += 2) {
                const u32 u = av[i], v = av[i+1];
                av[i] = gpu31::add(u, v, p.p);
                av[i+1] = gpu31::mont(gpu31::sub(u, v, p.p), t1, p);
            }
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) sa[ix[i]] = av[i];
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned q = 0; q < 4; ++q) {
        if (q >= n16) break;
        const unsigned s = (n16 - 1 - q) << 2, H = 1u << s;
        for (unsigned g = rw; g < HC / 16; g += 8) {
            const unsigned w = g & (H - 1);
            const unsigned r0 = (g - w) * 16 + w;
            unsigned ix[16];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) ix[i] = (r0 + i * H) * S + lane;
            u32 t2[2], t4[4], t8[8];
#pragma unroll
            for (unsigned k = 0; k < 2; ++k) t2[k] = gpu31::mont(wai[s+1], tiB[(2 * H - 1) + w + k * H], p);
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) t4[k] = gpu31::mont(wai[s+2], tiB[(4 * H - 1) + w + k * H], p);
#pragma unroll
            for (unsigned k = 0; k < 8; ++k) t8[k] = gpu31::mont(wai[s+3], tiB[(8 * H - 1) + w + k * H], p);
            const u32 t1 = gpu31::mont(wai[s], tiB[(H - 1) + w], p);
            u32 av[16];
#pragma unroll
            for (unsigned i = 0; i < 16; ++i) av[i] = sa[ix[i]];
            // stage s+3: pairs (i,i+8)
#pragma unroll
            for (unsigned i = 0; i < 8; ++i) {
                const u32 u = av[i], v = av[i+8];
                av[i] = gpu31::add(u, v, p.p);
                av[i+8] = gpu31::mont(gpu31::sub(u, v, p.p), t8[i], p);
            }
            // stage s+2: pairs (i,i+4), i mod 4 = k
#pragma unroll
            for (unsigned k = 0; k < 4; ++k) {
#pragma unroll
                for (unsigned i = k; i < 16; i += 8) {
                    const u32 u = av[i], v = av[i+4];
                    av[i] = gpu31::add(u, v, p.p);
                    av[i+4] = gpu31::mont(gpu31::sub(u, v, p.p), t4[k], p);
                }
            }
            // stage s+1: pairs (i,i+2), i mod 2 = k
#pragma unroll
            for (unsigned k = 0; k < 2; ++k) {
#pragma unroll
                for (unsigned i = k; i < 16; i += 4) {
                    const u32 u = av[i], v = av[i+2];
                    av[i] = gpu31::add(u, v, p.p);
                    av[i+2] = gpu31::mont(gpu31::sub(u, v, p.p), t2[k], p);
                }
            }
            // stage s: pairs (i,i+1)
#pragma unroll
            for (unsigned i = 0; i < 16; i += 2) {
                const u32 u = av[i], v = av[i+1];
                av[i] = gpu31::add(u, v, p.p);
                av[i+1] = gpu31::mont(gpu31::sub(u, v, p.p), t1, p);
            }
#if OC10_R16 && OC12_IO
            if (q + 1 == n16) {
                // OC12: last inverse group writes the finished coefficients
                // straight to global, skipping the shared write + store read.
#pragma unroll
                for (unsigned i = 0; i < 16; ++i)
                    rc[plane + u64(r0 + i * H) * NT_TILE + low] = av[i];
            } else
#endif
            {
#pragma unroll
                for (unsigned i = 0; i < 16; ++i) sa[ix[i]] = av[i];
            }
        }
        __syncthreads();
    }
#elif OC8_R8
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
            sa[second * S + lane] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
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
                av[i + 4] = gpu31::mont(gpu31::sub(u, v, p.p), t4[i], p);
            }
#pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned i = (j < 2) ? j : j + 2;   // 0,1,4,5
                const u32 tw = (i & 1) ? t2b : t2a;
                const u32 u = av[i], v = av[i + 2];
                av[i] = gpu31::add(u, v, p.p);
                av[i + 2] = gpu31::mont(gpu31::sub(u, v, p.p), tw, p);
            }
#pragma unroll
            for (unsigned i = 0; i < 8; i += 2) {
                const u32 u = av[i], v = av[i + 1];
                av[i] = gpu31::add(u, v, p.p);
                av[i + 1] = gpu31::mont(gpu31::sub(u, v, p.p), t1, p);
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
            sa[second * S + lane] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
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
            x0 = gpu31::add(u, v, p.p); x2 = gpu31::mont(gpu31::sub(u, v, p.p), t2, p);
            u = x1; v = x3;
            x1 = gpu31::add(u, v, p.p); x3 = gpu31::mont(gpu31::sub(u, v, p.p), t3, p);
            u = x0; v = x1;
            x0 = gpu31::add(u, v, p.p); x1 = gpu31::mont(gpu31::sub(u, v, p.p), t1, p);
            u = x2; v = x3;
            x2 = gpu31::add(u, v, p.p); x3 = gpu31::mont(gpu31::sub(u, v, p.p), t1, p);
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
            sa[second * S + lane] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
        }
        __syncthreads();
    }
#endif
    TCP_T(f4);
#if OC10_R16 && OC12_IO
    if (n16 == 0)
#endif
    {
        for (unsigned h = rw; h < HC; h += 8)
            rc[plane + u64(h) * NT_TILE + low] = sa[h * S + lane];
    }
    TCP_T(f5);
    TCP_ADD(5, f0, f1); TCP_ADD(6, f1, f2); TCP_ADD(7, f2, f3);
    TCP_ADD(8, f3, f4); TCP_ADD(9, f4, f5);
}

// inverse DIF low stages in 512-tiles (output: bit-reversed slot order).
__global__ void __launch_bounds__(256)
inv_low(u32* __restrict__ c, u32* __restrict__ dst2, const gpu31::Prime* __restrict__ primes,
        const u32* __restrict__ packed, unsigned n, unsigned K) {
    __shared__ u32 sa[NT_TILE + NT_TILE / 32], sr[TC_LT];
    const unsigned t = threadIdx.x;
    const u64 base = (u64(blockIdx.z) * K + blockIdx.y) * n + u64(blockIdx.x) * NT_TILE;
    const gpu31::Prime p = primes[blockIdx.y];
    for (unsigned x = t; x < TC_LT; x += 256) sr[x] = packed[u64(blockIdx.y) * TC_LT + x];
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
        const u32 w = sr[TWI(m, j)];
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
        const u32 w = sr[TWI(m, j)];
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
#if OC32_PREWEIGHT
    if (TC22_MERGED) {
        const u32 cw = c_crt_slotc[blockIdx.y];
        a0 = gpu31::mont(a0, cw, p);
        a1 = gpu31::mont(a1, cw, p);
    }
#endif
    if (OC31_TILE_RC && dst2) {
        // tile-major: [pair][tile][prime][512].  One CTA writes two
        // contiguous 1 KB blocks (lanes 0..255 and 256..511).
        const u64 tile = u64(blockIdx.z) * (n / NT_TILE) + blockIdx.x;
        const u64 ob = (tile * K + blockIdx.y) * NT_TILE;
        dst2[ob + t] = a0;
        dst2[ob + t + 256] = a1;
    } else {
        c[base + t] = a0;
        c[base + t + 256] = a1;
    }
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
    __shared__ u32 sR[EF_PR][TC_LT];
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
#if TC28_PFA
    if (t == 0) {
        unsigned nsm;
        asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsm));
        const unsigned lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        const unsigned nxt = lin + nsm;
        if (nxt < gridDim.x * gridDim.y * gridDim.z) {
            const unsigned bx = nxt % gridDim.x, rest = nxt / gridDim.x;
            const unsigned by = rest % gridDim.y, bz = rest / gridDim.y;
            const u32* psrc = (by ? src_b : src_a) + u64(bz) * n * l;
#pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                const unsigned i = bx * EF_ROWS + rowbase + (r >> 1) * 16 + g + (r & 1) * 8;
                const char* rp = reinterpret_cast<const char*>(psrc + u64(__brev(i) >> (32 - logN)) * l);
                for (unsigned off = 0; off < l * 4u; off += 128u)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(rp + off));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(rp + l * 4u - 1u));
            }
        }
    }
#endif
#if TC19_PF
    // TC19: the next group's twiddles and B fragments are fetched into
    // registers while the current group runs its DIT; setup is then only
    // shared stores.  sR: 2048 words = 4 per thread; sB: <= 1024 uint2 = 2.
    constexpr unsigned PF_R = EF_PR * TC_LT / 512, PF_B = 2;
    u32 pfR[PF_R];
    uint2 pfB[PF_B];
    auto prefetch = [&](unsigned g0n) {
#pragma unroll
        for (unsigned u = 0; u < PF_R; ++u) {
            const unsigned idx = tid + u * 512;
            const unsigned q = idx / TC_LT, r = idx % TC_LT;
            const unsigned kk = g0n * 2 + q;
            pfR[u] = kk < K ? packed[u64(kk) * TC_LT + r] : 0u;
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
            sR[idx / TC_LT][idx % TC_LT] = pfR[u];
        }
#pragma unroll
        for (unsigned u = 0; u < PF_B; ++u) {
            const unsigned idx = tid + u * 512;
            if (idx < EF_NTB * KS * 32) sB[idx] = pfB[u];
        }
#else
        // twiddles of this group's primes
        for (unsigned idx = tid; idx < EF_PR * TC_LT; idx += blockDim.x) {
            const unsigned q = idx / TC_LT, r = idx % TC_LT;
            const unsigned kk = nt0 * 2 + q;
            sR[q][r] = kk < K ? packed[u64(kk) * TC_LT + r] : 0u;
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
                tw[mt][j] = (rtwist && k < K) ? rtwist[u64(k) * n + i] : 0u;
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
                    val = rtwist ? gpu31::mont(gpu31::add(lo, hi, p.p), tw[mt][j], p)
                                 : gpu31::add(lo, hi, p.p);
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
                const u32 w = sR[q][TWI(m, j)];
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
                gpu31::butterfly(sT[q][padw(first)], sT[q][padw(second)], sR[q][TWI(m, j)], pq[rep]);
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
                        const u32 tw = tw8[TWI(m + st, j)];
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
                gpu31::butterfly(sT[q][padw(first)], sT[q][padw(second)], sR[q][TWI(m, j)], pq[rep]);
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
                gpu31::butterfly(sT[q][padw(first)], sT[q][padw(second)], sR[q][TWI(m, j)], pq[rep]);
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

// ---------------------------------------------------------------------------
// TC29: encode + forward DIT low stages 0..7 on 256-row tiles, 256 threads,
// 2 CTAs/SM.  Same per-warp layout as encode_fwdlow (32 rows, 2 m-tiles,
// A fragments in registers); stage 8 is done by fused_high (ef256).
// grid (n/256, 2, B), block 256, static smem ~23.8 KB.
// ---------------------------------------------------------------------------
constexpr unsigned E2_ROWS = 256;
// OC28_EPICONF: 272 (== 16 mod 32) removes the epilogue's 2-way bank conflict
// on the q=0/q=1 sT rows; 0 restores the original 264 (== 8 mod 32).
constexpr unsigned E2_PT = OC28_EPICONF ? (E2_ROWS + E2_ROWS / 16) : (E2_ROWS + E2_ROWS / 32);
__global__ void __launch_bounds__(256, 2)
encode_fwdlow256(const u32* __restrict__ src_a, const u32* __restrict__ src_b,
                 u32* __restrict__ ra, u32* __restrict__ rb,
                 const uint2* __restrict__ frag, const gpu31::Prime* __restrict__ primes,
                 const gpu31::Weight* __restrict__ weights,
                 const u32* __restrict__ packed, unsigned n, unsigned l, unsigned K,
                 unsigned logN, unsigned KS, unsigned NT) {
    __shared__ u32 sT[EF_PR][E2_PT];
#if TC30_CPA
    __shared__ __align__(16) u32 sRb[2][EF_PR][E2_ROWS];
    __shared__ __align__(16) uint2 sBb[2][EF_NTB * ENC_KS_MAX * 32];
    __shared__ __align__(16) gpu31::Prime sPb[2][EF_PR];    // TC32
    __shared__ __align__(16) gpu31::Weight sWb[2][EF_PR];
#else
    __shared__ u32 sR[EF_PR][E2_ROWS];
    __shared__ uint2 sB[EF_NTB * ENC_KS_MAX * 32];
#endif
    const u32* src = blockIdx.y ? src_b : src_a;
    u32* dst = blockIdx.y ? rb : ra;
    const unsigned pair = blockIdx.z;
    const unsigned start = blockIdx.x * E2_ROWS;
    const u64 base = u64(pair) * n * l;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid >> 5, lane = tid & 31;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned rowbase = warp * 32;
    TCP_T(e0);
    long long pe[5] = {0, 0, 0, 0, 0};
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
#if TC28_PFA
    if (t == 0) {
        unsigned nsm;
        asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsm));
        const unsigned lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        const unsigned nxt = lin + 2 * nsm;
        if (nxt < gridDim.x * gridDim.y * gridDim.z) {
            const unsigned bx = nxt % gridDim.x, rest = nxt / gridDim.x;
            const unsigned by = rest % gridDim.y, bz = rest / gridDim.y;
            const u32* psrc = (by ? src_b : src_a) + u64(bz) * n * l;
#pragma unroll
            for (unsigned r = 0; r < 4; ++r) {
                const unsigned i = bx * E2_ROWS + rowbase + (r >> 1) * 16 + g + (r & 1) * 8;
                const char* rp = reinterpret_cast<const char*>(psrc + u64(__brev(i) >> (32 - logN)) * l);
                for (unsigned off = 0; off < l * 4u; off += 128u)
                    asm volatile("prefetch.global.L2 [%0];" :: "l"(rp + off));
                asm volatile("prefetch.global.L2 [%0];" :: "l"(rp + l * 4u - 1u));
            }
        }
    }
#endif
#if TC30_CPA
    // TC30: group twiddles/B fragments arrive by cp.async into a double
    // buffer one group ahead (no registers, overlaps the previous group).
    auto issue = [&](unsigned nb, unsigned buf) {
        for (unsigned idx = tid; idx < EF_PR * E2_ROWS; idx += 256) {
            const unsigned q = idx / E2_ROWS, r = idx % E2_ROWS;
            const unsigned kk = nb * 2 + q;
            const bool ok = kk < K;
            const u32* gp = packed + (ok ? u64(kk) * TC_LT + r : 0);
            const unsigned sa = unsigned(__cvta_generic_to_shared(&sRb[buf][q][r]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 4 : 0) : "memory");
        }
        for (unsigned idx = tid; idx < EF_NTB * KS * 32; idx += 256) {
            const unsigned j = idx / (KS * 32), rem = idx - j * (KS * 32);
            const unsigned nt = nb + j;
            const bool ok = nt < NT;
            const uint2* gp = frag + (ok ? std::size_t(nt) * KS * 32 + rem : 0);
            const unsigned sa = unsigned(__cvta_generic_to_shared(&sBb[buf][idx]));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 8 : 0) : "memory");
        }
#if TC32_PW
        if (tid < 2 * EF_PR) {
            // TC32: this group's primes and first-limb weights ride along,
            // so the epilogue/DIT read them from shared, not global.
            const unsigned q = tid & (EF_PR - 1);
            const unsigned kk = nb * 2 + q;
            const bool ok = kk < K;
            if (tid < EF_PR) {
                const gpu31::Prime* gp = primes + (ok ? kk : 0);
                const unsigned sa = unsigned(__cvta_generic_to_shared(&sPb[buf][q]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 16 : 0) : "memory");
            } else {
                const gpu31::Weight* gp = weights + (ok ? std::size_t(kk) * l : 0);
                const unsigned sa = unsigned(__cvta_generic_to_shared(&sWb[buf][q]));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(sa), "l"(gp), "r"(ok ? 8 : 0) : "memory");
            }
        }
#endif
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    issue(0, 0);
#endif
    for (unsigned nt0 = 0; nt0 < NT; nt0 += EF_NTB) {
        TCP_T(g0);
#if TC30_CPA
        const unsigned cb = (nt0 / EF_NTB) & 1;
        u32 (*sR)[E2_ROWS] = sRb[cb];
        uint2* sB = sBb[cb];
#if TC32_PW
        const gpu31::Prime* sP = sPb[cb];
        const gpu31::Weight* sW = sWb[cb];
#endif
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (nt0 + EF_NTB < NT) issue(nt0 + EF_NTB, cb ^ 1);
#else
        for (unsigned idx = tid; idx < EF_PR * E2_ROWS; idx += 256) {
            const unsigned q = idx / E2_ROWS, r = idx % E2_ROWS;
            const unsigned kk = nt0 * 2 + q;
            sR[q][r] = kk < K ? packed[u64(kk) * TC_LT + r] : 0u;
        }
        for (unsigned idx = tid; idx < EF_NTB * KS * 32; idx += 256) {
            const unsigned j = idx / (KS * 32), rem = idx - j * (KS * 32);
            const unsigned nt = nt0 + j;
            sB[idx] = nt < NT ? frag[std::size_t(nt) * KS * 32 + rem] : make_uint2(0u, 0u);
        }
        __syncthreads();
#endif
        TCP_T(g1);
        int acc[2][EF_NTB][4];
#pragma unroll
        for (unsigned mt = 0; mt < 2; ++mt)
#pragma unroll
            for (unsigned j = 0; j < EF_NTB; ++j)
                acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0;
#pragma unroll
        for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
            if (ks < KS) {
                uint2 b[EF_NTB];
#pragma unroll
                for (unsigned j = 0; j < EF_NTB; ++j) b[j] = sB[(j * KS + ks) * 32 + lane];
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
#if TC30_CPA && TC32_PW
            if (k < K) { p = sP[j * 2 + (t >> 1)]; w0 = sW[j * 2 + (t >> 1)]; }
#else
            if (k < K) { p = primes[k]; w0 = weights[std::size_t(k) * l]; }
#endif
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
                    val = gpu31::add(lo, hi, p.p);
                }
                const unsigned rl = rowbase + mt * 16 + g + (odd ? 8 : 0);
                sT[j * 2 + (t >> 1)][padw(rl)] = val;
            }
        }
        __syncthreads();
        TCP_T(g3);
#if OC29_DITSHFL
        // OC29: one warp per prime; the 256-point row lives in 8 registers per
        // lane at position p = lane + 32*e (coalesced: bank = (lane+e)%32).
        // Stages 0..4 are lane-xor shfl; stages 5..7 are an in-thread radix-8
        // over e.  No shared round trips / barriers between DIT stages; the
        // epilogue->DIT handoff barrier above is the only one.  Store goes
        // straight from registers to global in coalesced 128B segments.
        {
            const unsigned q = warp;
            const unsigned kk = nt0 * 2 + q;
#if TC30_CPA && TC32_PW
            const gpu31::Prime pr = sP[q];
#else
            const gpu31::Prime pr = primes[kk < K ? kk : 0];
#endif
            const u32* tw = sR[q];
            u32 v[8];
#pragma unroll
            for (unsigned e = 0; e < 8; ++e) v[e] = sT[q][padw(lane + 32 * e)];

#pragma unroll
            for (unsigned s = 0; s < 5; ++s) {
                const unsigned Hb = 1u << s;
                const u32 wt = tw[TWI(s, lane & (Hb - 1))];
                const bool upper = (lane & Hb) != 0;
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    const u32 other = __shfl_xor_sync(0xffffffffu, v[e], Hb);
                    // bf(lower,upper): lower' = lower+mont(upper), upper'=lower-mont(upper)
                    if (!upper) {
                        const u32 u = v[e], t = gpu31::mont(other, wt, pr);
                        v[e] = gpu31::add(u, t, pr.p);
                    } else {
                        const u32 u = other, t = gpu31::mont(v[e], wt, pr);
                        v[e] = gpu31::sub(u, t, pr.p);
                    }
                }
            }

#pragma unroll
            for (unsigned st = 0; st < 3; ++st) {
                const unsigned span = 1u << st;
                const unsigned stage = 5 + st;
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) {
                    if (e & span) continue;
                    const unsigned jtw = lane + (e & (span - 1)) * 32;
                    gpu31::butterfly(v[e], v[e + span], tw[TWI(stage, jtw)], pr);
                }
            }

            if (kk < K) {
                const u64 base_out = rbase + u64(kk) * n + start;
#pragma unroll
                for (unsigned e = 0; e < 8; ++e)
                    dst[base_out + lane + 32 * e] = v[e];
            }
        }
#else
        // DIT stages 0..7: radix-8 (0-2), radix-8 (3-5): 8 primes x 32 groups
        // = 256 groups, one per thread; radix-4 (6-7): 8 x 64 groups, 2/thread.
        {
            const unsigned q = tid >> 5, gi = tid & 31;
            const unsigned kk = nt0 * 2 + q;
#if TC30_CPA && TC32_PW
            const gpu31::Prime pr8 = sP[q];
#else
            const gpu31::Prime pr8 = primes[kk < K ? kk : 0];
#endif
            u32* row = sT[q];
            const u32* tw8 = sR[q];
#pragma unroll
            for (unsigned m = 0; m < 6; m += 3) {
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
                        gpu31::butterfly(v[e], v[e + span], tw8[TWI(m + st, j)], pr8);
                    }
                }
#pragma unroll
                for (unsigned e = 0; e < 8; ++e) row[padw(x0 + e * H)] = v[e];
                __syncthreads();
            }
        }
#pragma unroll
        for (unsigned rep = 0; rep < 2; ++rep) {
            const unsigned q = (tid >> 6) + rep * 4, w = tid & 63;   // H = 64, x0 = w
            const unsigned kk = nt0 * 2 + q;
#if TC30_CPA && TC32_PW
            const gpu31::Prime pr4 = sP[q];
#else
            const gpu31::Prime pr4 = primes[kk < K ? kk : 0];
#endif
            u32* row = sT[q];
            const unsigned i0 = padw(w), i1 = padw(w + 64), i2 = padw(w + 128), i3 = padw(w + 192);
            u32 v0 = row[i0], v1 = row[i1], v2 = row[i2], v3 = row[i3];
            const u32 t1 = sR[q][TWI(6, w)];
            gpu31::butterfly(v0, v1, t1, pr4);
            gpu31::butterfly(v2, v3, t1, pr4);
            gpu31::butterfly(v0, v2, sR[q][TWI(7, w)], pr4);
            gpu31::butterfly(v1, v3, sR[q][TWI(7, w + 64)], pr4);
            row[i0] = v0; row[i1] = v1; row[i2] = v2; row[i3] = v3;
        }
        __syncthreads();
#endif
        TCP_T(g4);
#if !OC29_DITSHFL
        for (unsigned idx = tid; idx < EF_PR * E2_ROWS; idx += 256) {
            const unsigned q = idx / E2_ROWS, pos = idx % E2_ROWS;
            const unsigned kk = nt0 * 2 + q;
            if (kk < K) dst[rbase + u64(kk) * n + start + pos] = sT[q][padw(pos)];
        }
        __syncthreads();
#endif
        TCP_T(g5);
        pe[0] += g1 - g0; pe[1] += g2 - g1; pe[2] += g3 - g2; pe[3] += g4 - g3; pe[4] += g5 - g4;
    }
    TCP_ADD(10, 0, pe[0]); TCP_ADD(11, 0, pe[1]); TCP_ADD(12, 0, pe[2]);
    TCP_ADD(13, 0, pe[3]); TCP_ADD(14, 0, pe[4]);
}

// ---------------------------------------------------------------------------
// TC26/TC27: inv_low fused into CRT (merged-psi path only).  A cluster of
// CL = 512/ROWS CTAs owns one 512-slot tile of one pair; CTA `rank` owns slots
// rank*ROWS..+ROWS-1 and the contiguous prime block rank*P..rank*P+P-1.
// Phase A: inverse DIF low stages per prime (one prime per 256-thread half,
// the next prime's tile + twiddles prefetched into registers), terms of 4
// consecutive primes packed and sent to the owner's sA with one 16-byte
// st.shared::cluster.v4.  Phases B..E = crt_tc2 on ROWS rows.
// TC26 (ROWS 64, cluster 8, 1 CTA/SM): correct but 78 ms.  TC27: ROWS 32,
// cluster 16 (non-portable) -> ~31 KB smem -> 3 CTAs/SM; v4 DSMEM stores.
// grid (n/ROWS, 1, B), cluster (CL,1,1) set at launch, block ROWS*8.
// ---------------------------------------------------------------------------
inline std::size_t invcrt_smem(unsigned rows, unsigned SA, unsigned SX) {
    return ((std::size_t(rows) * SA * 4 + 15) & ~std::size_t(15)) + std::size_t(rows) * SX * 8;
}
inline unsigned invcrt_P(unsigned K, unsigned rows) {   // primes per rank
    const unsigned cl = NT_TILE / rows, halves = rows / 32;
    const unsigned q = 4 * halves;
    return ((K + cl - 1) / cl + q - 1) / q * q;
}
#ifdef __CUDACC__
__device__ __forceinline__ unsigned cl_rank() {
    unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
__device__ __forceinline__ void cl_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void st_cluster4(const u32* local, unsigned rank, u32 a, u32 b, u32 c, u32 d) {
    const u32 ad = u32(__cvta_generic_to_shared(local));
    u32 ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(ad), "r"(rank));
    asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(ra), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

template <unsigned ROWS>
__global__ void __launch_bounds__(ROWS * 8, ROWS == 32 ? 3 : 1)
inv_crt(gpu31::CrtArgs pl, const u32* __restrict__ rc, u32* __restrict__ out,
        const uint2* __restrict__ frag, CrtTc cfg,
        const u32* __restrict__ packed, unsigned logN, unsigned P) {
    constexpr unsigned NTH = ROWS * 8, WARPS = NTH / 32, MT = ROWS / 16, HALVES = ROWS / 32;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    u32* sA = reinterpret_cast<u32*>(smem_raw);
    unsigned char* r2 = smem_raw + ((std::size_t(ROWS) * cfg.SA * 4 + 15) & ~std::size_t(15));
    u64* sX = reinterpret_cast<u64*>(r2);
    const unsigned n = pl.n, l = pl.limbs, K = pl.count;
    const unsigned tid = threadIdx.x, pair = blockIdx.z;
    const unsigned rank = cl_rank();
    const unsigned tile = blockIdx.x / (NT_TILE / ROWS);
    const unsigned start = tile * NT_TILE + rank * ROWS;
    const unsigned W = cfg.KS * 8;

    if (W > K + 1)
    for (unsigned idx = tid; idx < ROWS * (W - K - 1); idx += NTH) {
        const unsigned r = idx / (W - K - 1), j = K + 1 + idx % (W - K - 1);
        sA[r * cfg.SA + j] = 0u;
    }
    cl_sync();   // all CTAs of the cluster running before DSMEM stores

    // ---- Phase A
    {
        const unsigned h = tid >> 8, t = tid & 255;
        u32* sa = reinterpret_cast<u32*>(r2) + h * (NT_TILE + NT_TILE / 32 + TC_LT);
        u32* sr = sa + NT_TILE + NT_TILE / 32;
        const unsigned PH = P / HALVES;                 // multiple of 4
        const unsigned jb = rank * P + h * PH;          // first prime of this half
        const unsigned o0 = t / ROWS, w0 = t % ROWS;
        const unsigned o1 = (t + 256) / ROWS, w1 = (t + 256) % ROWS;
        const u64 tb = u64(tile) * NT_TILE;
        u32 c0 = 0, c1 = 0, cw0 = 0, cw1 = 0;
        if (jb < K) {
            const u64 base = (u64(pair) * K + jb) * n + tb;
            c0 = rc[base + t]; c1 = rc[base + t + 256];
            cw0 = packed[u64(jb) * TC_LT + t]; cw1 = packed[u64(jb) * TC_LT + t + 256];
        }
        u32 k0[4], k1[4];
        for (unsigned jj = 0; jj < PH; ++jj) {
            const unsigned j = jb + jj;
            u32 n0 = 0, n1 = 0, nw0 = 0, nw1 = 0;
            if (jj + 1 < PH && j + 1 < K) {
                const u64 base = (u64(pair) * K + j + 1) * n + tb;
                n0 = rc[base + t]; n1 = rc[base + t + 256];
                nw0 = packed[u64(j + 1) * TC_LT + t]; nw1 = packed[u64(j + 1) * TC_LT + t + 256];
            }
            sa[padw(t)] = c0; sa[padw(t + 256)] = c1;
            sr[t] = cw0; sr[t + 256] = cw1;
            __syncthreads();
            const gpu31::Prime p = c_crt_prime[j < K ? j : 0];
#pragma unroll
            for (int m = 8; m >= 5; --m) {
                const unsigned half = 1u << m, jx = t & (half - 1);
                const unsigned first = 2 * (t - jx) + jx, second = first + half;
                const u32 w = sr[TWI(m, jx)];
                const u32 u = sa[padw(first)], v = sa[padw(second)];
                sa[padw(first)] = gpu31::add(u, v, p.p);
                sa[padw(second)] = gpu31::mont(gpu31::sub(u, v, p.p), w, p);
                __syncthreads();
            }
            u32 a0 = sa[padw(t)], a1 = sa[padw(t + 256)];
#pragma unroll
            for (int m = 4; m >= 0; --m) {
                const unsigned half = 1u << m;
                const unsigned jx = t & (half - 1);
                const u32 w = sr[TWI(m, jx)];
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
            const unsigned q = jj & 3;
            u32 e0 = 0u, e1 = 0u;
            if (j < K) {
                const u32 cc = c_crt_slotc[j];
                e0 = gpu31::mont(a0, cc, p);
                e1 = gpu31::mont(a1, cc, p);
            }
#pragma unroll
            for (unsigned z = 0; z < 4; ++z) if (z == q) { k0[z] = e0; k1[z] = e1; }
            if (q == 3) {
                const unsigned j0 = j - 3;
                if (j0 + 4 <= W) {
                    st_cluster4(sA + w0 * cfg.SA + j0, o0, k0[0], k0[1], k0[2], k0[3]);
                    st_cluster4(sA + w1 * cfg.SA + j0, o1, k1[0], k1[1], k1[2], k1[3]);
                }
            }
            c0 = n0; c1 = n1; cw0 = nw0; cw1 = nw1;
            __syncthreads();   // sa/sr reused next iteration
        }
    }
    cl_sync();

    // ---- Phase B: v = round(sum term_j/p_j), 8 threads/row
    {
        const unsigned row = tid >> 3, q = tid & 7;
        u64 fsum = 0;
        for (unsigned j = q; j < K; j += 8)
            fsum += (u64(sA[row * cfg.SA + j]) * c_crt_corr[j]) >> 30;
        fsum += __shfl_xor_sync(0xffffffffu, fsum, 1);
        fsum += __shfl_xor_sync(0xffffffffu, fsum, 2);
        fsum += __shfl_xor_sync(0xffffffffu, fsum, 4);
        if (q == 0) sA[row * cfg.SA + K] = u32((fsum + (u64(1) << 31)) >> 32);
    }
    __syncthreads();

    // ---- Phase C: GEMM, warps split n-tiles, MT m-tiles per B fragment
    {
        const unsigned warp = tid >> 5, lane = tid & 31;
        const unsigned g = lane >> 2, t = lane & 3;
        const bool odd = t & 1;
        const unsigned sh = (t & 1) * 16;
        for (unsigned nt0 = warp; nt0 < cfg.NT; nt0 += 2 * WARPS) {
            const unsigned ntA = nt0, ntB = nt0 + WARPS;
            const bool hasB = ntB < cfg.NT;
            int acc[MT][2][4];
#pragma unroll
            for (unsigned mt = 0; mt < MT; ++mt)
#pragma unroll
                for (unsigned q = 0; q < 2; ++q) acc[mt][q][0] = acc[mt][q][1] = acc[mt][q][2] = acc[mt][q][3] = 0;
#pragma unroll 3
            for (unsigned ks = 0; ks < cfg.KS; ++ks) {
                const uint2 b0 = frag[(std::size_t(ntA) * cfg.KS + ks) * 32 + lane];
                const uint2 b1 = hasB ? frag[(std::size_t(ntB) * cfg.KS + ks) * 32 + lane] : make_uint2(0u, 0u);
#pragma unroll
                for (unsigned mt = 0; mt < MT; ++mt) {
                    const u32* q0 = sA + (mt * 16 + g) * cfg.SA + ks * 8 + t;
                    const u32* q1 = q0 + 8 * cfg.SA;
                    const u32 x0 = q0[0], x1 = q1[0], x2 = q0[4], x3 = q1[4];
                    mma_u8(acc[mt][0], x0, x1, x2, x3, b0.x, b0.y);
                    if (hasB) mma_u8(acc[mt][1], x0, x1, x2, x3, b1.x, b1.y);
                }
            }
#pragma unroll
            for (unsigned mt = 0; mt < MT; ++mt)
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
    }
    __syncthreads();

    // ---- Phase D: carry + reduce
    const unsigned DS = 2 * cfg.SX;
    u32* digits = reinterpret_cast<u32*>(sX);
    if (tid < ROWS) {
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

    // ---- Phase E: store
    const u64 obase = u64(pair) * n * l;
    for (unsigned idx = tid; idx < ROWS * l; idx += NTH) {
        const unsigned r = idx / l, cc = idx - r * l;
        const unsigned pos = __brev(start + r) >> (32 - logN);
        out[obase + u64(pos) * l + cc] = digits[r * DS + cc];
    }
}
#endif

#endif  // __CUDACC__

}  // namespace tc
