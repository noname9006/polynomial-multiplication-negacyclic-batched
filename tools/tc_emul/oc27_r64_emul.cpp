// oc27_r64_emul.cpp — CPU validation of OC27 (crt_tc2_r64, 64 rows/CTA).
//
// Three paths compared bit-for-bit on identical inputs:
//   REF : naive big-integer sum_j term_j * W_j (little-endian u32 limbs),
//         then gpu31::reduce_sum<1> (the REAL host-compiled device helper,
//         included from ../../gpu31.cuh without __CUDACC__).
//   P32 : the current crt_tc2 path — mma emulation (gemm_emul layout),
//         32 rows/CTA, single 56-column carry chain (OC26 4-thread logic
//         emulated sequentially: segment scans + owner propagation),
//         reduce_sum<1>, store.
//   P64 : the OC27 path — same mma emulation but 64 rows/CTA, 4 m-tiles per
//         warp, n-tiles split into two halves of NTH=(NT+1)/2; per half the
//         oc27_carry_half logic (two 32-row waves, half-0 low-digit store,
//         half-1 cin0 fold + shift + add + reduce) mirrored step by step.
//
// Coverage: combat shape K=114,L=55 (KS=15,NT=28,SA=132,SXH=29) plus shape
// fuzz (K=8/31/63/114/127, L=8/28/55) and edge data: zeros, all-0xffffffff
// terms, all-(p-1) terms, max W bytes, seam-stress (huge half-0 sum forcing
// multi-word cin0), full-u64 addend fuzz at the carry level, random fuzz.
// Also checks the host guards crt2_r64_ok / crt_tc2_r64_smem byte counts.
//
// Build (from repo root):
//   g++ -std=c++17 -O2 -I. tools/tc_emul/oc27_r64_emul.cpp -o oc27_emul
// (gpu31.cuh compiles host-only without __CUDACC__; tc.cuh's device section
// is guarded, only bake_crt / crt_sa_stride / crt2_* helpers are used.)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <string>

#include "gpu31.cuh"   // host path: real reduce_sum<1>, subtract_if_ge, etc.
#include "tc.cuh"      // host helpers: bake_crt, crt_sa_stride, crt2_sxh, ...

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using namespace tc;

// definitions for the (never-called) device kernels' extern shared symbols
namespace gpu31 { u64 shared[4]; u32 stage[4]; u32 data[4]; }
namespace tc { unsigned char smem_raw[16]; u32 sm[4]; }

// ---------------------------------------------------------------------------
// mma emulation (identical layout to gemm_emul.cpp): C[16][8] += A*B
static void mma_emul(const u32 A[32][4], const u32 Bf[32][2], long long C[16][8]) {
    unsigned char a[16][32] = {}, b[32][8] = {};
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; ++i) {
            a[g][t * 4 + i] = A[lane][0] >> (8 * i);
            a[g + 8][t * 4 + i] = A[lane][1] >> (8 * i);
            a[g][16 + t * 4 + i] = A[lane][2] >> (8 * i);
            a[g + 8][16 + t * 4 + i] = A[lane][3] >> (8 * i);
            b[t * 4 + i][g] = Bf[lane][0] >> (8 * i);
            b[16 + t * 4 + i][g] = Bf[lane][1] >> (8 * i);
        }
    }
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 8; ++c)
            for (int k = 0; k < 32; ++k) C[r][c] += a[r][k] * b[k][c];
}

// epilogue of one (mt,nt) tile: sX[row][nt*2+(t>>1)] for all 32 lanes
static void epi_emul(const long long C[16][8], unsigned nt, unsigned SX,
                     unsigned mt, u64* sXrow /* row base, stride SX */) {
    for (int lane = 0; lane < 32; ++lane) {
        int t = lane & 3, g = lane >> 2;
        unsigned sh = (t & 1) * 16;
        bool odd = t & 1;
        int tp = (lane ^ 1) & 3;
        unsigned shp = (tp & 1) * 16;
        u64 pg = (u64(u32(C[g][2 * t])) << sh) + (u64(u32(C[g][2 * t + 1])) << (sh + 8));
        u64 pg8 = (u64(u32(C[g + 8][2 * t])) << sh) + (u64(u32(C[g + 8][2 * t + 1])) << (sh + 8));
        u64 ppg = (u64(u32(C[g][2 * tp])) << shp) + (u64(u32(C[g][2 * tp + 1])) << (shp + 8));
        u64 ppg8 = (u64(u32(C[g + 8][2 * tp])) << shp) + (u64(u32(C[g + 8][2 * tp + 1])) << (shp + 8));
        u64 tot = (odd ? pg8 : pg) + (odd ? ppg8 : ppg);
        unsigned row = mt * 16 + g + (odd ? 8 : 0);
        sXrow[row * SX + nt * 2 + (t >> 1)] = tot;
    }
}

// ---------------------------------------------------------------------------
// OC26 carry (sequential mirror of oc26_carry_row): xs[words] -> dg[l2]
static void carry26(const u64* xs, u32* dg, unsigned words, unsigned l2) {
    const unsigned base = words >> 2, rem = words & 3;
    u64 cout[4];
    bool big[4];
    for (unsigned s = 0; s < 4; ++s) {
        const unsigned a = s * base + (s < rem ? s : rem);
        const unsigned len = base + (s < rem ? 1u : 0u);
        u64 run = 0, bg = 0;
        for (unsigned j = 0; j < len; ++j) {
            const u64 x = xs[a + j];
            const u64 t = run + x;
            dg[a + j] = u32(t);
            run = t >> 32;
            bg |= x >> 48;
        }
        cout[s] = run;
        big[s] = bg != 0;
    }
    bool replay = big[0] || big[1] || big[2] || big[3];
    u64 cin = 0;
    if (replay) {
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
            cin = cout[s] + delta;
        }
    }
    for (unsigned w = words; w < l2; ++w) {
        dg[w] = u32(cin);
        cin >>= 32;
    }
}

// ---------------------------------------------------------------------------
// OC27 carry-half mirror (oc27_carry_half, one row; waves are independent).
// xs: this half's u64 columns (stride SXH); cinout: handover slot;
// out: the row's l global words (half 0 writes, half 1 read-modify-writes).
static void carry27_half(const u64* xs, unsigned words, unsigned l,
                         unsigned DSH, unsigned half, unsigned w0,
                         u64* cinout, u32* out,
                         unsigned bits, u32 gap, const u32* q) {
    const unsigned l2 = l + 2;
    std::vector<u32> dg(DSH, 0);
    // parallel segment scans (sequential mirror) + owner propagation
    const unsigned base = words >> 2, rem = words & 3;
    u64 cout[4];
    bool big[4];
    for (unsigned s = 0; s < 4; ++s) {
        const unsigned a = s * base + (s < rem ? s : rem);
        const unsigned len = base + (s < rem ? 1u : 0u);
        u64 run = 0, bg = 0;
        for (unsigned j = 0; j < len; ++j) {
            const u64 v = xs[a + j];
            const u64 t = run + v;
            dg[a + j] = u32(t);
            run = t >> 32;
            bg |= v >> 48;
        }
        cout[s] = run;
        big[s] = bg != 0;
    }
    u64 cin = 0;
    if (big[0] || big[1] || big[2] || big[3]) {
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
            cin = cout[s] + delta;
        }
    }
    if (half == 0) {
        *cinout = cin;
        for (unsigned w = words; w < l2; ++w) dg[w] = 0;   // LOW digits only
        gpu31::reduce_sum<1>(dg.data(), l, bits, gap, q);
        for (unsigned cc = 0; cc < l; ++cc) out[cc] = dg[cc];
    } else {
        u64 tail = cin;
        for (unsigned w = words; w < l2; ++w) {
            dg[w] = u32(tail);
            tail >>= 32;
        }
        u64 add = *cinout;
        for (unsigned w = 0; w < l2 && add; ++w) {
            const u64 sum = u64(dg[w]) + u32(add);
            dg[w] = u32(sum);
            add = (add >> 32) + (sum >> 32);
        }
        for (unsigned j = DSH; j-- > w0;)
            dg[j] = (j - w0 < l2) ? dg[j - w0] : 0u;
        for (unsigned j = 0; j < w0; ++j) dg[j] = 0;
        u64 carry = 0;
        for (unsigned cc = 0; cc < l; ++cc) {
            const u64 v = u64(dg[cc]) + u64(out[cc]) + carry;
            dg[cc] = u32(v);
            carry = v >> 32;
        }
        for (unsigned w = l; w < DSH && carry; ++w) {
            const u64 v = u64(dg[w]) + carry;
            dg[w] = u32(v);
            carry = v >> 32;
        }
        gpu31::reduce_sum<1>(dg.data(), l, bits, gap, q);
        for (unsigned cc = 0; cc < l; ++cc) out[cc] = dg[cc];
    }
}

// ---------------------------------------------------------------------------
struct Shape {
    unsigned K, L, KS, NT, SA, SX, SXH, NTH;
};
static Shape make_shape(unsigned K, unsigned L) {
    Shape s{};
    s.K = K; s.L = L;
    unsigned ks, nt;
    std::vector<u32> dummyW(std::size_t(L) * (K + 1), 0);
    bake_crt(dummyW, L, K + 1, ks, nt);
    s.KS = ks; s.NT = nt;
    unsigned sa = ks * 8 + 8;
    while (sa % 32 != 4 && sa % 32 != 28) ++sa;
    s.SA = sa;
    s.SX = 2 * nt + 1;
    s.SXH = crt2_sxh(nt);
    s.NTH = (s.SXH - 1) / 2;
    return s;
}

// naive reference: out = sum_j term_j * W_j mod q (l+2-word chain + reduce)
static void ref_row(const u32* terms, const u32* W, unsigned pitch,
                    unsigned L, unsigned bits, u32 gap, const u32* q,
                    u32* out) {
    const unsigned l2 = L + 2;
    std::vector<u32> acc(l2, 0);
    for (unsigned j = 0; j < pitch; ++j) {
        u64 cc = 0;
        for (unsigned i = 0; i < l2; ++i) {
            const u64 v = cc + acc[i] + (i < L ? u64(terms[j]) * W[std::size_t(i) * pitch + j] : 0);
            acc[i] = u32(v);
            cc = v >> 32;
        }
    }
    gpu31::reduce_sum<1>(acc.data(), L, bits, gap, q);
    for (unsigned i = 0; i < L; ++i) out[i] = acc[i];
}

// ---------------------------------------------------------------------------
// OC34 staging mirror: cp.async double-buffered residue loads, batched by
// OC34_CPB primes, buffers aliased on the dead sX region.  Emulates the
// issue/wait/mount order per batch (the copies themselves are synchronous
// here — what is validated is the indexing, the zero-fill predicates, the
// j%NG==jg mount split, and that every (row, prime) term lands in sA exactly
// as the synchronous path produces it).  Returns the staged sA terms.
static void stage34(const u32* rc, u64 rbase2, unsigned start, unsigned n,
                    unsigned K, unsigned SA, unsigned SXH,
                    std::vector<u32>& sA /* rows*SA */) {
    const unsigned CPB = OC34_CPB;
    const unsigned NB = (K + CPB - 1) / CPB;
    // buffer fit check identical to oc34_ok's shared-size clause
    if (std::size_t(2) * CPB * 64 * 4 > std::size_t(64) * SXH * 8) {
        std::printf("OC34 BUFFER FIT FAIL\n");
        std::exit(2);
    }
    std::vector<u32> rbuf(std::size_t(2) * CPB * 64, 0xdeadbeefu);
    auto issue = [&](unsigned b, unsigned buf) {
        for (unsigned tid = 0; tid < 128; ++tid) {
            const unsigned cq = tid >> 6, cr = tid & 63;
            const bool ract = (start + cr) < n;
            for (unsigned p = 0; p < CPB / 2; ++p) {
                const unsigned j = b * CPB + p * 2 + cq;
                const bool ok = j < K && ract;
                const u32 v = ok ? rc[(rbase2 + j) * 512u + ((start + cr) & 511u)] : 0u;
                rbuf[std::size_t(buf) * CPB * 64 + (p * 2 + cq) * 64 + cr] = v;
            }
        }
    };
    issue(0, 0);
    for (unsigned b = 0; b < NB; ++b) {
        const unsigned buf = b & 1;
        if (b + 1 < NB) issue(b + 1, buf ^ 1);
        // mount: thread tid (c = tid&63, jg = tid>>6) takes primes j%2==jg
        for (unsigned tid = 0; tid < 128; ++tid) {
            const unsigned c = tid & 63, jg = tid >> 6;
            for (unsigned p = 0; p < CPB; ++p) {
                const unsigned j = b * CPB + p;
                if (j < K && j % 2 == jg) {
                    const u32 term = rbuf[std::size_t(buf) * CPB * 64 + p * 64 + c];
                    sA[std::size_t(c) * SA + j] = term;
                }
            }
        }
    }
}

// one full test: rows x terms x weights -> P32 vs P64 vs REF.
// tile_rc: emulate the OC31 tile-major residue layout [pair][tile][prime][512]
// (terms are packed into that layout and read back through the same RC index
// arithmetic as the kernels: (rbase2 + j)*512 + (coeff & 511), rbase2 =
// (pair*ntiles + start>>9)*K).  The mma/carry paths are layout-agnostic once
// terms are staged, so this validates the RC addressing for 64-row CTAs
// (start multiples of 64, tile = start>>9, coeff&511) incl. partial CTAs.
static int run_shape(const Shape& sh, unsigned rows, unsigned mode,
                     std::mt19937_64& rng, const char* name,
                     unsigned bits, u32 gap, const u32* q,
                     bool tile_rc = false) {
    const unsigned K = sh.K, L = sh.L, pitch = K + 1;
    const unsigned SA = sh.SA, SX = sh.SX, SXH = sh.SXH, NTH = sh.NTH;
    const unsigned KS = sh.KS, NT = sh.NT;
    const unsigned W = KS * 8;
    std::vector<u32> Wt(std::size_t(L) * pitch);
    for (auto& w : Wt) {
        if (mode == 3) w = 0xffffffffu;              // max weights
        else if (mode == 4) w = (rng() & 1) ? 0xffffffffu : 0;
        else w = u32(rng());
    }
    auto frag = bake_crt(Wt, L, pitch, const_cast<unsigned&>(KS), const_cast<unsigned&>(NT));
    // terms per row
    std::vector<u32> T(std::size_t(rows) * SA, 0);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned j = 0; j < pitch; ++j) {
            u32 v;
            switch (mode) {
                case 0: v = u32(rng()) & 0x7fffffffu; break;   // random 31-bit
                case 1: v = 0; break;                          // zeros
                case 2: v = 0xffffffffu; break;                // max u32
                case 3: v = 0x7fffffffu; break;                // max 31-bit
                case 4: v = (rng() & 1) ? 0xffffffffu : 0; break;
                case 5: v = u32(rng()); break;                 // full u32
                default: v = u32(rng()) & 0x7fffffffu; break;
            }
            T[r * SA + j] = v;
        }
    // seam stress: make half-0 columns huge so cin0 spans 2+ words
    if (mode == 6) {
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned j = 0; j < pitch; ++j)
                T[r * SA + j] = (j < pitch / 2) ? 0xffffffffu : u32(rng() & 0xffff);
    }

    // OC31 tile-major round-trip: pack T into [tile][prime][512] and read
    // back with the kernels' RC arithmetic.  Both P32 and P64 stage from
    // the same RC lambda; a mismatch here would corrupt the A operands.
    if (tile_rc) {
        const unsigned ntiles = (rows + 511) / 512;
        std::vector<u32> rc(std::size_t(ntiles) * pitch * 512, 0xdeadbeefu);
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned j = 0; j < pitch; ++j)
                rc[(std::size_t(r >> 9) * pitch + j) * 512 + (r & 511)] = T[r * SA + j];
        auto RC = [&](unsigned start, unsigned j, unsigned coeff) -> u32 {
            const u64 rbase2 = u64(start >> 9) * pitch;   // pair = 0
            return rc[(rbase2 + j) * 512u + (coeff & 511u)];
        };
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned j = 0; j < pitch; ++j) {
                // read via the CTA-start the kernel would use (64-row grid)
                const unsigned start64 = (r / 64) * 64;
                const u32 v = RC(start64, j, r);
                if (v != T[r * SA + j]) {
                    std::printf("RC ROUNDTRIP FAIL r=%u j=%u\n", r, j);
                    return 1;
                }
            }
        // stage T from RC exactly as the kernels do (per-CTA start)
        for (unsigned r = 0; r < rows; ++r)
            for (unsigned j = 0; j < pitch; ++j)
                T[r * SA + j] = RC((r / 64) * 64, j, r);

        // OC34: validate the cp.async double-buffer staging mirror against
        // the synchronous RC staging, per 64-row CTA (incl. partial CTAs).
        // terms == residues here (OC32 preweight: term = r directly).
        // The kernel stages K = pitch-1 primes (column K is the fsum
        // correction, written later by the reducer — NOT staged).
        const unsigned Kst = pitch - 1;
        for (unsigned cta = 0; cta * 64 < rows; cta++) {
            const unsigned start = cta * 64;
            const unsigned nr = (rows - start < 64) ? rows - start : 64;
            std::vector<u32> sA34(std::size_t(64) * SA, 0);
            stage34(rc.data(), u64(start >> 9) * pitch, start, rows,
                    Kst, SA, SXH, sA34);
            for (unsigned r = 0; r < nr; ++r)
                for (unsigned j = 0; j < Kst; ++j) {
                    if (sA34[r * SA + j] != T[(start + r) * SA + j]) {
                        std::printf("OC34 STAGE MISMATCH cta=%u r=%u j=%u got=%08x want=%08x\n",
                                    cta, r, j, sA34[r * SA + j], T[(start + r) * SA + j]);
                        return 1;
                    }
                }
            // column K must NOT be staged (reducer writes it from fsum)
            for (unsigned r = 0; r < 64; ++r)
                if (sA34[r * SA + Kst] != 0) {
                    std::printf("OC34 COL-K STAGED FAIL cta=%u r=%u\n", cta, r);
                    return 1;
                }
            // inactive rows (partial CTA) must be zero-filled
            for (unsigned r = nr; r < 64; ++r)
                for (unsigned j = 0; j < Kst; ++j)
                    if (sA34[r * SA + j] != 0) {
                        std::printf("OC34 ZERO-FILL FAIL cta=%u r=%u j=%u\n", cta, r, j);
                        return 1;
                    }
        }
    }

    // ---- P32: 32-row CTAs, single chain ----
    std::vector<u32> out32(std::size_t(rows) * L, 0);
    for (unsigned cta = 0; cta * 32 < rows; cta++) {
        const unsigned r0 = cta * 32;
        const unsigned nr = (rows - r0 < 32) ? rows - r0 : 32;
        std::vector<u64> sX(std::size_t(32) * SX, 0);
        for (unsigned warp = 0; warp < 4; ++warp) {
            for (unsigned nt0 = warp; nt0 < NT; nt0 += 8) {
                const unsigned ntA = nt0, ntB = nt0 + 4;
                const bool hasB = ntB < NT;
                long long C[2][2][16][8] = {};
                for (unsigned ks = 0; ks < KS; ++ks) {
                    u32 B0[32][2], B1[32][2];
                    for (int lane = 0; lane < 32; ++lane) {
                        B0[lane][0] = frag[((std::size_t(ntA) * KS + ks) * 32 + lane) * 2];
                        B0[lane][1] = frag[((std::size_t(ntA) * KS + ks) * 32 + lane) * 2 + 1];
                        if (hasB) {
                            B1[lane][0] = frag[((std::size_t(ntB) * KS + ks) * 32 + lane) * 2];
                            B1[lane][1] = frag[((std::size_t(ntB) * KS + ks) * 32 + lane) * 2 + 1];
                        } else { B1[lane][0] = B1[lane][1] = 0; }
                    }
                    for (unsigned mt = 0; mt < 2; ++mt) {
                        u32 A[32][4];
                        for (int lane = 0; lane < 32; ++lane) {
                            int g = lane >> 2, t = lane & 3;
                            unsigned rr = r0 + mt * 16 + g;
                            unsigned rr8 = rr + 8;
                            bool ok = rr < rows, ok8 = rr8 < rows;
                            A[lane][0] = ok ? T[rr * SA + ks * 8 + t] : 0;
                            A[lane][1] = ok8 ? T[rr8 * SA + ks * 8 + t] : 0;
                            A[lane][2] = ok ? T[rr * SA + ks * 8 + 4 + t] : 0;
                            A[lane][3] = ok8 ? T[rr8 * SA + ks * 8 + 4 + t] : 0;
                        }
                        mma_emul(A, B0, C[mt][0]);
                        if (hasB) mma_emul(A, B1, C[mt][1]);
                    }
                }
                for (unsigned mt = 0; mt < 2; ++mt) {
                    epi_emul(C[mt][0], ntA, SX, mt, sX.data());
                    if (hasB) epi_emul(C[mt][1], ntB, SX, mt, sX.data());
                }
            }
        }
        for (unsigned r = 0; r < nr; ++r) {
            std::vector<u32> dg(2 * SX, 0);
            carry26(sX.data() + r * SX, dg.data(), NT * 2, L + 2);
            gpu31::reduce_sum<1>(dg.data(), L, bits, gap, q);
            for (unsigned cc = 0; cc < L; ++cc) out32[(r0 + r) * L + cc] = dg[cc];
        }
    }

    // ---- P64: 64-row CTAs, two n-halves, oc27_carry_half ----
    std::vector<u32> out64(std::size_t(rows) * L, 0);
    const unsigned DSH = 2 * SXH;
    for (unsigned cta = 0; cta * 64 < rows; cta++) {
        const unsigned r0 = cta * 64;
        const unsigned nr = (rows - r0 < 64) ? rows - r0 : 64;
        std::vector<u64> cinv(64, 0);
        for (unsigned half = 0; half < 2; ++half) {
            std::vector<u64> sX(std::size_t(64) * SXH, 0);
            const unsigned UP = (NTH + 3) / 4;
            for (unsigned warp = 0; warp < 4; ++warp) {
                for (unsigned u = 0; u < UP; ++u) {
                    const unsigned ntA = half * NTH + warp + 4 * u;
                    const unsigned ntB = ntA + 8;
                    const bool hasA = (warp + 4 * u) < NTH && ntA < NT;
                    const bool hasB = (warp + 4 * u + 8) < NTH && ntB < NT;
                    if (!hasA && !hasB) continue;
                    long long C[4][2][16][8] = {};
                    for (unsigned ks = 0; ks < KS; ++ks) {
                        u32 B0[32][2], B1[32][2];
                        for (int lane = 0; lane < 32; ++lane) {
                            if (hasA) {
                                B0[lane][0] = frag[((std::size_t(ntA) * KS + ks) * 32 + lane) * 2];
                                B0[lane][1] = frag[((std::size_t(ntA) * KS + ks) * 32 + lane) * 2 + 1];
                            } else { B0[lane][0] = B0[lane][1] = 0; }
                            if (hasB) {
                                B1[lane][0] = frag[((std::size_t(ntB) * KS + ks) * 32 + lane) * 2];
                                B1[lane][1] = frag[((std::size_t(ntB) * KS + ks) * 32 + lane) * 2 + 1];
                            } else { B1[lane][0] = B1[lane][1] = 0; }
                        }
                        for (unsigned mt = 0; mt < 4; ++mt) {
                            u32 A[32][4];
                            for (int lane = 0; lane < 32; ++lane) {
                                int g = lane >> 2, t = lane & 3;
                                unsigned rr = r0 + mt * 16 + g;
                                unsigned rr8 = rr + 8;
                                bool ok = rr < rows, ok8 = rr8 < rows;
                                A[lane][0] = ok ? T[rr * SA + ks * 8 + t] : 0;
                                A[lane][1] = ok8 ? T[rr8 * SA + ks * 8 + t] : 0;
                                A[lane][2] = ok ? T[rr * SA + ks * 8 + 4 + t] : 0;
                                A[lane][3] = ok8 ? T[rr8 * SA + ks * 8 + 4 + t] : 0;
                            }
                            if (hasA) mma_emul(A, B0, C[mt][0]);
                            if (hasB) mma_emul(A, B1, C[mt][1]);
                        }
                    }
                    for (unsigned mt = 0; mt < 4; ++mt) {
                        if (hasA) epi_emul(C[mt][0], ntA - half * NTH, SXH, mt, sX.data());
                        if (hasB) epi_emul(C[mt][1], ntB - half * NTH, SXH, mt, sX.data());
                    }
                }
            }
            for (unsigned r = 0; r < nr; ++r) {
                carry27_half(sX.data() + r * SXH, NTH * 2, L, DSH, half,
                             half * NTH * 2, &cinv[r],
                             out64.data() + (r0 + r) * L, bits, gap, q);
            }
        }
    }

    // ---- REF + compare ----
    int bad32 = 0, bad64 = 0;
    std::vector<u32> ref(L);
    for (unsigned r = 0; r < rows; ++r) {
        ref_row(T.data() + r * SA, Wt.data(), pitch, L, bits, gap, q, ref.data());
        for (unsigned cc = 0; cc < L; ++cc) {
            if (out32[r * L + cc] != ref[cc]) ++bad32;
            if (out64[r * L + cc] != ref[cc]) ++bad64;
        }
    }
    std::printf("%-34s K=%3u L=%2u KS=%2u NT=%2u rows=%3u  bad32=%d bad64=%d\n",
                name, K, L, KS, NT, rows, bad32, bad64);
    return bad32 + bad64;
}

int main() {
    std::mt19937_64 rng(0x0c27);
    int total = 0;

    // host guards + exact shared bytes for the combat shape
    {
        Shape sh = make_shape(114, 55);
        std::size_t sm64 = crt_tc2_r64_smem(sh.SA, sh.SXH);
        std::size_t sm32 = crt_tc2_smem(sh.SA, sh.SX);
        std::printf("combat: SA=%u SX=%u SXH=%u NTH=%u  smem32=%zu smem64=%zu (limit 49152)\n",
                    sh.SA, sh.SX, sh.SXH, sh.NTH, sm32, sm64);
        std::printf("crt2_r64_ok(combat)=%d\n", int(crt2_r64_ok(sh.KS, sh.NT, sh.SA, sh.L)));
        if (sm64 > 49152 || !crt2_r64_ok(sh.KS, sh.NT, sh.SA, sh.L)) {
            std::printf("GUARD FAIL\n");
            return 2;
        }
    }

    // q for reduce: use the ranked point W=1747, N=1024 (55 limbs) when
    // L=55; for other L build a pseudo-q = 2^(32L) - small gap (the reduce
    // only needs bits/gap/q consistency, primality is irrelevant here).
    auto make_q = [&](unsigned L, unsigned& bits, u32& gap, std::vector<u32>& q) {
        q.assign(L, 0xffffffffu);
        q[0] = 0xfffffffbu;                 // gap 4
        gap = 4;
        bits = 32 * L - 1;
    };

    // combat shape, all data modes, rows covering full + partial CTAs
    {
        Shape sh = make_shape(114, 55);
        unsigned bits, gap; std::vector<u32> q;
        make_q(55, bits, gap, q);
        const char* names[] = {"random31", "zeros", "max-u32", "max31+maxW",
                               "alt-huge", "full-u32", "seam-stress"};
        for (unsigned mode = 0; mode <= 6; ++mode)
            for (unsigned rows : {64u, 128u, 100u})   // 100: partial 2nd CTA
                total += run_shape(sh, rows, mode, rng, names[mode], bits, gap, q.data());
    }
    // shape fuzz (guards must hold for each; skip shapes the host rejects)
    for (unsigned K : {8u, 31u, 63u, 114u, 127u}) {
        for (unsigned L : {8u, 28u, 55u}) {
            Shape sh = make_shape(K, L);
            if (!crt2_r64_ok(sh.KS, sh.NT, sh.SA, sh.L)) {
                std::printf("shape K=%u L=%u: host guard rejects (fallback crt_tc2) — skipped\n", K, L);
                continue;
            }
            unsigned bits, gap; std::vector<u32> q;
            make_q(L, bits, gap, q);
            total += run_shape(sh, 70, 0, rng, "shape-fuzz", bits, gap, q.data());
            total += run_shape(sh, 70, 6, rng, "shape-fuzz-seam", bits, gap, q.data());
        }
    }
    // random fuzz on combat shape
    {
        Shape sh = make_shape(114, 55);
        unsigned bits, gap; std::vector<u32> q;
        make_q(55, bits, gap, q);
        for (unsigned t = 0; t < 20; ++t)
            total += run_shape(sh, 64, 0, rng, "fuzz", bits, gap, q.data());
    }
    // OC31 tile-major RC layout: all modes, rows crossing tile boundaries
    // (512), partial tiles, and the combat N=65536-scale multi-tile case.
    {
        Shape sh = make_shape(114, 55);
        unsigned bits, gap; std::vector<u32> q;
        make_q(55, bits, gap, q);
        const char* names[] = {"rc-random31", "rc-zeros", "rc-max-u32", "rc-max31+maxW",
                               "rc-alt-huge", "rc-full-u32", "rc-seam-stress"};
        for (unsigned mode = 0; mode <= 6; ++mode)
            for (unsigned rows : {64u, 512u, 576u, 1024u, 1000u})
                total += run_shape(sh, rows, mode, rng, names[mode], bits, gap, q.data(), true);
        for (unsigned t = 0; t < 10; ++t)
            total += run_shape(sh, 1024, 0, rng, "rc-fuzz", bits, gap, q.data(), true);
    }
    std::printf("TOTAL BAD = %d\n", total);
    return total != 0;
}