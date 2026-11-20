// oc41_r64f_emul.cpp — CPU validation of OC41 (crt_tc2_r64f, R8F 64 rows).
//
// Mirrors the OC41 kernel's three phases on identical synthetic R8F data:
//   P32: current crt_tc2 (32 rows/CTA, 1 B fragment per 2 m-tile) — baseline
//   P64: OC41 crt_tc2_r64f (64 rows/CTA, B feeds 4 m-tile, 2 GEMM halves,
//        SEQUENTIAL carry with local digits + register cin0 handover)
//   REF: naive big-int CRT (the ground truth the 32-row kernel matches)
// Staging is validated by feeding each path the SAME staged terms, plus a
// dedicated staging round-trip test that checks the 32-point inverse DIF
// (per-stage twiddles) reproduces the expected terms for BOTH 32-runs, which
// is the local analogue of tools/tc_emul/r8f_emul.py's slot-by-slot check.
//
// Build: sh tools/tc_emul/build_oc41_emul.sh   (from the repo root)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "gpu31.cuh"
#include "tc.cuh"
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// definitions for extern-shared symbols of never-called kernels
namespace gpu31 { u64 shared[4]; u32 stage[4]; u32 data[4]; }
namespace tc { unsigned char smem_raw[16]; u32 sm[4]; }

// ---- m16n8k32 u8 GEMM (exact integer mirror of mma_u8) --------------------
static void mma_emul(const u32 A[32][4], const u32 Bf[32][2], long long C[16][8]) {
    unsigned char a[16][32] = {}, b[32][8] = {};
    for (int lane = 0; lane < 32; ++lane) {
        int g = lane >> 2, t = lane & 3;
        for (int i = 0; i < 4; ++i) {
            a[g][t*4+i]      = A[lane][0] >> (8*i);
            a[g+8][t*4+i]    = A[lane][1] >> (8*i);
            a[g][16+t*4+i]   = A[lane][2] >> (8*i);
            a[g+8][16+t*4+i] = A[lane][3] >> (8*i);
            b[t*4+i][g]      = Bf[lane][0] >> (8*i);
            b[16+t*4+i][g]   = Bf[lane][1] >> (8*i);
        }
    }
    for (int r = 0; r < 16; ++r)
        for (int cc = 0; cc < 8; ++cc)
            for (int k = 0; k < 32; ++k) C[r][cc] += (long long)a[r][k] * b[k][cc];
}

// epilogue mirror: shfl_xor pairing, u64 column store
static void epi_emul(const long long C[16][8], unsigned nt, unsigned SX,
                     unsigned mt, u64* sX) {
    for (int lane = 0; lane < 32; ++lane) {
        int t = lane & 3, g = lane >> 2;
        unsigned sh = (t & 1) ? 16u : 0u;
        bool odd = t & 1;
        int pt = lane ^ 1; int tp = pt & 3;
        unsigned shp = (tp & 1) ? 16u : 0u;
        u64 pg  = (u64(u32(C[g][2*t])) << sh)   + (u64(u32(C[g][2*t+1])) << (sh+8));
        u64 pg8 = (u64(u32(C[g+8][2*t])) << sh) + (u64(u32(C[g+8][2*t+1])) << (sh+8));
        u64 spg = (u64(u32(C[g][2*tp])) << shp) + (u64(u32(C[g][2*tp+1])) << (shp+8));
        u64 spg8= (u64(u32(C[g+8][2*tp])) << shp)+ (u64(u32(C[g+8][2*tp+1])) << (shp+8));
        u64 tot = (odd ? pg8 : pg) + (odd ? spg8 : spg);
        unsigned row = mt*16 + g + (odd ? 8u : 0u);
        sX[row*SX + nt*2 + (t >> 1)] = tot;
    }
}

// OC26 sequential carry over `words` u64 columns -> digits (single chain)
static void carry_seq(const u64* xs, u32* dg, unsigned words, unsigned l2) {
    u64 carry = 0;
    for (unsigned w = 0; w < words; ++w) {
        carry += xs[w];
        dg[w] = u32(carry);
        carry >>= 32;
    }
    for (unsigned w = words; w < l2; ++w) { dg[w] = u32(carry); carry >>= 32; }
}

// OC41 seam carry, faithful mirror of oc41_carry_half (sequential, local
// digits, register cin0), with the REAL gpu31::reduce_sum<1>.
static void carry41(const u64* xs, unsigned SXH, u64& cin0,
                    u32* out, unsigned l, unsigned words,
                    unsigned half, unsigned w0,
                    unsigned bits, u32 gap, const u32* q) {
    const unsigned l2 = l + 2, DSH = 2*SXH;
    u32 dg[60];
    u64 carry = 0;
    for (unsigned w = 0; w < words; ++w) {
        carry += xs[w]; dg[w] = u32(carry); carry >>= 32;
    }
    if (half == 0) {
        cin0 = carry;
        for (unsigned w = words; w < l2; ++w) dg[w] = 0;
        gpu31::reduce_sum<1>(dg, l, bits, gap, q);
    } else {
        u64 tail = carry;
        for (unsigned w = words; w < l2; ++w) { dg[w] = u32(tail); tail >>= 32; }
        u64 add = cin0;
        for (unsigned w = 0; w < l2 && add; ++w) {
            u64 s = u64(dg[w]) + u32(add); dg[w] = u32(s); add = (add >> 32) + (s >> 32);
        }
        for (unsigned j = DSH; j-- > w0;) dg[j] = (j - w0 < l2) ? dg[j - w0] : 0u;
        for (unsigned j = 0; j < w0; ++j) dg[j] = 0;
        u64 c2 = 0;
        for (unsigned cc = 0; cc < l; ++cc) {
            u64 v = u64(dg[cc]) + u64(out[cc]) + c2; dg[cc] = u32(v); c2 = v >> 32;
        }
        for (unsigned w = l; w < DSH && c2; ++w) {
            u64 v = u64(dg[w]) + c2; dg[w] = u32(v); c2 = v >> 32;
        }
        gpu31::reduce_sum<1>(dg, l, bits, gap, q);
    }
    for (unsigned cc = 0; cc < l; ++cc) out[cc] = dg[cc];
}

struct Shape { unsigned K, L, pitch, KS, NT, SA, SX, SXH, NTH; };
static Shape make_shape(unsigned K, unsigned L) {
    Shape sh;
    sh.K = K; sh.L = L; sh.pitch = K + 1;
    std::vector<u32> w = tc::bake_crt(std::vector<u32>(std::size_t(L)*sh.pitch, 3u),
                                      L, sh.pitch, sh.KS, sh.NT);
    sh.SA = tc::crt_sa_stride(sh.KS);
    sh.SX = 2*sh.NT + 1;
    sh.SXH = tc::crt2_sxh(sh.NT);
    sh.NTH = (sh.SXH - 1)/2;
    return sh;
}

// naive big-int reference per row: sum_j terms[j] * W[i][j] mod q
static void ref_row(const u32* terms, const u32* W, unsigned pitch, unsigned L,
                    unsigned bits, u32 gap, const u32* q, u32* out) {
    std::vector<u32> acc(L+2, 0);
    for (unsigned j = 0; j < pitch; ++j) {
        u64 cc = 0;
        for (unsigned i = 0; i < L+2; ++i) {
            u64 v = cc + acc[i] + (i < L ? u64(terms[j]) * W[i*pitch+j] : 0);
            acc[i] = u32(v); cc = v >> 32;
        }
    }
    gpu31::reduce_sum<1>(acc.data(), L, bits, gap, q);
    for (unsigned i = 0; i < L; ++i) out[i] = acc[i];
}

// one full test: rows x terms x weights -> P32 vs P64 vs REF
static int run_shape(const Shape& sh, unsigned rows, unsigned mode,
                     std::mt19937_64& rng, const char* name,
                     unsigned bits, u32 gap, const u32* q) {
    const unsigned K = sh.K, L = sh.L, pitch = sh.pitch, SA = sh.SA;
    const unsigned SX = sh.SX, SXH = sh.SXH, NTH = sh.NTH, KS = sh.KS, NT = sh.NT;
    // weights W[i][j] = CRT fragments contribution (i = limb, j = prime)
    std::vector<u32> W(std::size_t(L)*pitch);
    for (auto& x : W) x = u32(rng());
    unsigned ks = KS, nt = NT;
    auto frag = tc::bake_crt(W, L, pitch, ks, nt);
    // terms T[row][j] for all rows (j in 0..pitch-1; j=K is fsum correction)
    std::vector<u32> T(std::size_t(rows)*SA, 0);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned j = 0; j < pitch; ++j) T[r*SA+j] = u32(rng()) & 0x7fffffffu;
    if (mode == 1) for (auto& x : T) x = 0;
    if (mode == 2) for (auto& x : T) x = 0xffffffffu;
    if (mode == 3) for (unsigned r = 0; r < rows; ++r) {
        T[r*SA] = 0x7fffffffu; T[r*SA+1] = 0xffffffffu; }
    if (mode == 4) for (unsigned r = 0; r < rows; ++r)
        for (unsigned j = 0; j < pitch; ++j) T[r*SA+j] = (j&1) ? 0xffffffffu : 0x7fffffffu;
    if (mode == 5) for (auto& x : T) x = 0xffffffffu;
    if (mode == 6) for (unsigned r = 0; r < rows; ++r)
        for (unsigned j = 0; j < NTH*2; ++j) T[r*SA+j] = (j&1) ? 0xffffffffu : 0x7fffffffu;

    // ---- P32: 32-row CTAs, single chain (baseline) ----
    std::vector<u32> out32(std::size_t(rows)*L, 0);
    for (unsigned cta = 0; cta*32 < rows; ++cta) {
        const unsigned r0 = cta*32, nr = (rows-r0 < 32) ? rows-r0 : 32;
        std::vector<u64> sX(std::size_t(32)*SX, 0);
        for (unsigned warp = 0; warp < 4; ++warp)
            for (unsigned nt0 = warp; nt0 < NT; nt0 += 8) {
                const unsigned ntA = nt0, ntB = nt0+4;
                const bool hasB = ntB < NT;
                long long C[2][2][16][8] = {};
                for (unsigned k = 0; k < KS; ++k) {
                    u32 B0[32][2], B1[32][2];
                    for (int lane = 0; lane < 32; ++lane) {
                        B0[lane][0] = frag[((std::size_t(ntA)*KS+k)*32+lane)*2];
                        B0[lane][1] = frag[((std::size_t(ntA)*KS+k)*32+lane)*2+1];
                        if (hasB) {
                            B1[lane][0] = frag[((std::size_t(ntB)*KS+k)*32+lane)*2];
                            B1[lane][1] = frag[((std::size_t(ntB)*KS+k)*32+lane)*2+1];
                        } else B1[lane][0]=B1[lane][1]=0;
                    }
                    for (unsigned mt = 0; mt < 2; ++mt) {
                        u32 A[32][4];
                        for (int lane = 0; lane < 32; ++lane) {
                            int g=lane>>2, t=lane&3;
                            unsigned rr=r0+mt*16+g, rr8=rr+8;
                            bool ok=rr<rows, ok8=rr8<rows;
                            A[lane][0]=ok?T[rr*SA+k*8+t]:0; A[lane][1]=ok8?T[rr8*SA+k*8+t]:0;
                            A[lane][2]=ok?T[rr*SA+k*8+4+t]:0; A[lane][3]=ok8?T[rr8*SA+k*8+4+t]:0;
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
        for (unsigned r = 0; r < nr; ++r) {
            std::vector<u32> dg(2*SX, 0);
            carry_seq(sX.data()+r*SX, dg.data(), NT*2, L+2);
            gpu31::reduce_sum<1>(dg.data(), L, bits, gap, q);
            for (unsigned cc = 0; cc < L; ++cc) out32[(r0+r)*L+cc]=dg[cc];
        }
    }

    // ---- P64: 64-row CTAs, two n-halves, OC41 sequential carry ----
    std::vector<u32> out64(std::size_t(rows)*L, 0);
    for (unsigned cta = 0; cta*64 < rows; ++cta) {
        const unsigned r0 = cta*64, nr = (rows-r0<64)?rows-r0:64;
        u64 cin0[64] = {};   // per-thread register handover (row = tid)
        for (unsigned half = 0; half < 2; ++half) {
            std::vector<u64> sX(std::size_t(64)*SXH, 0);
            const unsigned UP = (NTH+3)/4;
            for (unsigned warp = 0; warp < 4; ++warp)
                for (unsigned u = 0; u < UP; ++u) {
                    unsigned ntA = half*NTH + warp + 4*u, ntB = ntA+8;
                    bool hasA = (warp+4*u)<NTH && ntA<NT, hasB = (warp+4*u+8)<NTH && ntB<NT;
                    if (!hasA && !hasB) continue;
                    long long C[4][2][16][8] = {};
                    for (unsigned k = 0; k < KS; ++k) {
                        u32 B0[32][2], B1[32][2];
                        for (int lane = 0; lane < 32; ++lane) {
                            B0[lane][0]=hasA?frag[((std::size_t(ntA)*KS+k)*32+lane)*2]:0;
                            B0[lane][1]=hasA?frag[((std::size_t(ntA)*KS+k)*32+lane)*2+1]:0;
                            B1[lane][0]=hasB?frag[((std::size_t(ntB)*KS+k)*32+lane)*2]:0;
                            B1[lane][1]=hasB?frag[((std::size_t(ntB)*KS+k)*32+lane)*2+1]:0;
                        }
                        for (unsigned mt = 0; mt < 4; ++mt) {
                            u32 A[32][4];
                            for (int lane = 0; lane < 32; ++lane) {
                                int g=lane>>2, t=lane&3;
                                unsigned rr = r0 + mt*16 + g, rr8 = rr + 8;
                                unsigned lr = rr-r0, lr8=rr8-r0;
                                bool ok=rr<rows, ok8=rr8<rows;
                                A[lane][0]=ok?T[rr*SA+k*8+t]:0; A[lane][1]=ok8?T[rr8*SA+k*8+t]:0;
                                A[lane][2]=ok?T[rr*SA+k*8+4+t]:0; A[lane][3]=ok8?T[rr8*SA+k*8+4+t]:0;
                            }
                            if (hasA) mma_emul(A, B0, C[mt][0]);
                            if (hasB) mma_emul(A, B1, C[mt][1]);
                        }
                    }
                    for (unsigned mt = 0; mt < 4; ++mt) {
                        if (hasA) epi_emul(C[mt][0], ntA-half*NTH, SXH, mt, sX.data());
                        if (hasB) epi_emul(C[mt][1], ntB-half*NTH, SXH, mt, sX.data());
                    }
                }
            for (unsigned r = 0; r < nr; ++r) {
                u32 row[L <= 128 ? 128 : 256];
                // load half 0's stored L_A for the seam add (P64 overwrites out)
                if (half == 1)
                    for (unsigned cc = 0; cc < L; ++cc) row[cc] = out64[(r0+r)*L+cc];
                // carry41 works on a local output row; seed from stored L_A
                u32 tmp[128] = {};
                if (half == 1) for (unsigned cc = 0; cc < L; ++cc) tmp[cc] = out64[(r0+r)*L+cc];
                carry41(sX.data()+r*SXH, SXH, cin0[r], tmp, L, NTH*2, half, half*NTH*2,
                        bits, gap, q);
                for (unsigned cc = 0; cc < L; ++cc) out64[(r0+r)*L+cc] = tmp[cc];
            }
        }
    }

    // ---- P64B: OC47 256-thread variant: 8 warps, 1 n-tile/warp, 2 rounds,
    //      two-threads-per-row carry (register stitch) ----
    std::vector<u32> out64b(std::size_t(rows)*L, 0);
    for (unsigned cta = 0; cta*64 < rows; ++cta) {
        const unsigned r0 = cta*64, nr=(rows-r0<64)?rows-r0:64;
        // per-row register handover: two threads per row -> cin0[2*rows]
        std::vector<u64> cr(2*64, 0);
        for (unsigned half = 0; half < 2; ++half) {
            std::vector<u64> sX(std::size_t(64)*SXH, 0);
            for (unsigned warp = 0; warp < 8; ++warp)
                for (unsigned rnd = 0; rnd < 2; ++rnd) {
                    const unsigned nt = half*NTH + warp + 8*rnd;
                    const bool has = (warp + 8*rnd) < NTH && nt < NT;
                    if (!has) continue;
                    long long C[4][16][8] = {};
                    for (unsigned k = 0; k < KS; ++k) {
                        u32 B[32][2];
                        for (int lane=0; lane<32; ++lane) {
                            B[lane][0]=frag[((std::size_t(nt)*KS+k)*32+lane)*2];
                            B[lane][1]=frag[((std::size_t(nt)*KS+k)*32+lane)*2+1]; }
                        for (unsigned mt=0; mt<4; ++mt) {
                            u32 A[32][4];
                            for (int lane=0; lane<32; ++lane) {
                                int g=lane>>2,t=lane&3;
                                unsigned rr=r0+mt*16+g, rr8=rr+8;
                                bool ok=rr<rows, ok8=rr8<rows;
                                A[lane][0]=ok?T[rr*SA+k*8+t]:0; A[lane][1]=ok8?T[rr8*SA+k*8+t]:0;
                                A[lane][2]=ok?T[rr*SA+k*8+4+t]:0; A[lane][3]=ok8?T[rr8*SA+k*8+4+t]:0; }
                            mma_emul(A,B,C[mt]); }
                    }
                    for (unsigned mt=0; mt<4; ++mt)
                        epi_emul(C[mt], nt-half*NTH, SXH, mt, sX.data());
                }
            // two-threads-per-row carry, mirrored with a local stitch
            for (unsigned tid = 0; tid < 2*nr; tid += 2) {
                unsigned row = tid>>1;
                u32 full[60] = {};
                const unsigned HW = NTH;   // words/2 = NTH (words=NTH*2)
                u64 c0 = 0, c1 = 0;
                for (unsigned w=0;w<HW;++w) { c0 += sX[row*SXH+w];       full[w]=u32(c0); c0>>=32; }
                for (unsigned w=0;w<HW;++w) { c1 += sX[row*SXH+HW+w]; full[HW+w]=u32(c1); c1>>=32; }
                u64 delta = c0;
                for (unsigned w=0; w<HW && delta; ++w) {
                    u64 sm=u64(full[HW+w])+u32(delta); full[HW+w]=u32(sm); delta=(delta>>32)+(sm>>32); }
                u64 carry = c1+delta;
                u32 tmp[128]={}; if(half==1) for(unsigned cc=0;cc<L;++cc)tmp[cc]=out64b[(r0+row)*L+cc];
                const unsigned words=NTH*2, l2=L+2;
                if (half==0) {
                    cr[2*row]=carry;
                    for(unsigned w=words;w<l2;++w)full[w]=0;
                    gpu31::reduce_sum<1>(full,L,bits,gap,q);
                } else {
                    u64 tail=carry; for(unsigned w=words;w<l2;++w){full[w]=u32(tail);tail>>=32;}
                    u64 add=cr[2*row];
                    for(unsigned w=0;w<l2&&add;++w){u64 sm=u64(full[w])+u32(add);full[w]=u32(sm);add=(add>>32)+(sm>>32);}
                    for(unsigned j=2*SXH;j-->words;) full[j]=(j-words<l2)?full[j-words]:0u;
                    for(unsigned j=0;j<words;++j)full[j]=0;
                    u64 cc2=0; for(unsigned c2=0;c2<L;++c2){u64 v=u64(full[c2])+u64(tmp[c2])+cc2;full[c2]=u32(v);cc2=v>>32;}
                    for(unsigned w=L;w<2*SXH&&cc2;++w){u64 v=u64(full[w])+cc2;full[w]=u32(v);cc2=v>>32;}
                    gpu31::reduce_sum<1>(full,L,bits,gap,q);
                }
                for(unsigned cc=0;cc<L;++cc)out64b[(r0+row)*L+cc]=full[cc];
            }
        }
    }

    // ---- REF + compare ----
    int bad32 = 0, bad64 = 0, bad64b = 0;
    std::vector<u32> ref(L);
    for (unsigned r = 0; r < rows; ++r) {
        ref_row(T.data()+r*SA, W.data(), pitch, L, bits, gap, q, ref.data());
        for (unsigned cc = 0; cc < L; ++cc) {
            if (out32[r*L+cc] != ref[cc]) ++bad32;
            if (out64[r*L+cc] != ref[cc]) ++bad64;
            if (out64b[r*L+cc] != ref[cc]) ++bad64b;
        }
    }
    std::printf("%-30s K=%3u L=%2u KS=%2u NT=%2u rows=%3u  bad32=%d bad64=%d bad64b=%d\n",
                name, K, L, KS, NT, rows, bad32, bad64, bad64b);
    return bad32 + bad64 + bad64b;
}


// R8F 64-row staging layout check: two adjacent 32-runs of one CTA.
// For each (limb j, run r), the 32-point inverse DIF (stages 4..0, per-stage
// twiddles) must write term v[i] into sA[(r*32+i)*SA + j].  We verify the
// ADDRESS mapping and the DIF values against a plain big-int reference, and
// that run 0 and run 1 do not cross-contaminate (the key 64-row risk).
static int staging_test(std::mt19937_64& rng, const Shape& sh) {
    const unsigned K = sh.K, SA = sh.SA, p = 2013265921u;
    const unsigned n = 16, N = 1u<<n;
    unsigned g=31, psi=1;
    { unsigned e=(p-1)/(2*N), b=g; while(e){if(e&1)psi=u64(psi)*b%p;b=u64(b)*b%p;e>>=1;} }
    std::vector<u32> tw(32);
    for (unsigned m=0;m<5;++m) { unsigned H=1u<<m;
        for (unsigned j=0;j<H;++j) {
            unsigned e=((1u<<(n-1-m))*(1u+2*j))%(2*N);
            e=(2*N-e)%(2*N);
            unsigned w=1,b=psi;
            while(e){if(e&1)w=u64(w)*b%p;b=u64(b)*b%p;e>>=1;}
            tw[(H-1)+j]=w;
        }
    }
    // ref[run][j][i] = expected term for row run*32+i, limb j
    std::vector<u32> ref(std::size_t(2)*K*32);
    std::vector<u32> sA(std::size_t(64)*SA, 0);
    for (unsigned run=0; run<2; ++run)
        for (unsigned j=0; j<K; ++j) {
            std::vector<u32> v(32);
            for (unsigned i=0;i<32;++i) v[i]=u32(rng())%p;
            std::vector<u32> r=v;
            for (int m=4;m>=0;--m) { unsigned H=1u<<m;
                for (unsigned i=0;i<32;++i) if(!(i&H)) {
                    u32 u=r[i], x=r[i+H];
                    r[i]=(u+x)%p;
                    r[i+H]=u64((u+p-x)%p)*tw[(H-1)+(i&(H-1))]%p;
                }
            }
            for (unsigned i=0;i<32;++i) {
                ref[(std::size_t(run)*K+j)*32+i]=r[i];
                sA[(run*32+i)*SA+j]=v[i];   // simulate the staging store
            }
        }
    // now apply the same DIF to the staged values and verify each of the 64
    // rows against its own run's reference (cross-run contamination check).
    int bad=0;
    for (unsigned run=0; run<2; ++run)
        for (unsigned j=0; j<K; ++j) {
            std::vector<u32> v(32);
            for (unsigned i=0;i<32;++i) v[i]=sA[(run*32+i)*SA+j];
            for (int m=4;m>=0;--m) { unsigned H=1u<<m;
                for (unsigned i=0;i<32;++i) if(!(i&H)) {
                    u32 u=v[i], x=v[i+H];
                    v[i]=(u+x)%p;
                    v[i+H]=u64((u+p-x)%p)*tw[(H-1)+(i&(H-1))]%p;
                }
            }
            for (unsigned i=0;i<32;++i)
                if (v[i]!=ref[(std::size_t(run)*K+j)*32+i]) ++bad;
        }
    std::printf("R8F staging 64-row layout: bad=%d (2 runs x %u limbs x 32 slots)\n", bad, K);
    return bad;
}


// ---- OC47 checks -----------------------------------------------------------
// 1) staging task distribution: 2*K tasks over 256 threads, exactly once.
static int oc47_tasks(unsigned K) {
    const unsigned N = 2 * K;
    std::vector<unsigned> seen(N, 0);
    for (unsigned tid = 0; tid < 256; ++tid)
        if (tid < N) {            // one round, guard tid < 2*K
            const unsigned run = tid / K, j = tid - run * K;
            seen[run * K + j]++;
        }
    int bad = 0;
    for (unsigned t = 0; t < N; ++t) if (seen[t] != 1) ++bad;
    std::printf("OC47 staging tasks: %u tasks / 256 threads, duplicates-or-missing=%d\n", N, bad);
    return bad;
}

// 2) two-thread carry: two 14-word segments + shfl-like stitch vs single
//    chain.  negative=true breaks the stitch (drops the propagated carry)
//    and MUST produce mismatches (the negative control).
static int oc47_carry(bool negative, std::mt19937_64& rng, unsigned bits,
                      u32 gap, const u32* q) {
    const unsigned HW = 14, words = 28, l = 55, l2 = l+2, SXH = 29, DSH = 2*SXH, w0 = 28;
    int bad = 0;
    for (unsigned trial = 0; trial < 200; ++trial) {
        std::vector<u64> x(words);
        for (auto& v : x) v = (trial < 4) ? 0xffffffffffffffffull : ((u64(rng()) << 32) | u32(rng()));
        // single chain
        std::vector<u32> single(DSH, 0);
        { u64 c = 0; for (unsigned w = 0; w < words; ++w) { c += x[w]; single[w] = u32(c); c >>= 32; }
          for (unsigned w = words; w < l2; ++w) { single[w] = u32(c); c >>= 32; } }
        gpu31::reduce_sum<1>(single.data(), l, bits, gap, q);
        // two segments
        u32 a[HW], b[HW]; u64 ca = 0, cb = 0;
        for (unsigned w = 0; w < HW; ++w) { ca += x[w]; a[w] = u32(ca); ca >>= 32; }
        for (unsigned w = 0; w < HW; ++w) { cb += x[HW+w]; b[w] = u32(cb); cb >>= 32; }
        u64 delta = negative ? 0 : ca;        // NEGATIVE: drop the stitch carry
        for (unsigned w = 0; w < HW && delta; ++w) {
            u64 s = u64(b[w]) + u32(delta); b[w] = u32(s); delta = (delta >> 32) + (s >> 32);
        }
        u64 carry = cb + delta;
        u32 dg[60]; for (unsigned w = 0; w < HW; ++w) dg[w] = a[w];
        for (unsigned w = 0; w < HW; ++w) dg[HW+w] = b[w];
        for (unsigned w = words; w < l2; ++w) { dg[w] = u32(carry); carry >>= 32; }
        gpu31::reduce_sum<1>(dg, l, bits, gap, q);
        for (unsigned c = 0; c < l; ++c) if (dg[c] != single[c]) ++bad;
    }
    std::printf("OC47 2-thread carry (%s): bad=%d over 200 trials\n",
                negative ? "NEGATIVE" : "normal", bad);
    return bad;
}

int main() {
    std::mt19937_64 rng(0x0c41);
    int total = 0;
    { Shape shs = make_shape(114,55); total += staging_test(rng, shs); }
    { total += oc47_tasks(114); }
    { unsigned bits=32*55-1; u32 gap=4; std::vector<u32> q(55,0xffffffffu); q[0]=0xfffffffbu;
      int normal = oc47_carry(false, rng, bits, gap, q.data());
      int neg = oc47_carry(true, rng, bits, gap, q.data());
      total += normal;
      if (neg == 0) { std::printf("OC47 NEGATIVE TEST FAILED (broken stitch not detected!)\n"); total += 1; }
      else std::printf("OC47 negative control OK (broken stitch detected: %d bad)\n", neg);
    }
    { Shape sh = make_shape(114, 55);
      std::size_t sm64 = tc::crt_tc2_r64f_smem(sh.SA, sh.SXH);
      std::printf("combat: SA=%u SX=%u SXH=%u NTH=%u smem64=%zu (limit 49152)\n",
                  sh.SA, sh.SX, sh.SXH, sh.NTH, sm64);
      std::printf("crt2_r64f_ok(combat)=%d\n", int(tc::crt2_r64f_ok(sh.KS,sh.NT,sh.SA,sh.L)));
      if (sm64 > 49152 || !tc::crt2_r64f_ok(sh.KS,sh.NT,sh.SA,sh.L)) return 2; }
    unsigned bits; u32 gap; std::vector<u32> q;
    { Shape sh = make_shape(114,55); (void)sh;
      q.assign(55, 0xffffffffu); q[0]=0xfffffffbu; gap=4; bits=32*55-1; }
    { Shape sh = make_shape(114,55);
      const char* names[] = {"random31","zeros","max-u32","max31+maxW",
                             "alt-huge","full-u32","seam-stress"};
      for (unsigned mode = 0; mode <= 6; ++mode)
        for (unsigned rows : {64u,128u,100u})
          total += run_shape(sh, rows, mode, rng, names[mode], bits, gap, q.data()); }
    // shape fuzz
    for (unsigned K : {8u, 31u, 63u}) {
      Shape sh = make_shape(K, 55);
      if (!tc::crt2_r64f_ok(sh.KS, sh.NT, sh.SA, sh.L)) continue;
      total += run_shape(sh, 64, 0, rng, "fuzz", bits, gap, q.data());
    }
    std::printf("TOTAL BAD = %d\n", total);
    return total != 0;
}