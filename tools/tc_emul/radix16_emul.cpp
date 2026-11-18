// CPU emulation of the OC10 fused_high radix-16 path (TC22 merged-psi NTT).
//
// This file mirrors tc.cuh::fused_high line by line (same stage partition,
// same group/within indices, same twiddle lookups, same butterfly order) and
// compares the whole TC NTT pipeline against a naive negacyclic product.
// The only simplification is that the 8 warps / 8 warps-per-row decomposition
// of a CTA is replayed sequentially (groups are disjoint, so the result is
// identical), and inv_low's warp-shuffle stages are replayed as shared stages.
//
// Build:  g++ -O2 -std=c++17 -o radix16_emul radix16_emul.cpp
// Run:    ./radix16_emul <logN> [r16=1]
//         logN = 11..16, r16=0 selects the OC8 radix-8 reference path.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
typedef uint32_t u32; typedef uint64_t u64;
struct Prime { u32 p, neg_inv, r2, one; };
static u32 P;
u32 mulm(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 mont(u32 a, u32 b, const Prime& pr) {
    u64 t = u64(a) * b;
    u32 m = u32(t) * pr.neg_inv;
    u32 u = u32((t + u64(m) * pr.p) >> 32);
    return u >= pr.p ? u - pr.p : u;
}
u32 add(u32 a, u32 b, u32 p) { u32 s = a + b; return s >= p ? s - p : s; }
u32 sub(u32 a, u32 b, u32 p) { return a >= b ? a - b : a + p - b; }
void bf(u32& a, u32& b, u32 w, const Prime& pr) {
    u32 u = a, v = mont(b, w, pr);
    a = add(u, v, pr.p); b = sub(u, v, pr.p);
}
void ifb(u32& a, u32& b, u32 w, const Prime& pr) {
    u32 u = a, v = b;
    a = add(u, v, pr.p); b = mont(sub(u, v, pr.p), w, pr);
}
u32 rev(u32 x, unsigned bits) { u32 r = 0; for (unsigned i = 0; i < bits; ++i) { r = (r << 1) | (x & 1); x >>= 1; } return r; }
u32 pw(u32 a, u64 e) { u32 r = 1; while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; } return r; }
bool isprime(u32 p) {
    if ((p & 1) == 0) return p == 2;
    for (u32 d = 3; u64(d) * d <= p; d += 2) if (p % d == 0) return false;
    return true;
}

int main(int argc, char** argv) {
    const unsigned logN = argc > 1 ? (unsigned)atoi(argv[1]) : 12;
    const bool R16 = !(argc > 2 && atoi(argv[2]) == 0);
    const unsigned N = 1u << logN, T = 512, HC = N / T, logH = logN - 9;
    if (logH < 2) { printf("logN must be >= 11\n"); return 1; }
    P = ((1u << 31) - 2) / (2 * N) * (2 * N) + 1;
    while (!isprime(P)) P -= 2 * N;
    u32 inv = 1; for (int j = 0; j < 5; ++j) inv *= 2 - P * inv;
    const u32 one = u32((u64(1) << 32) % P), r2 = mulm(one, one);
    Prime pr{P, u32(0 - inv), r2, one};
    u32 psi = 0;
    for (u32 a = 2; !psi; ++a) { u32 c = pw(a, (u64(P - 1) / (2 * N))); if (pw(c, N) == P - 1) psi = c; }
    // psi^e (Montgomery) and psi^-e (Montgomery), as build_ntt_tables.
    auto fpsi = [&](std::size_t e) { return mont(pw(psi, e % (2u * N)), r2, pr); };
    auto ipsi = [&](std::size_t e) { e %= 2u * N; return e == 0 ? mont(1u, r2, pr) : P - fpsi(N - e); };

    // ---- tables: low packed (512/prime), high tf/ti (7*512+127/prime) ----
    // ---- tables: low packed (512/prime), high tf/ti (7*512+127/prime) ----
    const std::size_t TC_LT = 512, TWS = 7 * T + 127;
    std::vector<u32> pf(TC_LT, 0), pi(TC_LT, 0), tf(TWS, 0), ti(TWS, 0);
    for (unsigned m = 0; m < 9; ++m) {
        const unsigned half = 1u << m;
        for (unsigned j = 0; j < half; ++j) {
            const std::size_t e = (std::size_t(1) << (logN - 1 - m)) * (1 + 2 * j);
            pf[(half - 1) + j] = fpsi(e); pi[(half - 1) + j] = ipsi(e);
        }
    }
    for (unsigned st = 0; st < logH; ++st) {
        const unsigned H = 1u << st;
        for (unsigned low = 0; low < T; ++low) {
            const std::size_t e = (std::size_t(1) << (logN - 10 - st)) * (1 + 2 * low);
            tf[st * T + low] = fpsi(e); ti[st * T + low] = ipsi(e);
        }
        for (unsigned w = 0; w < H; ++w) {
            const std::size_t e = std::size_t(w) << (logN - st);
            tf[7 * T + (H - 1) + w] = fpsi(e); ti[7 * T + (H - 1) + w] = ipsi(e);
        }
    }

    // ---- random plain inputs, encoded to Montgomery residues at brv slots ----
    std::vector<u32> A(N), B(N);
    std::mt19937_64 g(0x0c10ULL + logN);
    for (unsigned i = 0; i < N; ++i) { A[i] = u32(g() % P); B[i] = u32(g() % P); }
    std::vector<u32> ra(N), rb(N), rc(N);
    for (unsigned i = 0; i < N; ++i) {
        ra[i] = mont(A[rev(i, logN)], r2, pr);
        rb[i] = mont(B[rev(i, logN)], r2, pr);
    }

    // forward DIT low stages in 512-tiles (encode_fwdlow), TC22 TWI indexing
    for (unsigned bx = 0; bx < N / T; ++bx) {
        u32* a = &ra[bx * T]; u32* b = &rb[bx * T];
        for (unsigned m = 0; m < 9; ++m) {
            const unsigned half = 1u << m;
            for (unsigned t = 0; t < 256; ++t) {
                const unsigned j = t & (half - 1);
                const unsigned first = 2 * (t - j) + j, second = first + half;
                const u32 w = pf[(half - 1) + j];
                bf(a[first], a[second], w, pr);
                bf(b[first], b[second], w, pr);
            }
        }
    }
    const unsigned n16 = logH >> 2, rs = n16 << 2;
    const unsigned n8 = (logH - rs) / 3, s8 = rs + n8 * 3;

    // ---- fused_high, one low column at a time (exact kernel indexing) ----
    for (unsigned low = 0; low < T; ++low) {
        std::vector<u32> sa(HC), sb(HC);
        for (unsigned h = 0; h < HC; ++h) { sa[h] = ra[h * T + low]; sb[h] = rb[h * T + low]; }
        u32 waf[7], wai[7];
        for (unsigned q = 0; q < 7; ++q) {
            waf[q] = q < logH ? tf[q * T + low] : 0u;
            wai[q] = q < logH ? ti[q * T + low] : 0u;
        }
        const u32* tfB = tf.data() + 7 * T;
        const u32* tiB = ti.data() + 7 * T;
        auto TWF = [&](unsigned s, unsigned within) { return mont(waf[s], tfB[(1u << s) - 1 + within], pr); };
        auto TWI = [&](unsigned s, unsigned within) { return mont(wai[s], tiB[(1u << s) - 1 + within], pr); };

        // ---------- forward DIT ----------
        if (R16) {
            for (unsigned q = 0; q < 4; ++q) {
                if (q >= n16) break;
                const unsigned s = q << 2, H = 1u << s;
                for (unsigned gg = 0; gg < HC / 16; ++gg) {
                    const unsigned w = gg & (H - 1);
                    const unsigned r0 = (gg - w) * 16 + w;
                    u32 av[16], bv[16];
                    for (unsigned i = 0; i < 16; ++i) { av[i] = sa[r0 + i * H]; bv[i] = sb[r0 + i * H]; }
                    const u32 t1 = TWF(s, w);
                    u32 t2[2], t4[4], t8[8];
                    for (unsigned k = 0; k < 2; ++k) t2[k] = TWF(s+1, w + k * H);
                    for (unsigned k = 0; k < 4; ++k) t4[k] = TWF(s+2, w + k * H);
                    for (unsigned k = 0; k < 8; ++k) t8[k] = TWF(s+3, w + k * H);
                    for (unsigned i = 0; i < 16; i += 2) { bf(av[i], av[i+1], t1, pr); bf(bv[i], bv[i+1], t1, pr); }
                    for (unsigned k = 0; k < 2; ++k)
                        for (unsigned i = k; i < 16; i += 4) { bf(av[i], av[i+2], t2[k], pr); bf(bv[i], bv[i+2], t2[k], pr); }
                    for (unsigned k = 0; k < 4; ++k)
                        for (unsigned i = k; i < 16; i += 8) { bf(av[i], av[i+4], t4[k], pr); bf(bv[i], bv[i+4], t4[k], pr); }
                    for (unsigned i = 0; i < 8; ++i) { bf(av[i], av[i+8], t8[i], pr); bf(bv[i], bv[i+8], t8[i], pr); }
                    for (unsigned i = 0; i < 16; ++i) { sa[r0 + i * H] = av[i]; sb[r0 + i * H] = bv[i]; }
                }
            }
            for (unsigned q = 0; q < 2; ++q) {
                if (q >= n8) break;
                const unsigned s = rs + q * 3, H = 1u << s;
                for (unsigned gg = 0; gg < HC / 8; ++gg) {
                    const unsigned w = gg & (H - 1);
                    const unsigned r0 = (gg - w) * 8 + w;
                    u32 av[8], bv[8];
                    for (unsigned i = 0; i < 8; ++i) { av[i] = sa[r0 + i * H]; bv[i] = sb[r0 + i * H]; }
                    const u32 t1 = TWF(s, w);
                    const u32 t2a = TWF(s+1, w), t2b = TWF(s+1, w + H);
                    u32 t4[4];
                    for (unsigned i = 0; i < 4; ++i) t4[i] = TWF(s+2, w + i * H);
                    for (unsigned i = 0; i < 8; i += 2) { bf(av[i], av[i+1], t1, pr); bf(bv[i], bv[i+1], t1, pr); }
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned i = (j < 2) ? j : j + 2;
                        const u32 tw = (i & 1) ? t2b : t2a;
                        bf(av[i], av[i+2], tw, pr); bf(bv[i], bv[i+2], tw, pr);
                    }
                    for (unsigned i = 0; i < 4; ++i) { bf(av[i], av[i+4], t4[i], pr); bf(bv[i], bv[i+4], t4[i], pr); }
                    for (unsigned i = 0; i < 8; ++i) { sa[r0 + i * H] = av[i]; sb[r0 + i * H] = bv[i]; }
                }
            }
            for (unsigned k = 0; k < 3; ++k) {
                if (k >= logH - s8) break;
                const unsigned s = s8 + k, H = 1u << s;
                for (unsigned hh = 0; hh < HC / 2; ++hh) {
                    const unsigned within = hh & (H - 1);
                    const unsigned first = 2 * (hh - within) + within, second = first + H;
                    const u32 w = TWF(s, within);
                    bf(sa[first], sa[second], w, pr); bf(sb[first], sb[second], w, pr);
                }
            }
        } else {
            // OC8 radix-8 reference path (fallback / cross-check).
            for (unsigned s = 0; s + 2 < 7; s += 3) {
                if (s + 2 >= logH) break;
                const unsigned H = 1u << s;
                for (unsigned gg = 0; gg < HC / 8; ++gg) {
                    const unsigned w = gg & (H - 1);
                    const unsigned r0 = (gg - w) * 8 + w;
                    u32 av[8], bv[8];
                    for (unsigned i = 0; i < 8; ++i) { av[i] = sa[r0 + i * H]; bv[i] = sb[r0 + i * H]; }
                    const u32 t1 = TWF(s, w);
                    const u32 t2a = TWF(s+1, w), t2b = TWF(s+1, w + H);
                    u32 t4[4];
                    for (unsigned i = 0; i < 4; ++i) t4[i] = TWF(s+2, w + i * H);
                    for (unsigned i = 0; i < 8; i += 2) { bf(av[i], av[i+1], t1, pr); bf(bv[i], bv[i+1], t1, pr); }
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned i = (j < 2) ? j : j + 2;
                        const u32 tw = (i & 1) ? t2b : t2a;
                        bf(av[i], av[i+2], tw, pr); bf(bv[i], bv[i+2], tw, pr);
                    }
                    for (unsigned i = 0; i < 4; ++i) { bf(av[i], av[i+4], t4[i], pr); bf(bv[i], bv[i+4], t4[i], pr); }
                    for (unsigned i = 0; i < 8; ++i) { sa[r0 + i * H] = av[i]; sb[r0 + i * H] = bv[i]; }
                }
            }
            for (unsigned k = 0; k < 3; ++k) {
                if (k >= logH % 3) break;
                const unsigned s = logH - (logH % 3) + k, H = 1u << s;
                for (unsigned hh = 0; hh < HC / 2; ++hh) {
                    const unsigned within = hh & (H - 1);
                    const unsigned first = 2 * (hh - within) + within, second = first + H;
                    const u32 w = TWF(s, within);
                    bf(sa[first], sa[second], w, pr); bf(sb[first], sb[second], w, pr);
                }
            }
        }

        // ---------- pointwise ----------
        for (unsigned h = 0; h < HC; ++h) sa[h] = mont(sa[h], sb[h], pr);

        // ---------- inverse DIF ----------
        if (R16) {
            for (unsigned k = 0; k < 3; ++k) {
                if (k >= logH - s8) break;
                const unsigned s = logH - 1 - k, H = 1u << s;
                for (unsigned hh = 0; hh < HC / 2; ++hh) {
                    const unsigned within = hh & (H - 1);
                    const unsigned first = 2 * (hh - within) + within, second = first + H;
                    const u32 w = TWI(s, within);
                    ifb(sa[first], sa[second], w, pr);
                }
            }
            for (unsigned q = 0; q < 2; ++q) {
                if (q >= n8) break;
                const unsigned s = s8 - 3 * (q + 1), H = 1u << s;
                for (unsigned gg = 0; gg < HC / 8; ++gg) {
                    const unsigned w = gg & (H - 1);
                    const unsigned r0 = (gg - w) * 8 + w;
                    u32 av[8];
                    for (unsigned i = 0; i < 8; ++i) av[i] = sa[r0 + i * H];
                    const u32 t4[4] = {TWI(s+2,w), TWI(s+2,w+H), TWI(s+2,w+2*H), TWI(s+2,w+3*H)};
                    const u32 t2a = TWI(s+1, w), t2b = TWI(s+1, w + H), t1 = TWI(s, w);
                    for (unsigned i = 0; i < 4; ++i) ifb(av[i], av[i+4], t4[i], pr);
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned i = (j < 2) ? j : j + 2;
                        ifb(av[i], av[i+2], (i & 1) ? t2b : t2a, pr);
                    }
                    for (unsigned i = 0; i < 8; i += 2) ifb(av[i], av[i+1], t1, pr);
                    for (unsigned i = 0; i < 8; ++i) sa[r0 + i * H] = av[i];
                }
            }
            for (unsigned q = 0; q < 4; ++q) {
                if (q >= n16) break;
                const unsigned s = (n16 - 1 - q) << 2, H = 1u << s;
                for (unsigned gg = 0; gg < HC / 16; ++gg) {
                    const unsigned w = gg & (H - 1);
                    const unsigned r0 = (gg - w) * 16 + w;
                    u32 av[16];
                    for (unsigned i = 0; i < 16; ++i) av[i] = sa[r0 + i * H];
                    u32 t2[2], t4[4], t8[8];
                    for (unsigned k = 0; k < 2; ++k) t2[k] = TWI(s+1, w + k * H);
                    for (unsigned k = 0; k < 4; ++k) t4[k] = TWI(s+2, w + k * H);
                    for (unsigned k = 0; k < 8; ++k) t8[k] = TWI(s+3, w + k * H);
                    const u32 t1 = TWI(s, w);
                    for (unsigned i = 0; i < 8; ++i) ifb(av[i], av[i+8], t8[i], pr);
                    for (unsigned k = 0; k < 4; ++k)
                        for (unsigned i = k; i < 16; i += 8) ifb(av[i], av[i+4], t4[k], pr);
                    for (unsigned k = 0; k < 2; ++k)
                        for (unsigned i = k; i < 16; i += 4) ifb(av[i], av[i+2], t2[k], pr);
                    for (unsigned i = 0; i < 16; i += 2) ifb(av[i], av[i+1], t1, pr);
                    for (unsigned i = 0; i < 16; ++i) sa[r0 + i * H] = av[i];
                }
            }
        } else {
            for (int s = 6; s >= 0; --s) {
                if ((unsigned)s >= logH) continue;
                const unsigned H = 1u << s;
                for (unsigned hh = 0; hh < HC / 2; ++hh) {
                    const unsigned within = hh & (H - 1);
                    const unsigned first = 2 * (hh - within) + within, second = first + H;
                    const u32 w = TWI(s, within);
                    ifb(sa[first], sa[second], w, pr);
                }
            }
        }
        for (unsigned h = 0; h < HC; ++h) rc[h * T + low] = sa[h];
    }

    // ---- inv_low: 9 inverse DIF low stages in 512-tiles ----
    for (unsigned bx = 0; bx < N / T; ++bx) {
        u32* a = &rc[bx * T];
        for (int m = 8; m >= 0; --m) {
            const unsigned half = 1u << m;
            for (unsigned t = 0; t < 256; ++t) {
                const unsigned j = t & (half - 1);
                const unsigned first = 2 * (t - j) + j, second = first + half;
                const u32 w = pi[(half - 1) + j];
                ifb(a[first], a[second], w, pr);
            }
        }
    }

    // ---- compare to naive negacyclic product (single prime, inv_m = 1) ----
    const u32 ninv = pw(N, P - 2);
    int bad = 0;
    for (unsigned k = 0; k < N; ++k) {
        u64 acc = 0;
        for (unsigned i = 0; i < N; ++i) {
            const unsigned j = (k + N - i) % N;
            const u64 prod = mulm(A[i], B[j]);
            acc = (i <= k) ? (acc + prod) % P : (acc + P - prod) % P;
        }
        const u32 got = mont(rc[rev(k, logN)], ninv, pr);
        if (got != u32(acc)) ++bad;
    }
    printf("radix16_emul: N=%u logH=%u r16=%d bad=%d/%u\n", N, logH, (int)R16, bad, N);
    return bad != 0;
}
