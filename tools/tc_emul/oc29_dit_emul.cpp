// CPU emulation of OC29_DITSHFL (agent A, encode_fwdlow256 DIT in registers).
//
// Mirrors tc.cuh::encode_fwdlow256's DIT for one group (8 primes x 256 points):
//   path 0 (OC29_DITSHFL=0): current shared path -- two radix-8 passes shared
//                            (m=0..2, 3..5) + radix-4 shared (m=6,7), 3 barriers.
//   path 1 (OC29_DITSHFL=1): new register path -- one warp per prime, row lives
//                            as v[e] at position lane+32e; stages 0..4 via
//                            __shfl_xor_sync, stages 5..7 in-thread radix-8.
// Twiddles come from the same table TWI(s,j) = (1<<s)-1+j as build_ntt_tables.
// Every butterfly uses the exact Montgomery add/sub as gpu31::butterfly, so a
// bit-exact match over all 8 primes x 256 points proves the new DIT is
// interchangeable with the shared one.
//
// Build: g++ -O2 -std=c++17 -o oc29_dit_emul oc29_dit_emul.cpp
// Run:   ./oc29_dit_emul
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
typedef uint32_t u32;
typedef uint64_t u64;
static const u32 MASK = 0xffffffffu;

struct Prime { u32 p, neg_inv, r2, one; };

static inline u32 padw(unsigned x) { return x + (x >> 5); }
static inline u32 mont(u32 a, u32 b, const Prime& pr) {
    u64 t = u64(a) * b;
    u32 m = u32(t) * pr.neg_inv;
    u32 u = u32((t + u64(m) * pr.p) >> 32);
    return u >= pr.p ? u - pr.p : u;
}
static inline u32 add(u32 a, u32 b, u32 p) { u32 s = a + b; return s >= p ? s - p : s; }
static inline u32 sub(u32 a, u32 b, u32 p) { return a >= b ? a - b : a + p - b; }
static inline void bf(u32& a, u32& b, u32 w, const Prime& pr) {
    u32 u = a, v = mont(b, w, pr);
    a = add(u, v, pr.p); b = sub(u, v, pr.p);
}
static inline unsigned TWI(unsigned s, unsigned j) { return (1u << s) - 1u + j; }

static const unsigned E2_ROWS = 256, EF_PR = 8;
static const unsigned S = 272;   // E2_PT with OC28 active at base 2d5cca8

// -------- shared path (flag 0): two radix-8 passes + radix-4 --------
// Operates in place on sh[0..S-1] (one prime row, logical pos via padw).
static void dit_shared(u32* sh, const Prime& pr, const u32* tw) {
    for (unsigned m = 0; m < 6; m += 3) {
        const unsigned H = 1u << m;
        for (unsigned gi = 0; gi < 32; ++gi) {
            const unsigned w = gi & (H - 1);
            const unsigned x0 = (gi - w) * 8 + w;
            u32 v[8];
            for (unsigned e = 0; e < 8; ++e) v[e] = sh[padw(x0 + e * H)];
            for (unsigned st = 0; st < 3; ++st) {
                const unsigned span = 1u << st;
                for (unsigned e = 0; e < 8; ++e) {
                    if (e & span) continue;
                    const unsigned j = w + (e & (span - 1)) * H;
                    bf(v[e], v[e + span], tw[TWI(m + st, j)], pr);
                }
            }
            for (unsigned e = 0; e < 8; ++e) sh[padw(x0 + e * H)] = v[e];
        }
    }
    for (unsigned w = 0; w < 64; ++w) {
        const unsigned i0 = padw(w), i1 = padw(w + 64);
        const unsigned i2 = padw(w + 128), i3 = padw(w + 192);
        u32 v0 = sh[i0], v1 = sh[i1], v2 = sh[i2], v3 = sh[i3];
        const u32 t1 = tw[TWI(6, w)];
        bf(v0, v1, t1, pr); bf(v2, v3, t1, pr);
        bf(v0, v2, tw[TWI(7, w)], pr);
        bf(v1, v3, tw[TWI(7, w + 64)], pr);
        sh[i0] = v0; sh[i1] = v1; sh[i2] = v2; sh[i3] = v3;
    }
}

// -------- register path (flag 1): shfl stages + in-thread radix-8 --------
// Simulates all 32 lanes; row[logical pos] -> out[logical pos].
static void dit_shfl_row(const u32* row, u32* out, const Prime& pr, const u32* tw) {
    u32 vreg[32][8];
    for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned e = 0; e < 8; ++e) vreg[lane][e] = row[lane + 32 * e];

    for (unsigned s = 0; s < 5; ++s) {
        const unsigned Hb = 1u << s;
        u32 old[32][8];   // __shfl_xor_sync reads all partners' pre-stage values
        for (unsigned lane = 0; lane < 32; ++lane)
            for (unsigned e = 0; e < 8; ++e) old[lane][e] = vreg[lane][e];
        for (unsigned lane = 0; lane < 32; ++lane) {
            const u32 wt = tw[TWI(s, lane & (Hb - 1))];
            const bool upper = (lane & Hb) != 0;
            for (unsigned e = 0; e < 8; ++e) {
                const u32 other = old[lane ^ Hb][e];
                if (!upper) {
                    const u32 u = old[lane][e], t = mont(other, wt, pr);
                    vreg[lane][e] = add(u, t, pr.p);
                } else {
                    const u32 u = other, t = mont(old[lane][e], wt, pr);
                    vreg[lane][e] = sub(u, t, pr.p);
                }
            }
        }
    }
    for (unsigned st = 0; st < 3; ++st) {
        const unsigned span = 1u << st, stage = 5 + st;
        for (unsigned lane = 0; lane < 32; ++lane)
            for (unsigned e = 0; e < 8; ++e) {
                if (e & span) continue;
                const unsigned jtw = lane + (e & (span - 1)) * 32;
                bf(vreg[lane][e], vreg[lane][e + span], tw[TWI(stage, jtw)], pr);
            }
    }
    for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned e = 0; e < 8; ++e) out[lane + 32 * e] = vreg[lane][e];
}

int main() {
    std::mt19937 rng(20261120u);
    std::uniform_int_distribution<u32> dist(0, (1u << 30) - 1);
    Prime pr{}; pr.p = (1u << 31) - 1;
    u32 inv = 1; for (int i = 0; i < 5; ++i) inv *= 2u - pr.p * inv;
    pr.neg_inv = u32(0u - inv);

    const unsigned TWS = 512;
    std::vector<u32> tw(TWS);
    for (auto& x : tw) x = dist(rng) % pr.p;

    for (unsigned trial = 0; trial < 20; ++trial) {
        std::vector<u32> sh((std::size_t)EF_PR * S, 0xdeadbeefu);
        std::vector<std::vector<u32>> logical(EF_PR, std::vector<u32>(E2_ROWS));
        for (unsigned q = 0; q < EF_PR; ++q)
            for (unsigned pos = 0; pos < E2_ROWS; ++pos) {
                const u32 x = dist(rng) % pr.p;
                logical[q][pos] = x;
                sh[q * S + padw(pos)] = x;    // epilogue wrote via padw
            }
        for (unsigned q = 0; q < EF_PR; ++q)
            dit_shared(sh.data() + q * S, pr, tw.data());

        std::vector<std::vector<u32>> regout(EF_PR, std::vector<u32>(E2_ROWS));
        for (unsigned q = 0; q < EF_PR; ++q)
            dit_shfl_row(logical[q].data(), regout[q].data(), pr, tw.data());

        unsigned mism = 0;
        for (unsigned q = 0; q < EF_PR; ++q)
            for (unsigned pos = 0; pos < E2_ROWS; ++pos)
                if (sh[q * S + padw(pos)] != regout[q][pos]) ++mism;
        if (mism) { printf("FAIL trial %u: %u mismatches\n", trial, mism); return 1; }
    }
    printf("OK: OC29 register DIT == current shared DIT, 20 trials x "
           "8 primes x 256 points = 40960 words, 0 mismatches\n");
    return 0;
}
