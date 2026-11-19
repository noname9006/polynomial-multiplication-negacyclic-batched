// CPU emulation of OC28_EPICONF (agent A, encode_fwdlow256 epilogue stride).
//
// Mirrors tc.cuh::encode_fwdlow256's epilogue -> DIT -> store indexing exactly,
// parameterised only by the sT row stride S:
//   S = 264 (E2_PT = E2_ROWS + E2_ROWS/32) -- original, 264 == 8 mod 32
//   S = 272 (E2_PT = E2_ROWS + E2_ROWS/16) -- OC28,    272 == 16 mod 32
// Shared is modelled as a flat u32 array of 8*S words; every logical access is
// via idx(q, pos) = q*S + padw(pos), with padw(x) = x + (x >> 5), exactly as in
// the kernel.  Data values and twiddles are arbitrary (< p); the Montgomery
// butterfly is bit-exact, so a full-pipeline match proves the swizzle changes
// only physical addresses (bank placement), never logical results.
//
// Also reports the epilogue's shared-bank multiplicity per stride: the q=0/q=1
// row writes of one warp must go from 2-way (S=264) to 1-way (S=272).
//
// Build: g++ -O2 -std=c++17 -o oc28_epiconf_emul oc28_epiconf_emul.cpp
// Run:   ./oc28_epiconf_emul
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
typedef uint32_t u32;
typedef uint64_t u64;

static inline u32 padw(unsigned x) { return x + (x >> 5); }
static inline u32 mont(u32 a, u32 b, u32 p, u32 ni) {
    u64 t = u64(a) * b;
    u32 m = u32(t) * ni;
    u32 u = u32((t + u64(m) * p) >> 32);
    return u >= p ? u - p : u;
}
static inline u32 addm(u32 a, u32 b, u32 p) { u32 s = a + b; return s >= p ? s - p : s; }
static inline u32 subm(u32 a, u32 b, u32 p) { return a >= b ? a - b : a + p - b; }
static inline void bf(u32& a, u32& b, u32 w, u32 p, u32 ni) {
    u32 v = mont(b, w, p, ni);
    u32 na = addm(a, v, p), nb = subm(a, v, p);
    a = na; b = nb;
}
// TC22_MERGED twiddle index: TWI(m,j) = (1<<m) - 1 + j
static inline unsigned TWI(unsigned m, unsigned j) { return (1u << m) - 1u + j; }

static const unsigned E2_ROWS = 256;
static const unsigned EF_PR = 8;       // 8 primes per CTA (j = 0..3, 2 each)

// One CTA-group's work for 8 primes x 256 positions, parameterised by stride S.
// Fills logical rows via the (warp, mt, g, t) -> rl epilogue map, runs the exact
// DIT sequence, then reads back via padw(pos).
static void run_pipeline(u32* sh, unsigned S,
                         const std::vector<u32>& init /* EF_PR*E2_ROWS */,
                         const std::vector<u32>& tw,   /* TWI-indexed twiddles */
                         u32 p, u32 ni) {
    // ---- epilogue: place init[q*256 + rl] into sh[idx(q, rl)] ----
    for (unsigned tid = 0; tid < 256; ++tid) {
        const unsigned warp = tid >> 5, lane = tid & 31;
        const unsigned g = lane >> 2, t = lane & 3;
        const unsigned rowbase = warp * 32;
        const bool odd = t & 1;
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned q = j * 2 + (t >> 1);
            for (unsigned mt = 0; mt < 2; ++mt) {
                const unsigned rl = rowbase + mt * 16 + g + (odd ? 8u : 0u);
                sh[q * S + padw(rl)] = init[q * E2_ROWS + rl];
            }
        }
    }
    // ---- DIT stages 0..5: two radix-8 passes (m = 0..2, 3..5) ----
    for (unsigned m = 0; m < 6; m += 3) {
        const unsigned H = 1u << m;
        for (unsigned q = 0; q < EF_PR; ++q) {
            u32* row = sh + q * S;
            for (unsigned gi = 0; gi < 32; ++gi) {
                const unsigned w = gi & (H - 1);
                const unsigned x0 = (gi - w) * 8 + w;
                u32 v[8];
                for (unsigned e = 0; e < 8; ++e) v[e] = row[padw(x0 + e * H)];
                for (unsigned st = 0; st < 3; ++st) {
                    const unsigned span = 1u << st;
                    for (unsigned e = 0; e < 8; ++e) {
                        if (e & span) continue;
                        const unsigned j = w + (e & (span - 1)) * H;
                        bf(v[e], v[e + span], tw[TWI(m + st, j)], p, ni);
                    }
                }
                for (unsigned e = 0; e < 8; ++e) row[padw(x0 + e * H)] = v[e];
            }
        }
    }
    // ---- DIT stages 6..7: radix-4, 2 reps x 128 threads ----
    for (unsigned rep = 0; rep < 2; ++rep) {
        for (unsigned tid = 0; tid < 256; ++tid) {
            const unsigned q = (tid >> 6) + rep * 4, w = tid & 63;
            u32* row = sh + q * S;
            const unsigned i0 = padw(w), i1 = padw(w + 64);
            const unsigned i2 = padw(w + 128), i3 = padw(w + 192);
            u32 v0 = row[i0], v1 = row[i1], v2 = row[i2], v3 = row[i3];
            bf(v0, v1, tw[TWI(6, w)], p, ni);
            bf(v2, v3, tw[TWI(6, w)], p, ni);
            bf(v0, v2, tw[TWI(7, w)], p, ni);
            bf(v1, v3, tw[TWI(7, w + 64)], p, ni);
            row[i0] = v0; row[i1] = v1; row[i2] = v2; row[i3] = v3;
        }
    }
}

// Epilogue bank multiplicity: for each (j, mt) the 32 lanes of a warp write
// q = 2j + (t>>1) and rl = rowbase + mt*16 + g + (odd?8:0).  A uniform warp
// offset (rowbase, mt, j*S base) shifts all lanes equally, so checking warp 0
// suffices; we still sweep all j and mt.  Returns the max lanes per bank.
static unsigned epilogue_bank_multiplicity(unsigned S) {
    unsigned worst = 0;
    for (unsigned j = 0; j < 4; ++j) {
        for (unsigned mt = 0; mt < 2; ++mt) {
            int bank[32] = {0};
            for (unsigned lane = 0; lane < 32; ++lane) {
                const unsigned g = lane >> 2, t = lane & 3;
                const unsigned q = j * 2 + (t >> 1);
                const unsigned rl = mt * 16 + g + ((t & 1) ? 8u : 0u);
                const unsigned addr = q * S + padw(rl);
                bank[addr % 32]++;
            }
            for (int b : bank) if ((unsigned)b > worst) worst = b;
        }
    }
    return worst;
}

int main() {
    std::mt19937 rng(20261120u);
    std::uniform_int_distribution<u32> dist(0, (1u << 30) - 1);
    const u32 p = (1u << 31) - 1;                 // odd modulus > 2^30
    u32 inv = 1; for (int i = 0; i < 5; ++i) inv *= 2u - p * inv;
    const u32 neg_inv = u32(0u - inv);

    const unsigned TW = 1u << 9;                   // TWI(8,.) max < 512
    std::vector<u32> tw(TW);
    for (auto& x : tw) x = dist(rng) % p;
    std::vector<u32> init(EF_PR * E2_ROWS);
    for (auto& x : init) x = dist(rng) % p;

    std::vector<u32> ref264(EF_PR * E2_ROWS), ref272(EF_PR * E2_ROWS);
    for (unsigned S : {264u, 272u}) {
        std::vector<u32> sh((std::size_t)EF_PR * S, 0xdeadbeefu);
        run_pipeline(sh.data(), S, init, tw, p, neg_inv);
        std::vector<u32>& dst = (S == 264) ? ref264 : ref272;
        for (unsigned q = 0; q < EF_PR; ++q)
            for (unsigned pos = 0; pos < E2_ROWS; ++pos)
                dst[q * E2_ROWS + pos] = sh[q * S + padw(pos)];
    }
    unsigned mism = 0;
    for (std::size_t i = 0; i < ref264.size(); ++i) if (ref264[i] != ref272[i]) ++mism;
    printf("logical words compared: %u, mismatches: %u\n",
           (unsigned)ref264.size(), mism);
    printf("epilogue bank multiplicity: S=264 -> %u-way, S=272 -> %u-way\n",
           epilogue_bank_multiplicity(264), epilogue_bank_multiplicity(272));
    if (mism != 0) { printf("FAIL\n"); return 1; }
    if (epilogue_bank_multiplicity(264) != 2 || epilogue_bank_multiplicity(272) != 1) {
        printf("FAIL: unexpected bank multiplicity\n"); return 1;
    }
    printf("OK: OC28 swizzle is bit-exact and conflict-free\n");
    return 0;
}
