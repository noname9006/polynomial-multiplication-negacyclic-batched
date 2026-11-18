// OC15 staging verifier: bit-exact A/B comparison of crt_tc2's staging paths.
//
// The correctness claim of OC15 is that the 64-bit coalesced pair-load staging
// produces EXACTLY the same sA[][] (and therefore the same MMA/carry/store
// result) as the original one-u32-per-load TC11 staging.  This emulator models
// both staging paths over the full thread map (tid -> c=tid&31, jg=tid>>5),
// including the fixed-point correction slots, and requires bit equality for
// many random shapes and boundary values.
//
// Build:
//   g++ -std=c++17 -O2 -I . -I tools/nvsyntax -include tools/nvsyntax/extra.keep \
//       -x c++ tools/tc_emul/oc15_staging_emul.cpp -o build-cpu/oc15_staging_emul.exe
#include "tc.cuh"
#include <cstdio>
#include <random>
#include <vector>
using namespace tc;

static u32 ref_mont(u32 a, u32 b, u32 p, u32 neg_inv) {
    u64 t = u64(a) * b;
    u32 m = u32(t) * neg_inv;
    u32 u = u32((t + u64(m) * p) >> 32);
    return u >= p ? u - p : u;
}

int main() {
    std::mt19937_64 rng(0x0c15);
    int total_bad = 0, cells = 0, cases = 0;
    const unsigned NG = 128 / tc::CRT2_ROWS;   // = 4
    const unsigned Ls[] = {55, 17, 8};
    const unsigned Ks[] = {114, 119, 8, 1, 7, 16, 33, 120};

    for (unsigned L : Ls) for (unsigned K : Ks) {
        ++cases;
        const unsigned pitch = K + 1;
        const unsigned cKS = (pitch + 7) / 8;
        const unsigned SA = cKS * 8 + 8;
        const unsigned Wd = cKS * 8;

        std::vector<u32> primes(K), neg_inv(K), corr(K), slotc(K);
        for (unsigned j = 0; j < K; ++j) {
            u32 p = (1u << 30) + 2u * (rng() % (1u << 29)) + 1u;
            primes[j] = p;
            u32 inv = 1;
            for (int it = 0; it < 5; ++it) inv *= 2u - p * inv;
            neg_inv[j] = u32(0u - inv);
            corr[j] = u32((u64(1) << 62) / p);
            slotc[j] = u32(rng()) % p;
        }
        // residues[prime][row] for 32 rows
        std::vector<u32> res(std::size_t(K) * tc::CRT2_ROWS);
        for (unsigned j = 0; j < K; ++j)
            for (unsigned r = 0; r < tc::CRT2_ROWS; ++r) {
                u32 x = u32(rng());
                switch (rng() % 8) {
                    case 0: x = 0; break;
                    case 1: x = primes[j] - 1; break;   // boundary term = p-1
                    case 2: x = 1; break;
                    default: break;
                }
                res[j * tc::CRT2_ROWS + r] = x;
            }

        // ---- Baseline staging (TC11): one u32 load per prime per thread ----
        std::vector<u32> sA_base(std::size_t(tc::CRT2_ROWS) * SA, 0);
        u64 fsum_base[128] = {};   // per-thread (tid = jg*32 + c)
        for (unsigned jg = 0; jg < NG; ++jg)
            for (unsigned c = 0; c < tc::CRT2_ROWS; ++c) {
                const unsigned tid = jg * tc::CRT2_ROWS + c;
                for (unsigned j0 = jg; j0 < K; j0 += 8 * NG) {
                    u32 r[8];
                    for (unsigned q = 0; q < 8; ++q) {
                        unsigned j = j0 + NG * q;
                        r[q] = (j < K) ? res[j * tc::CRT2_ROWS + c] : 0u;
                    }
                    for (unsigned q = 0; q < 8; ++q) {
                        unsigned j = j0 + NG * q;
                        if (j < K) {
                            u32 term = ref_mont(r[q], slotc[j], primes[j], neg_inv[j]);
                            sA_base[c * SA + j] = term;
                            fsum_base[tid] += (u64(term) * corr[j]) >> 30;
                        }
                    }
                }
                for (unsigned j = K + 1 + jg; j < Wd; j += NG) sA_base[c * SA + j] = 0;
                sA_base[c * SA + Wd + 2 * jg] = u32(fsum_base[tid]);
                sA_base[c * SA + Wd + 2 * jg + 1] = u32(fsum_base[tid] >> 32);
            }
        // combine correction: jg==0 thread of each row
        for (unsigned c = 0; c < tc::CRT2_ROWS; ++c) {
            u64 s = 0;
            for (unsigned jg = 0; jg < NG; ++jg) s += fsum_base[jg * tc::CRT2_ROWS + c];
            sA_base[c * SA + K] = u32((s + (u64(1) << 31)) >> 32);
        }

        // ---- OC15 staging (pair u64 loads, per-row fsum) ----
        std::vector<u32> sA_new(std::size_t(tc::CRT2_ROWS) * SA, 0);
        u64 fsum_new[128] = {};
        for (unsigned jg = 0; jg < NG; ++jg)
            for (unsigned c = 0; c < tc::CRT2_ROWS; ++c) {
                const unsigned tid = jg * tc::CRT2_ROWS + c;
                const bool is_lo = (c & 1u) == 0;
                const unsigned c2 = c & ~1u;
                const bool hi_valid = (c2 + 1) < tc::CRT2_ROWS;
                u64 fh = 0;
                for (unsigned j0 = jg; j0 < K; j0 += 8 * NG) {
                    u32 r0[8] = {}, r1[8] = {};
                    for (unsigned q = 0; q < 8; ++q) {
                        unsigned j = j0 + NG * q;
                        if (!is_lo || j >= K) continue;
                        u64 rp = u64(res[j * tc::CRT2_ROWS + c2]) |
                                 (u64(res[j * tc::CRT2_ROWS + (c2 + 1)]) << 32);
                        r0[q] = u32(rp);
                        r1[q] = hi_valid ? u32(rp >> 32) : 0u;
                    }
                    for (unsigned q = 0; q < 8; ++q) {
                        unsigned j = j0 + NG * q;
                        if (j >= K || !is_lo) continue;
                        u32 t0 = ref_mont(r0[q], slotc[j], primes[j], neg_inv[j]);
                        sA_new[c2 * SA + j] = t0;
                        fsum_new[jg * tc::CRT2_ROWS + c2] += (u64(t0) * corr[j]) >> 30;
                        if (hi_valid) {
                            u32 t1 = ref_mont(r1[q], slotc[j], primes[j], neg_inv[j]);
                            sA_new[(c2 + 1) * SA + j] = t1;
                            fh += (u64(t1) * corr[j]) >> 30;
                        }
                    }
                }
                if (is_lo && hi_valid) fsum_new[jg * tc::CRT2_ROWS + (c2 + 1)] += fh;
            }
        // Each row's total fsum across all jg; write correction slot K.
        for (unsigned row = 0; row < tc::CRT2_ROWS; ++row) {
            u64 s = 0;
            for (unsigned jg = 0; jg < NG; ++jg) s += fsum_new[jg * tc::CRT2_ROWS + row];
            sA_new[row * SA + K] = u32((s + (u64(1) << 31)) >> 32);
        }

        // ---- Compare bit-exactly over all terms + correction cells ----
        for (unsigned row = 0; row < tc::CRT2_ROWS; ++row) {
            for (unsigned j = 0; j <= K; ++j) {
                ++cells;
                if (sA_base[row * SA + j] != sA_new[row * SA + j]) {
                    if (++total_bad <= 5)
                        std::printf("  mismatch L=%u K=%u row=%u j=%u base=%u new=%u\n",
                                    L, K, row, j, sA_base[row * SA + j], sA_new[row * SA + j]);
                }
            }
        }
    }
    std::printf("OC15 staging A/B bad=%d cells=%d cases=%d\n", total_bad, cells, cases);
    return total_bad == 0 ? 0 : 1;
}
