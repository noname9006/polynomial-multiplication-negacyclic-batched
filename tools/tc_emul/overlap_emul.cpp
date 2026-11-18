// Emulator of the OC25 encode_fwdlow group pipeline (E1).
//
// Mirrors the exact group / prime / shared-buffer indexing of tc.cuh:
//   NT primes are processed in groups of EF_NTB (=4 n-tiles = 8 primes);
//   sR/sB are double buffered (buf = g&1, nbuf = (g+1)&1);
//   epilogue of group g writes sT, DIT of group g reads/writes sT, store
//   writes the group's 8 prime residue rows;
//   OC25_OVERLAP=1 issues the MMA chunks of group g+1 between the three
//   radix-8 DIT passes of group g, with the accumulator reused after the
//   previous group's epilogue.
//
// MMA/epilogue/DIT are modelled with simple modular arithmetic -- the point is
// to verify that every group is processed exactly once, with the right buffer
// and prime row, so the per-prime final residues match the sequential path.
//
// Build: g++ -O2 -std=c++17 -o overlap_emul overlap_emul.cpp
// Run:   ./overlap_emul [NT] [OVERLAP]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
typedef uint32_t u32; typedef uint64_t u64;
constexpr u32 P = 2147483629u;   // prime used only as a modular ring here
constexpr unsigned EF_NTB = 4, EF_PR = 8, KS = 7, T = 512, THREADS = 512;

static u32 addm(u32 a, u32 b) { u32 s = a + b; return s >= P ? s - P : s; }

int main(int argc, char** argv) {
    const unsigned NT = argc > 1 ? (unsigned)atoi(argv[1]) : 28;
    const bool OVERLAP = !(argc > 2 && atoi(argv[2]) == 0);
    const unsigned NG = (NT + EF_NTB - 1) / EF_NTB;

    // Plain inputs for every prime and position (deterministic).
    std::vector<u32> A(std::size_t(NT) * T), B(std::size_t(NT) * T);
    for (unsigned k = 0; k < NT; ++k)
        for (unsigned i = 0; i < T; ++i) {
            A[k * T + i] = (u32)(0x9e3779b9u * (k * T + i + 1));
            B[k * T + i] = (u32)(0x85ebca6bu * (k * T + i + 7));
        }

    // model mma -> epilogue for one group: residue = A*B*(mix) mod P
    auto group_mma = [&](unsigned g0, std::vector<u32>& acc) {
        for (unsigned q = 0; q < EF_PR; ++q) {
            const unsigned k = g0 + q;
            for (unsigned pos = 0; pos < T; ++pos) {
                const u32 a = k < NT ? A[k * T + pos] : 0u;
                const u32 b = k < NT ? B[k * T + pos] : 0u;
                acc[q * T + pos] = k < NT ? u32((u64)a * (b | 1u) % P) : 0u;
            }
        }
    };
    auto group_epi = [&](std::vector<u32>& acc, std::vector<u32>& sT) {
        sT = acc;   // epilogue writes the group's residues into sT
    };
    // model one radix-8 DIT pass (m = 0,3,6) on sT
    auto group_dit = [&](std::vector<u32>& sT) {
        for (unsigned m = 0; m < 9; m += 3) {
            const unsigned H = 1u << m;
            for (unsigned gi = 0; gi < 64; ++gi) {
                const unsigned w = gi & (H - 1), x0 = (gi - w) * 8 + w;
                for (unsigned st = 0; st < 3; ++st) {
                    const unsigned span = 1u << st;
                    for (unsigned e = 0; e < 8; ++e) {
                        if (e & span) continue;
                        u32& lo = sT[x0 + e * H];
                        u32& hi = sT[x0 + (e + span) * H];
                        const u32 u = lo, v = hi;
                        lo = addm(u, v); hi = u >= v ? u - v : u + P - v;
                    }
                }
            }
        }
    };

    std::vector<u32> out_seq(std::size_t(NT) * T, 0), out_pipe(std::size_t(NT) * T, 0);
    auto run = [&](bool overlap, std::vector<u32>& out) {
        std::vector<u32> sT(std::size_t(EF_PR) * T), acc(std::size_t(EF_PR) * T);
        if (!overlap) {
            for (unsigned g = 0; g < NG; ++g) {
                const unsigned nt0 = g * EF_NTB;
                group_mma(nt0, acc); group_epi(acc, sT);
                group_dit(sT);
                for (unsigned q = 0; q < EF_PR; ++q)
                    if (nt0 + q < NT)
                        for (unsigned pos = 0; pos < T; ++pos)
                            out[(nt0 + q) * T + pos] = sT[q * T + pos];
            }
        } else {
            // group 0's MMA+epilogue before the loop (as in the kernel)
            group_mma(0, acc); group_epi(acc, sT);
            for (unsigned g = 0; g < NG; ++g) {
                const unsigned nt0 = g * EF_NTB;
                const bool has_next = (g + 1 < NG);
                // MMA chunks of g+1 are issued between DIT passes of g; the
                // epilogue that writes sT for g+1 runs after store of g.
                if (has_next) group_mma(nt0 + EF_NTB, acc);
                group_dit(sT);
                for (unsigned q = 0; q < EF_PR; ++q)
                    if (nt0 + q < NT)
                        for (unsigned pos = 0; pos < T; ++pos)
                            out[(nt0 + q) * T + pos] = sT[q * T + pos];
                if (has_next) group_epi(acc, sT);
            }
        }
    };
    run(false, out_seq);
    run(OVERLAP, out_pipe);

    unsigned bad = 0;
    for (std::size_t i = 0; i < out_seq.size(); ++i)
        if (out_seq[i] != out_pipe[i]) ++bad;
    printf("overlap_emul: NT=%u NG=%u OVERLAP=%d bad=%u/%zu\n",
           NT, NG, (int)OVERLAP, bad, out_seq.size());
    return bad != 0;
}
