// Emulator of the OC24 input-A load in tc.cuh::encode_fwdlow.
//
// It replays the exact thread -> (row, limb) mapping of the kernel
// (warp/lane -> g, t, rowbase; i0/i1 at bit-reversed rows; w0/w1 = ks*8+t,
// w0+4) and compares two paths:
//   ALOAD=0 - the original eager read (56 conditional LDGs per thread);
//   ALOAD=1 - cp.async chunked read into a 2-buffer shared staging area,
//             clamped source pointer + predicate restored when the chunk is
//             copied into a[2][ENC_KS_MAX][4].
// Both must produce bit-identical fragments.  The rest of the NTT pipeline is
// unchanged, so this is the only thing the flag can affect.
//
// Build: g++ -O2 -std=c++17 -o aload_emul aload_emul.cpp
// Run:   ./aload_emul [KS] [L] [ALOAD]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
typedef uint32_t u32;
constexpr unsigned ENC_KS_MAX = 7, EF_ROWS = 512, WARPS = 16, LANES = 32;

int main(int argc, char** argv) {
    const unsigned KS = argc > 1 ? (unsigned)atoi(argv[1]) : 7;
    const unsigned L  = argc > 2 ? (unsigned)atoi(argv[2]) : 55;
    const bool ALOAD = !(argc > 3 && atoi(argv[3]) == 0);
    const unsigned logN = 16, N = 1u << logN;   // one tile starts at 0
    // input rows: pair 0 of operand A, N rows x L limbs
    std::vector<u32> src(std::size_t(N) * L);
    for (auto& x : src) x = 0x9e3779b9u * (u32)(&x - src.data() + 1);

    auto rev = [&](u32 x) { u32 r = 0; for (unsigned i = 0; i < logN; ++i) { r = (r << 1) | (x & 1); x >>= 1; } return r; };

    unsigned bad = 0;
    // emulate one CTA (512 threads), tile start 0, pair base 0
    for (unsigned tid = 0; tid < EF_ROWS; ++tid) {
        const unsigned warp = tid >> 5, lane = tid & 31;
        const unsigned g = lane >> 2, t = lane & 3;
        const unsigned rowbase = warp * 32;
        u32 eager[2][ENC_KS_MAX][4] = {}, got[2][ENC_KS_MAX][4] = {};

        // ---- path 0: original eager reads ----
        for (unsigned mt = 0; mt < 2; ++mt) {
            const unsigned i0 = rowbase + mt * 16 + g, i1 = i0 + 8;
            const u32* r0 = src.data() + std::size_t(rev(i0)) * L;
            const u32* r1 = src.data() + std::size_t(rev(i1)) * L;
            for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
                const unsigned w0 = ks * 8 + t, w1 = w0 + 4;
                const bool on = ks < KS;
                eager[mt][ks][0] = (on && w0 < L) ? r0[w0] : 0u;
                eager[mt][ks][1] = (on && w0 < L) ? r1[w0] : 0u;
                eager[mt][ks][2] = (on && w1 < L) ? r0[w1] : 0u;
                eager[mt][ks][3] = (on && w1 < L) ? r1[w1] : 0u;
            }
        }

        // ---- path 1: cp.async chunked reads, clamped pointer + restored pred ----
        // sA[buf][mt][tid][4]; one (mt,ks) chunk per buffer, 8 words/thread.
        u32 sA[2][2][EF_ROWS][4] = {};
        if (!ALOAD) {
            // flag 0: exact original path is the "got" side
            for (unsigned mt = 0; mt < 2; ++mt)
                for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks)
                    for (unsigned j = 0; j < 4; ++j) got[mt][ks][j] = eager[mt][ks][j];
        } else
        for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks) {
            const unsigned buf = ks & 1;
            for (unsigned mt = 0; mt < 2; ++mt) {
                const unsigned i0 = rowbase + mt * 16 + g, i1 = i0 + 8;
                const u32* r0 = src.data() + std::size_t(rev(i0)) * L;
                const u32* r1 = src.data() + std::size_t(rev(i1)) * L;
                const unsigned w0 = ks * 8 + t, w1 = w0 + 4;
                const bool on0 = ks < KS && w0 < L, on1 = ks < KS && w1 < L;
                // cp.async4 with clamped source (src[0] when inactive)
                sA[buf][mt][tid][0] = on0 ? r0[w0] : src[0];
                sA[buf][mt][tid][1] = on0 ? r1[w0] : src[0];
                sA[buf][mt][tid][2] = on1 ? r0[w1] : src[0];
                sA[buf][mt][tid][3] = on1 ? r1[w1] : src[0];
                // astore_chunk: restore the predicate on copy
                got[mt][ks][0] = on0 ? sA[buf][mt][tid][0] : 0u;
                got[mt][ks][1] = on0 ? sA[buf][mt][tid][1] : 0u;
                got[mt][ks][2] = on1 ? sA[buf][mt][tid][2] : 0u;
                got[mt][ks][3] = on1 ? sA[buf][mt][tid][3] : 0u;
            }
        }

        for (unsigned mt = 0; mt < 2; ++mt)
            for (unsigned ks = 0; ks < ENC_KS_MAX; ++ks)
                for (unsigned j = 0; j < 4; ++j)
                    if (eager[mt][ks][j] != got[mt][ks][j]) ++bad;
    }
    printf("aload_emul: KS=%u L=%u ALOAD=%d bad=%u\n", KS, L, (int)ALOAD, bad);
    return bad != 0;
}
