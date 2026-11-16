// Checks the device kernels in polymul_gpu.cuh against the host path in
// polymul.hpp, on a machine with no GPU.
//
//   g++ -O2 -std=c++17 -o build-cpu/gpucheck tools/gpucheck.cpp
//   ./build-cpu/gpucheck
//
// The shim runs one thread per block, so this exercises every arithmetic step —
// Montgomery products, the 64x64->128 carry chain in big_mul_add, the twiddles,
// Garner, Barrett, the negative branch — and compares the finished coefficient
// vectors bit for bit. What it does not exercise is the block and thread
// indexing; that is checked at fherma_init on the device itself, untimed.
#include "shim/cuda_shim.hpp"

#include "../polymul_gpu.cuh"
#include "../polymul.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using polymul::Big;
using polymul::divmod;
using polymul::u32;
using polymul::u64;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  FAIL %s\n", what.c_str());
        ++failures;
    }
}

// Tables flattened out of Plan into the arrays Params points at.
struct Flat {
    std::vector<u64> primes, pinv, r2;
    std::vector<u64> root_fwd, root_inv, twist_fwd, twist_inv, limb_w, inv_p;
    std::vector<u32> P_half, P_mod_q, qlimbs, mu;
    std::vector<u64> crt_w, crt_c, q64, pmod64, qsh64;
    pgpu::Params par{};
};

Flat flatten(const polymul::Plan& plan) {
    Flat f;
    const std::size_t k = plan.k(), N = plan.N, L = plan.L;

    for (std::size_t m = 0; m < k; ++m) {
        f.primes.push_back(plan.primes[m]);
        f.pinv.push_back(plan.mods[m].pinv);
        f.r2.push_back(plan.mods[m].r2);
        f.root_fwd.insert(f.root_fwd.end(), plan.root_fwd[m].begin(), plan.root_fwd[m].end());
        f.root_inv.insert(f.root_inv.end(), plan.root_inv[m].begin(), plan.root_inv[m].end());
        f.twist_fwd.insert(f.twist_fwd.end(), plan.twist_fwd[m].begin(), plan.twist_fwd[m].end());
        f.twist_inv.insert(f.twist_inv.end(), plan.twist_inv[m].begin(), plan.twist_inv[m].end());
        f.limb_w.insert(f.limb_w.end(), plan.limb_w[m].begin(), plan.limb_w[m].end());
    }

    f.inv_p.assign(k * k, 0);
    for (std::size_t m = 1; m < k; ++m)
        for (std::size_t j = 0; j < m; ++j) f.inv_p[j * k + m] = plan.inv_p[j][m];

    f.P_half.resize(plan.P_half.limbs());
    plan.P_half.to_le(f.P_half.data(), f.P_half.size());
    f.P_mod_q.resize(plan.P_mod_q.limbs());
    plan.P_mod_q.to_le(f.P_mod_q.data(), f.P_mod_q.size());
    f.qlimbs.resize(L);
    plan.q.to_le(f.qlimbs.data(), L);
    f.mu.resize(plan.mu.limbs());
    plan.mu.to_le(f.mu.data(), f.mu.size());

    f.par.N = static_cast<u32>(N);
    f.par.L = static_cast<u32>(L);
    f.par.k = static_cast<u32>(k);
    f.par.primes = f.primes.data();
    f.par.pinv = f.pinv.data();
    f.par.r2 = f.r2.data();
    f.par.root_fwd = f.root_fwd.data();
    f.par.root_inv = f.root_inv.data();
    f.par.twist_fwd = f.twist_fwd.data();
    f.par.twist_inv = f.twist_inv.data();
    f.par.limb_w = f.limb_w.data();
    f.par.inv_p = f.inv_p.data();
    f.par.P_half = f.P_half.data();
    f.par.P_mod_q = f.P_mod_q.data();
    f.par.qlimbs = f.qlimbs.data();
    f.par.mu = f.mu.data();
    f.par.np = static_cast<u32>(f.P_half.size());
    f.par.nq = static_cast<u32>(f.P_mod_q.size());
    f.par.nmu = static_cast<u32>(f.mu.size());
    f.par.m_limbs = static_cast<u32>(plan.m_limbs);

    f.par.crt_nq = plan.crt_nq;
    if (plan.crt_nq) {
        for (std::size_t j = 0; j < k; ++j)
            f.crt_w.insert(f.crt_w.end(), plan.crt_w[j].begin(), plan.crt_w[j].end());
        for (std::size_t j = 0; j < k; ++j)
            f.crt_c.insert(f.crt_c.end(), plan.crt_c[j].begin(), plan.crt_c[j].end());
        f.q64 = plan.q64;
        f.pmod64 = plan.pmod64;
        f.qsh64 = plan.qsh64;
        f.par.crt_w = f.crt_w.data();
        f.par.crt_c = f.crt_c.data();
        f.par.q64 = f.q64.data();
        f.par.pmod64 = f.pmod64.data();
        f.par.qsh64 = f.qsh64.data();
        f.par.qneg_inv = plan.qneg_inv;
    }
    return f;
}

// A value in [0, q), so the inputs are the ones the specification promises.
Big random_below(std::mt19937_64& rng, const Big& q) {
    Big x;
    x.d.resize(q.limbs() + 1);
    for (auto& v : x.d) v = static_cast<u32>(rng());
    x.trim();
    Big quot, rem;
    divmod(x, q, quot, rem);
    return rem;
}

// q is the largest prime below 2^W that is 1 mod 2N. Deriving it here would need
// a primality proof on a 868-bit integer, which is the bundle's job, so the test
// carries the answers tools/qtable.py derives into tools/qtable.hpp.
#include "qtable.hpp"

Big from_hex(const char* hex) {
    Big x;
    const std::size_t digits = std::strlen(hex);
    const std::size_t limbs = (digits + 7) / 8;
    x.d.assign(limbs, 0u);
    for (std::size_t i = 0; i < digits; ++i) {
        const char c = hex[digits - 1 - i];
        const u32 nibble = (c >= '0' && c <= '9') ? static_cast<u32>(c - '0')
                                                  : static_cast<u32>((c | 32) - 'a' + 10);
        x.d[i / 8] |= nibble << (4 * (i % 8));
    }
    x.trim();
    return x;
}

// The same stage order and the same launches as gpu_multiply in solve.cu, one
// block at a time: every block index the real grid would create is visited.
void device_multiply(const Flat& f, const u32* a, const u32* b, u64* resa, u64* resb,
                     u32* out, u32 B) {
    const u32 N = f.par.N, k = f.par.k;

    threadIdx.x = 0;
    threadIdx.y = 0;
    threadIdx.z = 0;
    blockDim.x = 1;
    blockDim.y = 1;
    blockDim.z = 1;
    blockIdx.y = 0;
    blockIdx.z = 0;

    const u32 tiles = (N + pgpu::TILE - 1) / pgpu::TILE;
    for (u32 j = 0; j < B; ++j)
        for (u32 t = 0; t < tiles; ++t) {
            blockIdx.x = t;
            blockIdx.y = j;
            pgpu::k_residues(f.par, a, resa);
            pgpu::k_residues(f.par, b, resb);
        }

    const bool two_pass =
        N >= pgpu::STEP && N % pgpu::STEP == 0 && N / pgpu::STEP <= pgpu::MAXROWS;
    if (two_pass) {
        const u32 rows = N / pgpu::STEP;
        const u32 cols = pgpu::STEP / pgpu::LINES;
        const u32 rtiles = (rows + pgpu::LINES - 1) / pgpu::LINES;
        for (u32 j = 0; j < B; ++j)
            for (u32 m = 0; m < k; ++m)
                for (u32 t = 0; t < cols; ++t) {
                    blockIdx.x = t;
                    blockIdx.y = m;
                    blockIdx.z = j;
                    pgpu::k_cols_fwd(f.par, resa, resb);
                }
        for (u32 j = 0; j < B; ++j)
            for (u32 m = 0; m < k; ++m)
                for (u32 t = 0; t < rtiles; ++t) {
                    blockIdx.x = t;
                    blockIdx.y = m;
                    blockIdx.z = j;
                    pgpu::k_rows_fwd_dot(f.par, resa, resb);
                }
        for (u32 j = 0; j < B; ++j)
            for (u32 m = 0; m < k; ++m)
                for (u32 t = 0; t < rtiles; ++t) {
                    blockIdx.x = t;
                    blockIdx.y = m;
                    blockIdx.z = j;
                    pgpu::k_rows_inv(f.par, resa);
                }
        for (u32 j = 0; j < B; ++j)
            for (u32 m = 0; m < k; ++m)
                for (u32 t = 0; t < cols; ++t) {
                    blockIdx.x = t;
                    blockIdx.y = m;
                    blockIdx.z = j;
                    pgpu::k_cols_inv_twist(f.par, resa);
                }
    } else {
        for (u32 j = 0; j < B; ++j)
            for (u32 m = 0; m < k; ++m) {
                blockIdx.x = j * k + m;
                pgpu::k_ntt(f.par, resa, resb);
            }
    }

    for (u32 j = 0; j < B; ++j)
        for (u32 i = 0; i < N; ++i) {
            blockIdx.x = i;
            blockIdx.y = j;
            // The kernel owns the pair offset: pass the whole buffer, exactly as
            // solve.cu does. Handing it a per-pair pointer here would hide the
            // very bug this checks for.
            u32* dst = out;
            switch (f.par.crt_nq) {
                case 4: pgpu::k_crt_w<4>(f.par, resa, dst); break;
                case 6: pgpu::k_crt_w<6>(f.par, resa, dst); break;
                case 8: pgpu::k_crt_w<8>(f.par, resa, dst); break;
                case 12: pgpu::k_crt_w<12>(f.par, resa, dst); break;
                case 16: pgpu::k_crt_w<16>(f.par, resa, dst); break;
                case 24: pgpu::k_crt_w<24>(f.par, resa, dst); break;
                case 32: pgpu::k_crt_w<32>(f.par, resa, dst); break;
                case 48: pgpu::k_crt_w<48>(f.par, resa, dst); break;
                case 64: pgpu::k_crt_w<64>(f.par, resa, dst); break;
                default: pgpu::k_crt(f.par, resa, dst); break;  // no table: Garner
            }
        }
}

void run_point(u32 N, u32 W, u32 B) {
    std::printf("N=%u W=%u B=%u\n", N, W, B);

    std::mt19937_64 rng(0x9e3779b97f4a7c15ull ^ (static_cast<u64>(N) << 32) ^ W);
    Big q;
    for (const KnownPoint& kp : kPoints)
        if (kp.N == N && kp.W == W) q = from_hex(kp.hex);
    check(q.limbs() != 0, "the test knows q for this point");
    if (q.limbs() == 0) return;
    check(q.bits() == W || q.bits() == W - 1, "q is about W bits");

    polymul::Plan plan = polymul::build_plan(N, W, (W + 31) / 32, q.d.data());
    check(plan.k() <= pgpu::MAXK, "k fits MAXK");
    check(plan.L <= pgpu::MAXL, "L fits MAXL");
    check(plan.P.limbs() <= pgpu::MAXLIMB, "P fits MAXLIMB");
    check(plan.mu.limbs() + plan.P.limbs() <= 2 * pgpu::MAXLIMB, "Barrett product fits");
    if (failures) return;

    Flat f = flatten(plan);
    f.par.B = B;

    const std::size_t per = static_cast<std::size_t>(N) * plan.L;
    std::vector<u32> a(B * per), b(B * per), got(B * per, 0), want(B * per, 0);
    for (std::size_t i = 0; i < B * N; ++i) {
        Big x = random_below(rng, q), y = random_below(rng, q);
        x.to_le(a.data() + i * plan.L, plan.L);
        y.to_le(b.data() + i * plan.L, plan.L);
    }

    polymul::Engine engine(plan, 1);
    for (u32 j = 0; j < B; ++j)
        engine.multiply(a.data() + j * per, b.data() + j * per, want.data() + j * per);

    // Device side, one thread per block.
    std::vector<u64> resa(static_cast<std::size_t>(plan.k()) * B * N);
    std::vector<u64> resb(resa.size());

    device_multiply(f, a.data(), b.data(), resa.data(), resb.data(), got.data(), B);

    std::size_t bad = 0;
    for (std::size_t i = 0; i < B * per; ++i)
        if (got[i] != want[i]) {
            if (bad < 3)
                std::printf("  coeff %zu limb %zu: got %08x want %08x\n", i / per, i % per,
                            got[i], want[i]);
            ++bad;
        }
    check(bad == 0, "device coefficients equal host coefficients");
    if (!bad) std::printf("  ok, %u pair(s), k=%zu, P=%zu bits\n", B, plan.k(), plan.P.bits());
}

}  // namespace

int main() {
    // Montgomery against a plain modular product first: everything else is
    // built on it.
    {
        std::mt19937_64 rng(12345);
        const u64 p = 4611686018427387847ull;  // a 62-bit prime
        const polymul::Mod md = polymul::Mod::make(p);
        u64 bad = 0;
        for (int i = 0; i < 200000; ++i) {
            const u64 a = rng() % p, b = rng() % p;
            const u64 got = pgpu::mont_mul(md.to_mont(a), md.to_mont(b), md.pinv, md.p);
            const u64 want = md.to_mont(polymul::mulmod(a, b, p));
            if (got != want) ++bad;
        }
        check(bad == 0, "mont_mul matches the host Montgomery product");

        bad = 0;
        for (int i = 0; i < 200000; ++i) {
            const u64 a = rng() % p, b = rng() % p;
            const u64 got = pgpu::raw_mul(a, b, p, md.pinv, md.r2);
            if (got != polymul::mulmod(a, b, p)) ++bad;
        }
        check(bad == 0, "raw_mul matches a plain modular product");
    }

    run_point(64, 32, 2);
    run_point(128, 64, 1);
    run_point(64, 128, 1);
    run_point(64, 254, 1);
    run_point(32, 868, 1);
    // The ranked point's arithmetic without its length: k=57, L=55, P=3534 bits.
    run_point(1024, 1747, 1);
    // ... and with two pairs, so the pair stride is checked in the two-pass path
    // and in the CRT write, not only in the stage-at-a-time kernels.
    run_point(1024, 1747, 2);
    // The ranked point's length with a narrow q, so all sixteen stages and the
    // column/row split run at N=65536 without paying k=57 for it.
    run_point(65536, 64, 1);
    run_point(65536, 64, 2);

    if (failures) {
        std::printf("\n%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("\nall device checks exact\n");
    return 0;
}
