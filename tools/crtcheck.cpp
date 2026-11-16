// crtcheck — the weighted-sum CRT, checked on the host.
//
// The device CRT (pgpu::k_crt_w) is about to replace a Garner chain with
//    S = sum res_j * crt_w[j]        (one big sum, then one CIOS pass)
// plus a quotient estimate that fixes the sign of the convolution. Both of
// those are exact claims about integers, not heuristics, so they can be checked
// here with polymul::Big arithmetic and no GPU, no shim, no kernels: feed the
// kernel's own inputs (residues) through the same formulas and compare against
// the definition of the answer.
//
// For each point: draw residues r_j in [0, p_j), so the exact integer
//    X = sum r_j * e_j
// is what the reconstruction sees, with the convolution C = X - round(X/P)*P.
// Then check, in order:
//    * t = (sum r_j * crt_c[j] + 2^191) >> 192 equals round(X/P);
//    * T = S + q*2^(64(NQ-1)) - t*Pq' is in [0, q*2^(64NQ)), so CIOS applies;
//    * the CIOS pass gives C mod q, bit for bit.
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../polymul.hpp"

#include "qtable.hpp"

using polymul::Big;
using polymul::u32;
using polymul::u64;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  FAIL %s\n", what.c_str());
        ++failures;
    }
}

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

u64 limb64(const Big& x, std::size_t l) {
    return static_cast<u64>(x.limb(2 * l)) | (static_cast<u64>(x.limb(2 * l + 1)) << 32);
}

std::string hex(const Big& x) {
    if (x.zero()) return "0";
    std::string s;
    for (std::size_t i = x.d.size(); i-- > 0;) {
        char buf[9];
        std::snprintf(buf, sizeof buf, "%08x", x.d[i]);
        s += buf;
    }
    return s;
}

Big from64(const std::vector<u64>& v, std::size_t n) {    Big x;
    x.d.resize(2 * n);
    for (std::size_t l = 0; l < n; ++l) {
        x.d[2 * l] = static_cast<u32>(v[l]);
        x.d[2 * l + 1] = static_cast<u32>(v[l] >> 32);
    }
    x.trim();
    return x;
}

// One Montgomery step: X <- (X + m*q)/2^64 mod q, valid for X < q*2^64.
// Mirrors what the kernel does limb by limb, in the same order: the low limb is
// zero by the choice of m, so the result is the array shifted down by one.
void mont_step(std::vector<u64>& x, const std::vector<u64>& q64, std::size_t NQ,
               u64 qneg_inv) {
    const u64 m = x[0] * qneg_inv;
    u64 carry = 0;
    for (std::size_t l = 0; l < NQ; ++l) {
        const unsigned __int128 cur = static_cast<unsigned __int128>(x[l]) +
                                     static_cast<unsigned __int128>(m) * q64[l] + carry;
        const u64 low = static_cast<u64>(cur);
        carry = static_cast<u64>(cur >> 64);
        if (l > 0) x[l - 1] = low;
        else check(low == 0, "the Montgomery low limb is zero");
    }
    x[NQ - 1] = carry;
}

Big cios2(const Big& T, const std::vector<u64>& q64, std::size_t NQ, u64 qneg_inv) {
    std::vector<u64> x(NQ);
    for (std::size_t l = 0; l < NQ; ++l) x[l] = limb64(T, l);
    mont_step(x, q64, NQ, qneg_inv);
    mont_step(x, q64, NQ, qneg_inv);
    Big R = from64(x, NQ);
    const Big qq = from64(q64, NQ);
    for (int i = 0; i < 3 && polymul::cmp(R, qq) >= 0; ++i) R = polymul::sub(R, qq);
    return R;
}

void run_point(u32 N, u32 W, u32 B, int cases, const Big& known = Big()) {
    std::printf("N=%u W=%u B=%u\n", N, W, B);
    Big q = known;
    for (const KnownPoint& kp : kPoints)
        if (kp.N == N && kp.W == W) q = from_hex(kp.hex);
    check(q.limbs() != 0, "the test knows q for this point");
    if (q.limbs() == 0) return;

    const polymul::Plan plan = polymul::build_plan(N, W, (W + 31) / 32, q.d.data());
    check(plan.crt_nq != 0, "the point got a weighted-sum table");
    if (plan.crt_nq == 0) return;
    const std::size_t NQ = plan.crt_nq, k = plan.k();
    const Big qq = from64(plan.q64, NQ);
    const Big pmod = from64(plan.pmod64, NQ);
    const Big qshift = polymul::shl_bits(qq, 68);  // q * 2^68, keeps T positive
    const Big qr0 = polymul::shl_bits(qq, 128);    // q * 2^128, the Montgomery bound

    // The idempotents are not kept (they are only used to build the tables), so
    // rebuild X the way the kernel sees it: straight from the weights is not
    // possible (they are reduced mod q), hence reconstruct from e_j again here.
    std::vector<Big> idem(k);
    Big rr;
    for (std::size_t j = 0; j < k; ++j) {
        Big mj, iq, mjm;
        polymul::divmod(plan.P, Big(plan.primes[j]), mj, rr);
        polymul::divmod(mj, Big(plan.primes[j]), iq, mjm);
        const u64 inv = polymul::inv_mod(limb64(mjm, 0), plan.primes[j]);
        idem[j] = polymul::mul(mj, Big(inv));
    }

    std::mt19937_64 rng(0x5eed ^ (static_cast<u64>(N) << 32) ^ W);
    for (int c = 0; c < cases; ++c) {
        // Residues as the RNS stage would produce them, plus the exact X.
        std::vector<u64> res(k);
        for (std::size_t j = 0; j < k; ++j) res[j] = rng() % plan.primes[j];
        Big X;
        for (std::size_t j = 0; j < k; ++j)
            X = polymul::add(X, polymul::mul(idem[j], Big(res[j])));

        // The weighted sum, exactly as the kernel accumulates it (big integer,
        // no reduction), then the same in the Montgomery domain.
        Big S;
        for (std::size_t j = 0; j < k; ++j)
            S = polymul::add(S, polymul::mul(from64(plan.crt_w[j], NQ), Big(res[j])));

        // t from the 192-bit fractional weights.
        Big num;
        for (std::size_t j = 0; j < k; ++j) {
            const Big cw = from64(plan.crt_c[j], 3);
            num = polymul::add(num, polymul::mul(cw, Big(res[j])));
        }
        Big half;
        half.d.assign(6, 0u);
        half.d[5] = 0x80000000u;  // 2^191, base 2^32
        num = polymul::add(num, half);
        Big t, rem;
        polymul::divmod(num, polymul::shl_limbs(Big(1), 6), t, rem);  // >> 192
        check(polymul::cmp(t, polymul::shl_limbs(Big(1), 3)) < 0, "t < 2^68");

        // The reference: C = X - t*P, which is signed, reduced mod q.
        Big XtP, cq, crem;

        // Two intermediate identities, so a failure says which step is wrong.
        {
            const Big r0b = polymul::shl_bits(Big(1), 128);
            Big a1, a2;
            polymul::divmod(S, qq, cq, a1);
            polymul::divmod(polymul::mul(X, r0b), qq, cq, a2);
            if (polymul::cmp(a1, a2) != 0) {
                std::printf("  FAIL case %d: S != X * 2^(64NQ) (mod q)\n", c);
                ++failures;
                return;
            }
            Big wt;
            polymul::divmod(polymul::add(X, polymul::halve(plan.P)), plan.P, wt, a1);
            if (polymul::cmp(wt, t) != 0) {
                std::printf("  FAIL case %d: t != round(X/P)\n", c);
                ++failures;
                return;
            }
        }
        if (polymul::cmp(X, polymul::mul(t, plan.P)) >= 0) {
            XtP = polymul::sub(X, polymul::mul(t, plan.P));
            polymul::divmod(XtP, qq, cq, crem);
        } else {
            XtP = polymul::sub(polymul::mul(t, plan.P), X);  // -(C)
            polymul::divmod(XtP, qq, cq, crem);
            if (!crem.zero()) crem = polymul::sub(qq, crem);
        }

        // And the kernel's route: T = S + q*2^68 - t*Pq', then two Montgomery steps.
        const Big sub = polymul::mul(t, pmod);
        check(polymul::cmp(polymul::add(S, qshift), sub) >= 0, "T stays non-negative");
        Big T = polymul::sub(polymul::add(S, qshift), sub);
        check(polymul::cmp(T, qr0) < 0, "T < q*2^128");
        const Big got = cios2(T, plan.q64, NQ, plan.qneg_inv);
        if (polymul::cmp(got, crem) != 0) {
            std::printf("  FAIL case %d: the weighted CRT disagrees\n", c);
            std::printf("    t=%s\n", hex(t).c_str());
            std::printf("    want=%s\n", hex(crem).c_str());
            std::printf("    got =%s\n", hex(got).c_str());
            ++failures;
            return;
        }
    }
    std::printf("  ok, k=%zu NQ=%zu, %d random residue sets\n", k, NQ, cases);
}

// A q read from a point directory, for the ranked point whose 1747-bit q is too
// large to carry around as a literal here.
Big read_q(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return Big();
    std::vector<u32> d;
    u32 w;
    while (std::fread(&w, sizeof w, 1, f) == 1) d.push_back(w);
    std::fclose(f);
    return Big::from_le(d.data(), d.size());
}

}  // namespace

int main(int argc, char** argv) {
    run_point(64, 32, 1, 40);
    run_point(128, 64, 1, 40);
    run_point(64, 128, 1, 40);
    run_point(64, 254, 1, 40);
    run_point(32, 868, 1, 20);
    run_point(1024, 1747, 1, 6);
    run_point(65536, 64, 1, 3);
    if (argc > 3) {
        // The ranked point: q comes from the point, not from qtable.hpp.
        const u32 N = static_cast<u32>(std::atoi(argv[2]));
        const u32 W = static_cast<u32>(std::atoi(argv[3]));
        const Big q = read_q(argv[1]);
        check(q.limbs() != 0 && q.bits() == W, "q was read and is W bits");
        if (q.limbs()) run_point(N, W, 1, 3, q);
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("weighted-sum CRT: exact at every point\n");
    return 0;
}
