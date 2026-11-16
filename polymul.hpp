// polymul.hpp — negacyclic polynomial multiplication over Z_q[X]/(X^N + 1)
// for a *wide* prime q (up to ~2^1747), exact and self-contained.
//
// Portable core: no CUDA, no dependencies, compiles with g++/clang/nvcc. It is
// the correct reference the GPU path is checked against, and it is the same
// algorithm the GPU path implements:
//
//   1. decompose every coefficient into residues modulo k NTT primes p_m,
//   2. negacyclic NTT under every p_m, pointwise multiply, inverse NTT,
//   3. Garner/CRT the exact integer convolution back (it is < 2N q^2),
//   4. Barrett-reduce it modulo q and store the L u32 limbs.
//
// Every p_m is < 2^62 and == 1 (mod 2N), so a 2N-th root of unity exists and a
// length-N NTT is defined over it. The k primes are chosen so that
// P = prod p_m exceeds the largest possible |convolution coefficient|,
// 2N(q-1)^2, which is what makes the reconstruction exact.
//
// Everything `build_plan` computes depends only on (N, W, q), i.e. it is the
// content of `fherma_init` — which the harness does not time.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

#if defined(_MSC_VER) && !defined(__clang__)
#error "polymul.hpp needs unsigned __int128: build with gcc, clang or nvcc"
#endif

namespace polymul {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

// ---------------------------------------------------------------- big integer
// Little-endian, base 2^32. Only what the CRT/Barrett path needs.
struct Big {
    std::vector<u32> d;  // empty == 0

    Big() = default;
    explicit Big(u64 v) {
        if (v) {
            d.push_back(static_cast<u32>(v));
            if (v >> 32) d.push_back(static_cast<u32>(v >> 32));
        }
    }

    void trim() { while (!d.empty() && d.back() == 0) d.pop_back(); }
    bool zero() const { return d.empty(); }
    std::size_t limbs() const { return d.size(); }
    u32 limb(std::size_t i) const { return i < d.size() ? d[i] : 0u; }

    std::size_t bits() const {
        if (d.empty()) return 0;
        std::size_t b = 32 * (d.size() - 1);
        for (u32 top = d.back(); top; top >>= 1) ++b;
        return b;
    }

    static Big from_le(const u32* p, std::size_t n) {
        Big x;
        x.d.assign(p, p + n);
        x.trim();
        return x;
    }
    void to_le(u32* out, std::size_t n) const {
        for (std::size_t i = 0; i < n; ++i) out[i] = limb(i);
    }
};

inline int cmp(const Big& a, const Big& b) {
    if (a.d.size() != b.d.size()) return a.d.size() < b.d.size() ? -1 : 1;
    for (std::size_t i = a.d.size(); i-- > 0;)
        if (a.d[i] != b.d[i]) return a.d[i] < b.d[i] ? -1 : 1;
    return 0;
}
inline bool operator<(const Big& a, const Big& b) { return cmp(a, b) < 0; }
inline bool operator>=(const Big& a, const Big& b) { return cmp(a, b) >= 0; }

inline Big add(const Big& a, const Big& b) {
    Big r;
    const std::size_t n = std::max(a.d.size(), b.d.size());
    r.d.resize(n);
    u64 carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
        u64 cur = static_cast<u64>(a.limb(i)) + b.limb(i) + carry;
        r.d[i] = static_cast<u32>(cur);
        carry = cur >> 32;
    }
    if (carry) r.d.push_back(static_cast<u32>(carry));
    return r;
}

// a - b, requires a >= b
inline Big sub(const Big& a, const Big& b) {
    Big r;
    r.d.resize(a.d.size());
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < a.d.size(); ++i) {
        std::int64_t cur = static_cast<std::int64_t>(a.d[i]) - b.limb(i) - borrow;
        borrow = cur < 0 ? 1 : 0;
        if (cur < 0) cur += std::int64_t(1) << 32;
        r.d[i] = static_cast<u32>(cur);
    }
    r.trim();
    return r;
}

inline Big mul(const Big& a, const Big& b) {
    if (a.zero() || b.zero()) return Big();
    Big r;
    r.d.assign(a.d.size() + b.d.size(), 0u);
    for (std::size_t i = 0; i < a.d.size(); ++i) {
        u64 carry = 0;
        const u64 ai = a.d[i];
        for (std::size_t j = 0; j < b.d.size(); ++j) {
            u64 cur = ai * b.d[j] + r.d[i + j] + carry;  // < 2^64 in base 2^32
            r.d[i + j] = static_cast<u32>(cur);
            carry = cur >> 32;
        }
        std::size_t k = i + b.d.size();
        while (carry) {
            u64 cur = static_cast<u64>(r.d[k]) + carry;
            r.d[k] = static_cast<u32>(cur);
            carry = cur >> 32;
            ++k;
        }
    }
    r.trim();
    return r;
}

// a * m + add, in place. m is a full 62-bit value, so the accumulator is 128-bit.
inline void mul_u64_add(Big& x, u64 m, u64 addend) {
    u128 carry = addend;
    for (std::size_t i = 0; i < x.d.size(); ++i) {
        u128 cur = static_cast<u128>(x.d[i]) * m + carry;
        x.d[i] = static_cast<u32>(cur);
        carry = cur >> 32;
    }
    while (carry) {
        x.d.push_back(static_cast<u32>(carry));
        carry >>= 32;
    }
    x.trim();
}

inline Big mul_scalar(const Big& a, u32 m) {
    if (m == 0 || a.zero()) return Big();
    Big r = a;
    mul_u64_add(r, m, 0);
    return r;
}

inline Big shl_limbs(const Big& a, std::size_t n) {
    if (a.zero()) return Big();
    Big r;
    r.d.assign(n, 0u);
    r.d.insert(r.d.end(), a.d.begin(), a.d.end());
    return r;
}

inline Big shl_bits(const Big& a, unsigned bits) {
    if (a.zero()) return Big();
    const std::size_t limbs = bits / 32;
    const unsigned sh = bits % 32;
    Big r;
    r.d.assign(limbs, 0u);
    u64 carry = 0;
    for (std::size_t i = 0; i < a.d.size(); ++i) {
        const u64 cur = (static_cast<u64>(a.d[i]) << sh) | carry;
        r.d.push_back(static_cast<u32>(cur));
        carry = sh ? (cur >> 32) : 0;
    }
    if (carry) r.d.push_back(static_cast<u32>(carry));
    r.trim();
    return r;
}

inline Big shr_limbs(const Big& a, std::size_t n) {
    if (n >= a.d.size()) return Big();
    Big r;
    r.d.assign(a.d.begin() + static_cast<std::ptrdiff_t>(n), a.d.end());
    r.trim();
    return r;
}

inline Big halve(const Big& a) {  // floor(a/2)
    Big r = a;
    if (r.d.empty()) return r;
    u32 carry = 0;
    for (std::size_t i = r.d.size(); i-- > 0;) {
        u32 cur = r.d[i];
        r.d[i] = (cur >> 1) | (carry << 31);
        carry = cur & 1u;
    }
    r.trim();
    return r;
}

// Long division, one quotient limb at a time, its value found by binary search.
// Setup-only code (a few 1e5 limb operations for the real point): clarity wins.
inline void divmod(const Big& u, const Big& v, Big& quot, Big& rem) {
    if (v.zero()) throw std::runtime_error("polymul: division by zero");
    quot = Big();
    rem = Big();
    if (cmp(u, v) < 0) { rem = u; return; }

    const std::size_t nu = u.limbs();
    quot.d.assign(nu, 0u);
    for (std::size_t i = nu; i-- > 0;) {
        rem.d.insert(rem.d.begin(), u.d[i]);  // rem = rem*B + u[i]
        rem.trim();
        if (cmp(rem, v) < 0) continue;

        u32 lo = 1, hi = 0xFFFFFFFFu, best = 1;
        while (true) {
            const u32 mid = lo + (hi - lo) / 2;
            const Big p = mul_scalar(v, mid);
            if (cmp(p, rem) <= 0) {
                best = mid;
                if (mid == 0xFFFFFFFFu) break;
                lo = mid + 1;
            } else {
                if (mid == 0) break;
                hi = mid - 1;
            }
            if (lo > hi) break;
        }
        quot.d[i] = best;
        rem = sub(rem, mul_scalar(v, best));
    }
    quot.trim();
    rem.trim();
}

// ------------------------------------------------------------------ modular
inline u64 mulmod(u64 a, u64 b, u64 p) { return static_cast<u64>(static_cast<u128>(a) * b % p); }

inline u64 powmod(u64 b, u64 e, u64 p) {
    u64 r = 1 % p;
    b %= p;
    while (e) {
        if (e & 1) r = mulmod(r, b, p);
        b = mulmod(b, b, p);
        e >>= 1;
    }
    return r;
}

inline u64 inv_mod(u64 a, u64 p) {  // p prime, a != 0 (mod p)
    __int128 t = 0, newt = 1;
    __int128 r = p, newr = a;
    while (newr) {
        const __int128 q = r / newr;
        __int128 tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (t < 0) t += p;
    return static_cast<u64>(t);
}

inline bool is_prime(u64 n) {
    if (n < 2) return false;
    for (u64 p : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull})
        if (n % p == 0) return n == p;

    u64 d = n - 1;
    int s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (u64 a : {2ull, 3ull, 5ull, 7ull, 11ull, 13ull, 17ull, 19ull, 23ull, 29ull, 31ull, 37ull}) {
        u64 x = powmod(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool composite = true;
        for (int i = 1; i < s; ++i) {
            x = mulmod(x, x, n);
            if (x == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

// Montgomery arithmetic for p < 2^62, R = 2^64.
struct Mod {
    u64 p = 0;
    u64 pinv = 0;  // -p^{-1} mod 2^64
    u64 r2 = 0;    // R^2 mod p

    static Mod make(u64 prime) {
        Mod m;
        m.p = prime;
        u64 inv = 1;
        for (int i = 0; i < 6; ++i) inv *= 2 - prime * inv;  // p^{-1} mod 2^64
        m.pinv = ~inv + 1;                                    // -p^{-1}
        const u64 rmod = static_cast<u64>((static_cast<u128>(1) << 64) % prime);
        m.r2 = mulmod(rmod, rmod, prime);
        return m;
    }

    u64 add(u64 a, u64 b) const { const u64 s = a + b; return s >= p ? s - p : s; }
    u64 sub(u64 a, u64 b) const { return a >= b ? a - b : a + p - b; }

    u64 mul(u64 a, u64 b) const {  // a*b*R^{-1} mod p
        const u128 t = static_cast<u128>(a) * b;
        const u64 m = static_cast<u64>(t) * pinv;
        const u128 u = t + static_cast<u128>(m) * p;
        const u64 r = static_cast<u64>(u >> 64);
        return r >= p ? r - p : r;
    }

    u64 to_mont(u64 a) const { return mul(a % p, r2); }
    u64 from_mont(u64 a) const { return mul(a, 1); }
};

// --------------------------------------------------------------------- Plan
struct Plan {
    u32 N = 0, W = 0, L = 0;
    Big q;
    std::vector<u64> primes;
    std::vector<Mod> mods;

    // per prime, Montgomery form. root_*[len + j] is the twiddle of the j-th
    // butterfly of the stage whose half-length is len.
    std::vector<std::vector<u64>> root_fwd, root_inv;
    std::vector<std::vector<u64>> twist_fwd;  // psi^i * R        (Montgomery)
    std::vector<std::vector<u64>> twist_inv;  // N^-1 * psi^-i    (plain)
    std::vector<std::vector<u64>> limb_w;     // (2^(32l) mod p) * R^2

    // Garner table: inv_p[j][m] = (p_j mod p_m)^{-1} mod p_m, for j < m
    std::vector<std::vector<u64>> inv_p;

    Big P, P_half, P_mod_q, mu;
    std::size_t m_limbs = 0;

    // ------------------------------------------------ weighted-sum CRT tables
    // The Garner chain above costs k^2/2 multiply-adds per coefficient, and the
    // big integers it walks are 112 limbs wide. The same reconstruction written
    // as one weighted sum costs k*n64 multiply-adds over 28-limb numbers, so
    // this is the difference between ~20k and ~2k per coefficient.
    //
    //   S = sum_j res_j * crt_w[j],   crt_w[j] = e_j * 2^128 mod q
    //
    // where e_j is the j-th CRT idempotent, (P/p_j)*(P/p_j)^{-1} mod p_j. Then S
    // is congruent to X*2^128, so two single-limb Montgomery steps (R0 = 2^128,
    // a compile-time shift) end the reduction: no Barrett divide, no 112-limb
    // product, no quotient estimate.
    //
    // S is the exact integer sum_j res_j*e_j, which is congruent to the
    // convolution but not equal to it: it lies in [0, 2^68 q), off by a multiple
    // of P. That multiple is t = round(X/P), and crt_c carries the fractional
    // part of each term with 192 bits of headroom:
    //
    //   crt_c[j] = floor(e_j * 2^192 / P),  t = round(sum res_j*crt_c[j] / 2^192)
    //
    // The estimate's error is at most sum res_j < 2^68 out of 2^192, i.e. below
    // 2^-124, while the true fractional part is at least (P - bound)/2P away
    // from 1/2 at every point, so the rounding is exact and not a heuristic.
    u32 crt_nq = 0;                       // 64-bit limbs of the sum; 0 = no table
    std::vector<std::vector<u64>> crt_w;  // k rows, 64 limbs each, first crt_nq live
    std::vector<std::vector<u64>> crt_c;  // k rows of 3 limbs (192 bits)
    std::vector<u64> q64;                 // 64 limbs: q, zero padded
    std::vector<u64> pmod64;              // 64 limbs: (P mod q) * 2^128 mod q
    std::vector<u64> qsh64;               // 64 limbs: q * 2^68, the offset that
                                          // keeps T positive (t < 2^68)
    u64 qneg_inv = 0;                     // -q^{-1} mod 2^64, the CIOS multiplier

    std::size_t k() const { return primes.size(); }
};

inline void build_tables(Plan& pl, std::size_t m);

// Sizes the device kernel is compiled for. Picking the smallest one that holds
// n64 + 2 limbs keeps the register array as small as the point allows; a point
// outside the list (q wider than 62 limbs) simply has no table and the device
// path keeps the Garner kernel.
inline u32 crt_pick_nq(std::size_t n64) {
    for (u32 nq : {4u, 6u, 8u, 12u, 16u, 24u, 32u, 48u, 64u})
        if (nq >= n64 + 2) return nq;
    return 0;
}

// The weighted-sum tables described in Plan. Setup only (a second or so of
// long division for the real point), and never timed.
inline void build_crt_tables(Plan& pl) {
    const std::size_t k = pl.k(), NQ = pl.crt_nq;
    const Big& q = pl.q;
    Big qq, rr;

    // The CRT idempotents: e_j = (P/p_j) * (P/p_j)^{-1} mod p_j, with e_j < P
    // and e_j == 1 (mod p_j), 0 (mod p_m) for m != j. Then X = sum res_j e_j is
    // the unique value in [0, kP) congruent to the convolution modulo P.
    std::vector<Big> idem(k);
    for (std::size_t j = 0; j < k; ++j) {
        Big mj;
        divmod(pl.P, Big(pl.primes[j]), mj, rr);
        Big inv_q, mj_mod;
        divmod(mj, Big(pl.primes[j]), inv_q, mj_mod);
        const u64 inv = inv_mod(static_cast<u64>(mj_mod.limb(0)) |
                                    (static_cast<u64>(mj_mod.limb(1)) << 32),
                                pl.primes[j]);
        idem[j] = mul(mj, Big(inv));
    }

    // 2^128: two Montgomery steps of R = 2^64, both compile-time shifts.
    const Big r0 = shl_limbs(Big(1), 4);

    // Base 2^32 on the host, base 2^64 in the tables: pack two limbs at a time.
    const auto to64 = [](const Big& x, u64* out, std::size_t n) {
        for (std::size_t l = 0; l < n; ++l)
            out[l] = static_cast<u64>(x.limb(2 * l)) |
                     (static_cast<u64>(x.limb(2 * l + 1)) << 32);
    };

    pl.q64.assign(64, 0);
    to64(q, pl.q64.data(), NQ);
    pl.pmod64.assign(64, 0);
    pl.crt_w.assign(k, std::vector<u64>(64, 0));
    pl.crt_c.assign(k, std::vector<u64>(3, 0));

    for (std::size_t j = 0; j < k; ++j) {
        divmod(idem[j], q, qq, rr);            // rr = e_j mod q
        divmod(mul(rr, r0), q, qq, rr);        // rr = e_j * 2^(64NQ) mod q
        to64(rr, pl.crt_w[j].data(), NQ);

        divmod(shl_limbs(idem[j], 6), pl.P, qq, rr);  // floor(e_j * 2^192 / P)
        to64(qq, pl.crt_c[j].data(), 3);
    }

    divmod(mul(pl.P_mod_q, r0), q, qq, rr);  // (P mod q) * 2^128 mod q
    to64(rr, pl.pmod64.data(), NQ);

    pl.qsh64.assign(64, 0);
    to64(shl_bits(q, 68), pl.qsh64.data(), NQ);  // q * 2^68 stays inside crt_nq

    u64 inv = 1;
    const u64 q0 = static_cast<u64>(q.limb(0)) | (static_cast<u64>(q.limb(1)) << 32);
    for (int i = 0; i < 6; ++i) inv *= 2 - q0 * inv;  // q^{-1} mod 2^64
    pl.qneg_inv = ~inv + 1;
}

inline Plan build_plan(u32 N, u32 W, u32 L, const u32* q_limbs) {
    if (N == 0 || (N & (N - 1)) != 0) throw std::runtime_error("polymul: N must be a power of two");
    Plan pl;
    pl.N = N;
    pl.W = W;
    pl.L = L;
    pl.q = Big::from_le(q_limbs, L);

    // Need P > 2N (q-1)^2  — twice the largest signed convolution coefficient.
    const Big qm1 = sub(pl.q, Big(1));
    const Big bound = mul(mul(qm1, qm1), Big(2ull * N));

    // Primes: descend from just below 2^62, keeping p == 1 (mod 2N).
    const u64 step = 2ull * N;
    const u64 cap = (1ull << 62) - 1;
    u64 cand = 1 + step * ((cap - 1) / step);

    pl.P = Big(1);
    while (cmp(pl.P, bound) <= 0) {
        while (cand > step && !is_prime(cand)) cand -= step;
        if (cand <= step) throw std::runtime_error("polymul: ran out of primes");
        pl.primes.push_back(cand);
        pl.P = mul(pl.P, Big(cand));
        cand -= step;
    }

    const std::size_t k = pl.primes.size();
    pl.mods.reserve(k);
    for (u64 p : pl.primes) pl.mods.push_back(Mod::make(p));

    pl.root_fwd.resize(k);
    pl.root_inv.resize(k);
    pl.twist_fwd.resize(k);
    pl.twist_inv.resize(k);
    pl.limb_w.resize(k);
    for (std::size_t m = 0; m < k; ++m) build_tables(pl, m);

    pl.inv_p.assign(k, std::vector<u64>(k, 0));
    for (std::size_t m = 1; m < k; ++m)
        for (std::size_t j = 0; j < m; ++j)
            pl.inv_p[j][m] = inv_mod(pl.primes[j] % pl.primes[m], pl.primes[m]);

    pl.P_half = halve(pl.P);
    Big qq, rr;
    divmod(pl.P, pl.q, qq, rr);
    pl.P_mod_q = rr;

    pl.m_limbs = pl.P.limbs() + 1;
    Big Bm;
    Bm.d.assign(pl.m_limbs + 1, 0u);
    Bm.d[pl.m_limbs] = 1u;  // 2^(32*m_limbs)
    divmod(Bm, pl.q, pl.mu, rr);

    pl.crt_nq = crt_pick_nq((pl.q.bits() + 63) / 64);
    if (pl.crt_nq) build_crt_tables(pl);
    return pl;
}

inline void build_tables(Plan& pl, std::size_t m) {
    const u64 p = pl.primes[m];
    const Mod& md = pl.mods[m];
    const std::size_t N = pl.N;

    // psi of exact order 2N
    u64 psi = 0;
    const u64 exp = (p - 1) / (2ull * N);
    for (u64 g = 2;; ++g) {
        const u64 c = powmod(g, exp, p);
        if (powmod(c, N, p) == p - 1) { psi = c; break; }
        if (g > 1000) throw std::runtime_error("polymul: no 2N-th root found");
    }
    const u64 omega = mulmod(psi, psi, p);
    const u64 omega_inv = inv_mod(omega, p);
    const u64 psi_inv = inv_mod(psi, p);

    pl.root_fwd[m].assign(N, 0);
    pl.root_inv[m].assign(N, 0);
    for (std::size_t len = N / 2; len >= 1; len >>= 1) {
        const u64 base = powmod(omega, static_cast<u64>(N / (2 * len)), p);
        const u64 base_inv = powmod(omega_inv, static_cast<u64>(N / (2 * len)), p);
        u64 acc = 1, acc_inv = 1;
        for (std::size_t j = 0; j < len; ++j) {
            pl.root_fwd[m][len + j] = md.to_mont(acc);
            pl.root_inv[m][len + j] = md.to_mont(acc_inv);
            acc = mulmod(acc, base, p);
            acc_inv = mulmod(acc_inv, base_inv, p);
        }
        if (len == 1) break;  // size_t: len >>= 1 would wrap
    }

    // twist tables
    pl.twist_fwd[m].assign(N, 0);
    pl.twist_inv[m].assign(N, 0);
    const u64 n_inv = inv_mod(static_cast<u64>(N) % p, p);
    u64 t = 1, ti = 1;
    for (std::size_t i = 0; i < N; ++i) {
        pl.twist_fwd[m][i] = md.to_mont(t);
        pl.twist_inv[m][i] = mulmod(ti, n_inv, p);  // plain, not Montgomery
        t = mulmod(t, psi, p);
        ti = mulmod(ti, psi_inv, p);
    }

    // limb weights: limb * 2^(32l) * R  in the Montgomery domain
    pl.limb_w[m].assign(pl.L, 0);
    const u64 two32 = (u64(1) << 32) % p;
    u64 w = 1 % p;
    for (std::size_t l = 0; l < pl.L; ++l) {
        pl.limb_w[m][l] = mulmod(w, md.r2, p);  // w * R^2
        w = mulmod(w, two32, p);
    }
}

// -------------------------------------------------------------- parallelism
// Dependency-free fork-join over a half-open index range, split into
// contiguous chunks. Every use below gives the chunk a private output range, so
// the answer is bit-identical to the serial loop and no synchronisation is
// needed inside `body`. `body` must not throw: an exception escaping a worker
// would terminate the process.
template <class F>
inline void parallel_chunks(std::size_t count, unsigned threads, F&& body) {
    if (count == 0) return;
    if (threads <= 1 || count == 1) {
        body(std::size_t(0), count);
        return;
    }
    if (threads > count) threads = static_cast<unsigned>(count);
    std::vector<std::thread> pool;
    pool.reserve(threads - 1);
    const auto chunk = [&](unsigned t) {
        body(count * t / threads, count * (t + 1) / threads);
    };
    for (unsigned t = 0; t + 1 < threads; ++t) pool.emplace_back(chunk, t);
    chunk(threads - 1);
    for (auto& th : pool) th.join();
}

// ------------------------------------------------------------------- engine
//
// Thread-safe across calls only in the sense that `multiply` parallelises
// internally over its own scratch; two concurrent calls would share A and B, so
// the caller owns one Engine per concurrent stream (as the GPU version would).
struct Engine {
    Plan plan;
    unsigned threads = 1;
    std::vector<u64> A, B;  // k * N each, slice m belongs to prime m

    explicit Engine(const Plan& p, unsigned nthreads = 1) : plan(p), threads(nthreads ? nthreads : 1) {
        A.assign(plan.k() * plan.N, 0);
        B.assign(plan.k() * plan.N, 0);
    }

    void ntt_fwd(u64* a, std::size_t m) const {
        const Mod& md = plan.mods[m];
        const u64* roots = plan.root_fwd[m].data();
        const std::size_t n = plan.N;
        for (std::size_t len = n / 2; len >= 1; len >>= 1) {
            for (std::size_t i = 0; i < n; i += 2 * len) {
                for (std::size_t j = 0; j < len; ++j) {
                    const u64 u = a[i + j], v = a[i + j + len];
                    a[i + j] = md.add(u, v);
                    a[i + j + len] = md.mul(md.sub(u, v), roots[len + j]);
                }
            }
            if (len == 1) break;
        }
    }

    void ntt_inv(u64* a, std::size_t m) const {
        const Mod& md = plan.mods[m];
        const u64* roots = plan.root_inv[m].data();
        const std::size_t n = plan.N;
        for (std::size_t len = 1; len < n; len <<= 1) {
            for (std::size_t i = 0; i < n; i += 2 * len) {
                for (std::size_t j = 0; j < len; ++j) {
                    const u64 u = a[i + j];
                    const u64 v = md.mul(a[i + j + len], roots[len + j]);
                    a[i + j] = md.add(u, v);
                    a[i + j + len] = md.sub(u, v);
                }
            }
        }
    }

    // coeffs: N x L u32 limbs -> residues (Montgomery, twisted) for prime m
    void decompose(const u32* coeffs, u64* out, std::size_t m) const {
        const Mod& md = plan.mods[m];
        const u64* lw = plan.limb_w[m].data();
        const u64* tw = plan.twist_fwd[m].data();
        for (std::size_t i = 0; i < plan.N; ++i) {
            const u32* c = coeffs + i * plan.L;
            u64 r = 0;
            for (std::size_t l = 0; l < plan.L; ++l)
                r = md.add(r, md.mul(c[l], lw[l]));
            out[i] = md.mul(r, tw[i]);
        }
    }

    // one negacyclic product: a, b, c are N x L u32 limbs
    void multiply(const u32* a, const u32* b, u32* c) {
        const std::size_t N = plan.N, k = plan.k();

        // Stage 1 — one independent RNS/NTT pipeline per prime. Slice m of A
        // and B is written and read only by the chunk that owns m.
        parallel_chunks(k, threads, [&](std::size_t mlo, std::size_t mhi) {
            for (std::size_t m = mlo; m < mhi; ++m) {
                u64* pa = A.data() + m * N;
                u64* pb = B.data() + m * N;
                decompose(a, pa, m);
                decompose(b, pb, m);
                ntt_fwd(pa, m);
                ntt_fwd(pb, m);
                const Mod& md = plan.mods[m];
                for (std::size_t i = 0; i < N; ++i) pa[i] = md.mul(pa[i], pb[i]);
                ntt_inv(pa, m);
                const u64* ti = plan.twist_inv[m].data();
                for (std::size_t i = 0; i < N; ++i) pa[i] = md.mul(pa[i], ti[i]);
            }
        });

        // Stage 2 — one independent output coefficient per index.
        parallel_chunks(N, threads, [&](std::size_t ilo, std::size_t ihi) {
            std::vector<u64> digit(k, 0);
            for (std::size_t i = ilo; i < ihi; ++i) {
                for (std::size_t m = 0; m < k; ++m) {
                    const u64 pm = plan.primes[m];
                    u64 t = A[m * N + i];
                    for (std::size_t j = 0; j < m; ++j)
                        t = mulmod((t + pm - digit[j]) % pm, plan.inv_p[j][m], pm);
                    digit[m] = t;
                }
                Big X(static_cast<u64>(digit[k - 1]));
                for (std::size_t m = k - 1; m-- > 0;) mul_u64_add(X, plan.primes[m], digit[m]);

                Big reduced = barrett(X);
                if (cmp(X, plan.P_half) > 0) {  // the true value is X - P, i.e. negative
                    reduced = cmp(reduced, plan.P_mod_q) >= 0
                                  ? sub(reduced, plan.P_mod_q)
                                  : sub(add(reduced, plan.q), plan.P_mod_q);
                }
                reduced.to_le(c + i * plan.L, plan.L);
            }
        });
    }

    Big barrett(const Big& x) const {
        const Big qhat = shr_limbs(mul(x, plan.mu), plan.m_limbs);
        Big r = sub(x, mul(qhat, plan.q));
        while (cmp(r, plan.q) >= 0) r = sub(r, plan.q);
        return r;
    }
};

}  // namespace polymul
