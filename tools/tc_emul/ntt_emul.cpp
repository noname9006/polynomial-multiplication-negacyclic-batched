// CPU emulation of the TC3 NTT pipeline (fwd_low, fused_high, inv_low, slot CRT read)
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <cstdlib>
typedef uint32_t u32; typedef uint64_t u64;
struct Prime { u32 p, neg_inv, r2, one; };
u32 mont(u32 a, u32 b, const Prime& p) { u64 t = u64(a) * b; u32 m = u32(t) * p.neg_inv; u32 u = u32((t + u64(m) * p.p) >> 32); return u >= p.p ? u - p.p : u; }
u32 add(u32 a, u32 b, u32 p) { u32 s = a + b; return s >= p ? s - p : s; }
u32 sub(u32 a, u32 b, u32 p) { return a >= b ? a - b : a + p - b; }
void butterfly(u32& a, u32& b, u32 w, const Prime& p) { u32 u = a, v = mont(b, w, p); a = add(u, v, p.p); b = sub(u, v, p.p); }
u32 rev(u32 x, unsigned bits) { u32 r = 0; for (unsigned i = 0; i < bits; i++) { r = (r << 1) | (x & 1); x >>= 1; } return r; }
u32 pw(u32 a, u64 e, u32 p) { u32 r = 1; while (e) { if (e & 1) r = u32(u64(r) * a % p); a = u32(u64(a) * a % p); e >>= 1; } return r; }
bool isprime(u32 p) { for (u32 d = 2; u64(d) * d <= p; d++) if (p % d == 0) return false; return true; }
int main(int argc, char** argv) {
  const unsigned logN = argc > 1 ? atoi(argv[1]) : 12, N = 1u << logN, T = 512, HC = N / T, logH = logN - 9;
  const bool R4 = argc > 2 && atoi(argv[2]);
  u32 pp = ((1u << 31) - 2) / (2 * N) * (2 * N) + 1; while (!isprime(pp)) pp -= 2 * N;
  u32 inv = 1; for (int j = 0; j < 5; j++) inv *= 2 - pp * inv;
  u32 one = u32((u64(1) << 32) % pp);
  Prime pr{pp, u32(0 - inv), u32(u64(one) * one % pp), one};
  u32 psi = 0; for (u32 a = 2; !psi; a++) { u32 c = pw(a, (pp - 1) / (2 * N), pp); if (pw(c, N, pp) == pp - 1) psi = c; }
  u32 psi_inv = pw(psi, pp - 2, pp), omega = u32(u64(psi) * psi % pp), omega_inv = pw(omega, pp - 2, pp), n_inv = pw(N, pp - 2, pp);
  std::vector<u32> roots(N / 2), iroots(N / 2), twist(N), untwist(N), rtwist(N);
  { u32 a = 1, b = 1; for (unsigned i = 0; i < N / 2; i++) { roots[i] = mont(a, pr.r2, pr); iroots[i] = mont(b, pr.r2, pr); a = u32(u64(a) * omega % pp); b = u32(u64(b) * omega_inv % pp); } }
  { u32 a = 1, b = n_inv; for (unsigned i = 0; i < N; i++) { twist[i] = mont(a, pr.r2, pr); untwist[i] = mont(b, pr.r2, pr); a = u32(u64(a) * psi % pp); b = u32(u64(b) * psi_inv % pp); } }
  for (unsigned i = 0; i < N; i++) rtwist[rev(i, logN)] = twist[i];
  // tables as in build_ntt_tables
  std::vector<u32> pf(T / 2), pi(T / 2), tf((HC - 1) * T), ti(tf.size());
  for (unsigned p = 0; p < T / 2; p++) { pf[p] = roots[p * HC]; pi[p] = iroots[p * HC]; }
  for (unsigned st = 0; st < logH; st++) { unsigned H = 1u << st, off = T * (H - 1);
    for (unsigned w = 0; w < H; w++) for (unsigned low = 0; low < T; low++) { size_t e = size_t(w * T + low) << (logH - st - 1); tf[off + w * T + low] = roots[e]; ti[off + w * T + low] = iroots[e]; } }
  std::mt19937 g(7); std::vector<u32> A(N), B(N); for (auto& x : A) x = g() % pp; for (auto& x : B) x = g() % pp;
  // encode: ra[i] = mont(X*R mod p, rtwist[i]) with X = A[rev i]
  std::vector<u32> ra(N), rb(N), rc(N);
  for (unsigned i = 0; i < N; i++) { u32 xa = u32((u64(A[rev(i, logN)]) << 32) % pp), xb = u32((u64(B[rev(i, logN)]) << 32) % pp); ra[i] = mont(xa, rtwist[i], pr); rb[i] = mont(xb, rtwist[i], pr); }
  // fwd_low
  for (unsigned bx = 0; bx < N / T; bx++) { u32* a = &ra[bx * T]; u32* b = &rb[bx * T];
    if (argc > 3 && atoi(argv[3]) == 8) { for (unsigned m = 0; m < 9; m += 3) { unsigned H = 1u << m; for (unsigned gi = 0; gi < 64; gi++) { unsigned w = gi & (H - 1), x0 = (gi - w) * 8 + w;
        for (u32* X : {a, b}) for (unsigned st = 0; st < 3; st++) { unsigned span = 1u << st; for (unsigned e = 0; e < 8; e++) { if (e & span) continue; unsigned j = w + (e & (span - 1)) * H; butterfly(X[x0 + e * H], X[x0 + (e + span) * H], pf[j << (8 - (m + st))], pr); } } } } } else
    if (R4) { for (unsigned m = 0; m < 8; m += 2) { unsigned H = 1u << m; for (unsigned gi = 0; gi < 128; gi++) { unsigned w = gi & (H - 1), x0 = (gi - w) * 4 + w; u32 t1 = pf[w << (8 - m)], t2 = pf[w << (7 - m)], t3 = pf[(w + H) << (7 - m)];
        for (u32* X : {a, b}) { butterfly(X[x0], X[x0 + H], t1, pr); butterfly(X[x0 + 2 * H], X[x0 + 3 * H], t1, pr); butterfly(X[x0], X[x0 + 2 * H], t2, pr); butterfly(X[x0 + H], X[x0 + 3 * H], t3, pr); } } }
      for (unsigned t = 0; t < 256; t++) { unsigned m = 8, half = 256, j = t & (half - 1), first = 2 * (t - j) + j, second = first + half; u32 w = pf[j << (8 - m)]; butterfly(a[first], a[second], w, pr); butterfly(b[first], b[second], w, pr); } } else
    for (unsigned m = 0; m < 9; m++) for (unsigned t = 0; t < 256; t++) { unsigned half = 1u << m, j = t & (half - 1), first = 2 * (t - j) + j, second = first + half; u32 w = pf[j << (8 - m)]; butterfly(a[first], a[second], w, pr); butterfly(b[first], b[second], w, pr); } }
  // fused_high (per column low)
  for (unsigned low = 0; low < T; low++) {
    std::vector<u32> sa(HC), sb(HC); for (unsigned h = 0; h < HC; h++) { sa[h] = ra[h * T + low]; sb[h] = rb[h * T + low]; }
    auto TWF=[&](unsigned st, unsigned w){ return mont(roots[size_t(low) << (logH - st - 1)], roots[size_t(w * T) << (logH - st - 1)], pr); };
    auto TWI=[&](unsigned st, unsigned w){ return mont(iroots[size_t(low) << (logH - st - 1)], iroots[size_t(w * T) << (logH - st - 1)], pr); };
    if (R4) {
      for (unsigned s = 0; s + 1 < logH; s += 2) { unsigned H = 1u << s; for (unsigned g = 0; g < HC / 4; g++) { unsigned w = g & (H - 1), r0 = (g - w) * 4 + w;
        u32 t1 = TWF(s, w), t2 = TWF(s + 1, w), t3 = TWF(s + 1, w + H);
        for (std::vector<u32>* X : {&sa, &sb}) { auto& x = *X; butterfly(x[r0], x[r0 + H], t1, pr); butterfly(x[r0 + 2 * H], x[r0 + 3 * H], t1, pr); butterfly(x[r0], x[r0 + 2 * H], t2, pr); butterfly(x[r0 + H], x[r0 + 3 * H], t3, pr); } } }
      if (logH & 1) { unsigned s = logH - 1, H = 1u << s; for (unsigned h = 0; h < HC / 2; h++) { unsigned within = h & (H - 1), first = 2 * (h - within) + within, second = first + H; u32 w = TWF(s, within); butterfly(sa[first], sa[second], w, pr); butterfly(sb[first], sb[second], w, pr); } }
    } else
    for (unsigned s = 0; s < logH; s++) { unsigned H = 1u << s, off = T * (H - 1); for (unsigned h = 0; h < HC / 2; h++) { unsigned within = h & (H - 1), first = 2 * (h - within) + within, second = first + H; u32 w = mont(roots[size_t(low) << (logH - s - 1)], roots[size_t(within * T) << (logH - s - 1)], pr); butterfly(sa[first], sa[second], w, pr); butterfly(sb[first], sb[second], w, pr); } }
    for (unsigned h = 0; h < HC; h++) sa[h] = mont(sa[h], sb[h], pr);
    if (R4) {
      if (logH & 1) { unsigned s = logH - 1, H = 1u << s; for (unsigned h = 0; h < HC / 2; h++) { unsigned within = h & (H - 1), first = 2 * (h - within) + within, second = first + H; u32 w = TWI(s, within); u32 u = sa[first], v = sa[second]; sa[first] = add(u, v, pp); sa[second] = mont(sub(u, v, pp), w, pr); } }
      for (int s = 4; s >= 0; s -= 2) { if (unsigned(s) + 2 > logH - (logH & 1)) continue; unsigned H = 1u << s;
        for (unsigned g = 0; g < HC / 4; g++) { unsigned w = g & (H - 1), r0 = (g - w) * 4 + w; u32 t2 = TWI(s + 1, w), t3 = TWI(s + 1, w + H), t1 = TWI(s, w);
          u32 x0 = sa[r0], x1 = sa[r0 + H], x2 = sa[r0 + 2 * H], x3 = sa[r0 + 3 * H], u, v;
          u = x0; v = x2; x0 = add(u, v, pp); x2 = mont(sub(u, v, pp), t2, pr);
          u = x1; v = x3; x1 = add(u, v, pp); x3 = mont(sub(u, v, pp), t3, pr);
          u = x0; v = x1; x0 = add(u, v, pp); x1 = mont(sub(u, v, pp), t1, pr);
          u = x2; v = x3; x2 = add(u, v, pp); x3 = mont(sub(u, v, pp), t1, pr);
          sa[r0] = x0; sa[r0 + H] = x1; sa[r0 + 2 * H] = x2; sa[r0 + 3 * H] = x3; } }
    } else
    for (unsigned s = logH; s-- > 0;) { unsigned H = 1u << s, off = T * (H - 1); for (unsigned h = 0; h < HC / 2; h++) { unsigned within = h & (H - 1), first = 2 * (h - within) + within, second = first + H; u32 w = mont(iroots[size_t(low) << (logH - s - 1)], iroots[size_t(within * T) << (logH - s - 1)], pr); u32 u = sa[first], v = sa[second]; sa[first] = add(u, v, pp); sa[second] = mont(sub(u, v, pp), w, pr); } }
    for (unsigned h = 0; h < HC; h++) rc[h * T + low] = sa[h];
  }
  // inv_low
  for (unsigned bx = 0; bx < N / T; bx++) { u32* a = &rc[bx * T];
    for (int m = 8; m >= 0; m--) for (unsigned t = 0; t < 256; t++) { unsigned half = 1u << m, j = t & (half - 1), first = 2 * (t - j) + j, second = first + half; u32 w = pi[j << (8 - m)]; u32 u = a[first], v = a[second]; a[first] = add(u, v, pp); a[second] = mont(sub(u, v, pp), w, pr); } }
  // slot read: term = mont(rc[slot], ts[slot]) with inv=1: ts[slot] = mont(untwist[rev slot], 1)
  std::vector<u32> got(N);
  for (unsigned sl = 0; sl < N; sl++) { unsigned pos = rev(sl, logN); got[pos] = mont(rc[sl], mont(untwist[pos], 1, pr), pr); }
  // naive negacyclic
  int bad = 0;
  for (unsigned kk = 0; kk < N; kk++) { u64 acc = 0;
    for (unsigned i = 0; i < N; i++) { unsigned j = (kk + N - i) % N; u64 prod = u64(A[i]) * B[j] % pp; if (i <= kk) acc = (acc + prod) % pp; else acc = (acc + pp - prod) % pp; }
    if (got[kk] != acc) bad++; }
  printf("ntt pipeline bad=%d of %u\n", bad, N);
}
