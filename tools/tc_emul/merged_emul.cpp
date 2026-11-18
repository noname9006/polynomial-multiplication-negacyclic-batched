// Merged-psi negacyclic NTT in the TC pipeline layout (no twist / untwist).
// Data are stored at bit-reversed positions (encode reads row brv(i)), the
// forward runs the DIT butterfly graph (low stages half=1..256 in 512-tiles,
// then high stages across tiles), pointwise product, then the exact inverse
// (DIF, high stages from the top, then low), which leaves n * (a*b) at the
// bit-reversed slots.  Forward twiddle of the pair (r, r+half):
//     w = psi^{ brv_n( half + brv_{log2 half}(r mod half) ) }
// Inverse twiddle = w^-1.  High-stage factorisation (half = 512*2^s,
// r mod half = within*512 + low):  exponent = 2^{n-10-s}(1+2 low) + within*2^{n-s}.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
typedef uint32_t u32; typedef uint64_t u64;
static u32 P;
u32 mulm(u32 a, u32 b) { return u32(u64(a) * b % P); }
u32 addm(u32 a, u32 b) { u32 s = a + b; return s >= P ? s - P : s; }
u32 subm(u32 a, u32 b) { return a >= b ? a - b : a + P - b; }
u32 pw(u32 a, u64 e) { u32 r = 1; while (e) { if (e & 1) r = mulm(r, a); a = mulm(a, a); e >>= 1; } return r; }
u32 brv(u32 x, unsigned bits) { u32 r = 0; for (unsigned i = 0; i < bits; i++) { r = (r << 1) | (x & 1); x >>= 1; } return r; }
bool isprime(u32 p) { for (u32 d = 2; u64(d) * d <= p; d++) if (p % d == 0) return false; return true; }
int main(int argc, char** argv) {
  const unsigned logN = argc > 1 ? atoi(argv[1]) : 12, N = 1u << logN, T = 512, logT = 9, HC = N / T, logH = logN - logT;
  P = ((1u << 31) - 2) / (2 * N) * (2 * N) + 1; while (!isprime(P)) P -= 2 * N;
  u32 psi = 0; for (u32 a = 2; !psi; a++) { u32 c = pw(a, (P - 1) / (2 * N)); if (pw(c, N) == P - 1) psi = c; }
  const u32 psii = pw(psi, P - 2);
  auto twF = [&](u32 half, u32 rmod) { unsigned lh = 0; while ((1u << lh) < half) lh++; return pw(psi, brv(half + brv(rmod, lh), logN)); };
  auto twI = [&](u32 half, u32 rmod) { unsigned lh = 0; while ((1u << lh) < half) lh++; return pw(psii, brv(half + brv(rmod, lh), logN)); };
  // factorised high twiddle check
  int fbad = 0;
  for (unsigned s = 0; s < logH; s++) for (u32 w = 0; w < (1u << s); w++) for (u32 low = 0; low < T; low += 37) {
    const u32 half = T << s; const u64 e = (u64(1) << (logN - 10 - s)) * (1 + 2 * low) + u64(w) * (u64(1) << (logN - s));
    if (pw(psi, e % (2 * N)) != twF(half, w * T + low)) fbad++; }
  std::mt19937 g(9); std::vector<u32> A(N), B(N); for (auto& x : A) x = g() % P; for (auto& x : B) x = g() % P;
  std::vector<u32> ra(N), rb(N);
  for (unsigned i = 0; i < N; i++) { ra[i] = A[brv(i, logN)]; rb[i] = B[brv(i, logN)]; }
  auto fwd = [&](std::vector<u32>& x) {
    for (unsigned m = 0; m < logN; m++) { const u32 half = 1u << m;
      for (u32 r = 0; r < N; r++) { if (r & half) continue; const u32 w = twF(half, r & (half - 1));
        const u32 u = x[r], v = mulm(x[r + half], w); x[r] = addm(u, v); x[r + half] = subm(u, v); } } };
  auto inv = [&](std::vector<u32>& x) {
    for (int m = logN - 1; m >= 0; m--) { const u32 half = 1u << m;
      for (u32 r = 0; r < N; r++) { if (r & half) continue; const u32 w = twI(half, r & (half - 1));
        const u32 u = x[r], v = x[r + half]; x[r] = addm(u, v); x[r + half] = mulm(subm(u, v), w); } } };
  fwd(ra); fwd(rb);
  std::vector<u32> rc(N); for (unsigned i = 0; i < N; i++) rc[i] = mulm(ra[i], rb[i]);
  inv(rc);
  const u32 ninv = pw(N, P - 2);
  int bad = 0;
  for (unsigned k = 0; k < N; k++) { u64 acc = 0;
    for (unsigned i = 0; i < N; i++) { unsigned j = (k + N - i) % N; u64 pr = u64(A[i]) * B[j] % P; acc = (i <= k) ? (acc + pr) % P : (acc + P - pr) % P; }
    if (mulm(rc[brv(k, logN)], ninv) != acc) bad++; }
  printf("merged-psi: N=%u factor_bad=%d result_bad=%d\n", N, fbad, bad);
}
