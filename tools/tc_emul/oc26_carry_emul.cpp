// oc26_carry_emul.cpp -- CPU emulation of crt_tc2 phase 4 carry propagation.
//
// FLAG=0: exact sequential path (current crt_tc2, one thread per row).
// FLAG=1: OC26_CARRY path: 4 threads per row, each handles one contiguous
//         segment of the NT*2 u64 words; digits go to a SEPARATE buffer;
//         a per-row owner propagates carries between the 4 segments using
//         digit-add with early exit, with an exact-replay fallback for
//         segments containing huge (>= 2^48) u64 addends (where u64 wrap
//         in `carry += xs[w]` breaks the small-carry recurrence).
//
// Build: g++ -std=c++17 -O2 -o oc26_carry_emul oc26_carry_emul.cpp
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "../qtable.hpp"

using u32 = std::uint32_t;
using u64 = std::uint64_t;

static constexpr unsigned SEGS = 4;

struct Seg { unsigned a, len; };
static Seg seg_of(unsigned words, unsigned s) {
    const unsigned base = words / SEGS, rem = words % SEGS;
    return Seg{ s * base + (s < rem ? s : rem), base + (s < rem ? 1u : 0u) };
}

// ---- FLAG 0: one thread per row, in-place over sX-wide digit array ----
static void carry_seq(const u64* xs, unsigned words, unsigned total, u32* dg) {
    u64 carry = 0;
    for (unsigned w = 0; w < words; ++w) {
        carry += xs[w];
        dg[w] = u32(carry);
        carry >>= 32;
    }
    for (unsigned w = words; w < total; ++w) {
        dg[w] = u32(carry);
        carry >>= 32;
    }
}

// ---- FLAG 1: 4 threads/row, separate digit buffer, owner propagation ----
static void carry_par(const u64* xs, unsigned words, unsigned total, u32* dg) {
    // Mirrors oc26_carry_row exactly: 4 threads/row, cout[] with bit63 replay.
    Seg sg[4]; u64 cout[4];
    for (unsigned s = 0; s < 4; ++s) {
        sg[s] = seg_of(words, s);
        u64 run = 0, big = 0;
        for (unsigned j = 0; j < sg[s].len; ++j) {
            const u64 x = xs[sg[s].a + j];
            const u64 t = run + x;
            dg[sg[s].a + j] = u32(t);
            run = t >> 32;
            big |= x >> 48;
        }
        cout[s] = run | (big ? (u64(1) << 63) : u64(0));
    }
    u64 cin = 0; bool replay = false;
    for (unsigned s = 0; s < 4; ++s) replay |= (cout[s] >> 63) != 0;
    if (replay) {
        u64 carry = 0;
        for (unsigned w = 0; w < words; ++w) {
            carry += xs[w]; dg[w] = u32(carry); carry >>= 32;
        }
        cin = carry;
    } else {
        for (unsigned s = 0; s < 4; ++s) {
            u64 delta = cin;
            if (delta) {
                for (unsigned j = 0; j < sg[s].len && delta; ++j) {
                    const u64 sum = u64(dg[sg[s].a + j]) + u32(delta);
                    dg[sg[s].a + j] = u32(sum);
                    delta = (delta >> 32) + (sum >> 32);
                }
            }
            cin = (cout[s] & ~(u64(1) << 63)) + delta;
        }
    }
    for (unsigned w = words; w < total; ++w) { dg[w] = u32(cin); cin >>= 32; }
}

// ---- gpu31 reduce_sum<1> subset, translated from gpu31.cuh ----
static u32 read_window(const u32* a, unsigned size, unsigned offset) {
    const unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos] >> off : 0;
    if (off && pos + 1 < size) lo |= a[pos + 1] << (32 - off);
    return lo;
}
static void sub_if_ge(u32* a, unsigned size, const u32* q, unsigned limbs, unsigned s) {
    u32 bor = 0;
    for (unsigned j = 0; j < size; ++j) {
        u32 v = a[j], qq = j < limbs ? q[j] : 0;
        u32 d = v - qq - bor;
        bor = (v < qq + bor) || (qq > 0 && v - bor > 0xffffffffu - qq && qq == 0xffffffffu);
        a[j] = d;
    }
}
static void reduce_sum1(u32* a, unsigned limbs, unsigned bits, u32 gap, const u32* q) {
    const unsigned size = limbs + 2;
    if (gap) {
        for (unsigned it = 0; it < 5; ++it) {
            const u32 h0 = read_window(a, size, bits), h1 = read_window(a, size, bits + 32);
            if (!(h0 | h1)) { sub_if_ge(a, size, q, limbs, 0); return; }
            unsigned cut = bits / 32, off = bits % 32;
            if (off) a[cut++] &= (u32(1) << off) - 1;
            for (unsigned j = cut; j < size; ++j) a[j] = 0;
            u64 carry = u64(h0) * gap + a[0]; a[0] = u32(carry); carry >>= 32;
            carry += u64(h1) * gap + a[1]; a[1] = u32(carry); carry >>= 32;
            for (unsigned j = 2; carry && j < size; ++j) { carry += a[j]; a[j] = u32(carry); carry >>= 32; }
        }
    }
    for (unsigned s = 43; s-- > 0;) {
        u32 bor = 0;
        for (unsigned j = 0; j < size; ++j) {
            u32 v = a[j], qq = j < limbs ? q[j] : 0;
            u32 d = v - qq - bor;
            bor = (v < qq + bor) || (qq > 0 && (v - bor) > 0xffffffffu - qq);
            a[j] = d;
        }
    }
}

static int check(const char* name, const std::vector<u64>& xs,
                 unsigned words, unsigned total) {
    std::vector<u32> a(total, 0), b(total, 0);
    carry_seq(xs.data(), words, total, a.data());
    carry_par(xs.data(), words, total, b.data());
    int bad = 0;
    for (unsigned i = 0; i < total; ++i) if (a[i] != b[i]) ++bad;
    std::printf("%-28s words=%u total=%u bad=%d\n", name, words, total, bad);
    if (bad) {
        for (unsigned i = 0; i < total && bad; ++i)
            if (a[i] != b[i]) std::printf("  first mismatch w=%u seq=%08x par=%08x\n", i, a[i], b[i]);
    }
    return bad;
}

int main() {
    int total_bad = 0;
    std::mt19937_64 rng(12345);
    const unsigned words = 56, total = 57;  // K=114,L=55

    { std::vector<u64> x(words, 0); total_bad += check("zeros", x, words, total); }
    { std::vector<u64> x(words, ~u64(0)); total_bad += check("all xs = 2^64-1", x, words, total); }
    { // long carry chain: every digit 0xffffffff, carry 1 threaded through all segments
      std::vector<u64> x(words, 0);
      for (unsigned w = 0; w < words; ++w) x[w] = u64(0xffffffffu);  // cout of each seg = 0
      x[words - 1] = (u64(1) << 32);  // force final carry 1 into trailing word
      total_bad += check("long chain 0xffffffff", x, words, total);
    }
    { // each segment emits carry 1 via xs = 2^32, rest zero
      std::vector<u64> x(words, 0);
      for (unsigned s = 0; s < SEGS; ++s) {
          Seg sg = seg_of(words, s);
          x[sg.a] = u64(1) << 32;
      }
      total_bad += check("carry 1 in every seg", x, words, total);
    }
    { // huge addends mixed with zeros (u64 wrap stress)
      std::vector<u64> x(words, 0);
      for (unsigned w = 0; w < words; w += 3) x[w] = ~u64(0);
      total_bad += check("alternating huge", x, words, total);
    }
    { // realistic magnitudes (< 2^40)
      std::vector<u64> x(words);
      for (auto& v : x) v = rng() & ((u64(1) << 40) - 1);
      total_bad += check("realistic 2^40", x, words, total);
    }
    { // full-range random fuzz
      for (unsigned trial = 0; trial < 200; ++trial) {
          std::vector<u64> x(words);
          for (auto& v : x) v = rng();
          if (check("fuzz full u64", x, words, total)) { ++total_bad; break; }
      }
    }
    { // other sizes: words not divisible by 4, trailing=0 or 1
      for (unsigned wds : {14u, 15u, 29u, 58u}) {
          for (unsigned tt = wds; tt <= wds + 1; ++tt) {
              std::vector<u64> x(wds);
              for (auto& v : x) v = rng() & ((u64(1) << 44) - 1);
              total_bad += check("shape fuzz", x, wds, tt);
          }
      }
    }
    { // end-to-end: carry+reduce for the ranked point W=1747 (55 limbs)
      const KnownPoint* kp = nullptr;
      for (const auto& k : kPoints) if (k.N == 1024 && k.W == 1747) kp = &k;
      unsigned limbs = 55, bits = 1747;
      std::vector<u32> q(limbs, 0);
      if (kp) {
          std::string h(kp->hex);
          for (unsigned i = 0; i < limbs; ++i) {
              u32 v = 0;
              for (unsigned b = 0; b < 8; ++b) {
                  int c = h[h.size() - 1 - (i * 8 + b)];
                  v |= u32(c <= '9' ? c - '0' : c - 'a' + 10) << (4 * b);
              }
              q[i] = v;
          }
      }
      u32 gap = 0xffffffffu - q[0];
      unsigned words = 56, total = 57;
      int e2e = 0;
      for (unsigned t = 0; t < 50; ++t) {
          std::vector<u64> x(words);
          for (auto& v : x) v = rng() & ((u64(1) << 44) - 1);
          std::vector<u32> a(total, 0), b(total, 0);
          carry_seq(x.data(), words, total, a.data());
          carry_par(x.data(), words, total, b.data());
          reduce_sum1(a.data(), limbs, bits, gap, q.data());
          reduce_sum1(b.data(), limbs, bits, gap, q.data());
          if (a != b) { ++e2e; break; }
      }
      std::printf("end-to-end carry+reduce (W=1747): bad=%d\n", e2e);
      total_bad += e2e;
    }
    std::printf("TOTAL BAD = %d\n", total_bad);
    return total_bad != 0;
}
