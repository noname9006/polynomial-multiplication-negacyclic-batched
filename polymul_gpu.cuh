// polymul_gpu.cuh — the device half of the answer.
//
// Only kernel bodies and device helpers live here: no CUDA runtime calls, no
// host code. solve.cu includes it inside #ifdef __CUDACC__ and owns the device
// memory, the streams and the launches. tools/gpucheck.cpp includes it the same
// way under tools/shim/cuda_shim.hpp, where the execution model is one thread
// per block, which is what makes the arithmetic checkable on a machine with no
// GPU.
//
// The kernels do exactly what polymul.hpp does on the host, in the same order,
// with the same tables and the same Montgomery constants, so the two answers
// are comparable bit for bit. Anything that would differ — a different k, a
// bigger big integer than MAXLIMB — makes fherma_init keep the host path
// instead. Slower, never wrong.
#pragma once

// The same types as polymul.hpp, not lookalikes: solve.cu assigns the one's
// vectors to the other's, and `unsigned long long` is not `unsigned long` on
// Linux even though both are 64 bits on Windows. <cstdint> is fine in device
// code; what it is not fine in is a kernel signature, and nothing here crosses
// a launch boundary.
#include <cstdint>

namespace pgpu {

typedef std::uint32_t u32;
typedef std::uint64_t u64;

// Built-for limits. fherma_init checks the real point against these.
enum {
    MAXK = 64,        // largest number of RNS primes
    MAXLIMB = 192,    // largest big integer, in 32-bit limbs
    MAXL = 64,        // largest limb count of one coefficient
    TILE = 128        // coefficients staged in shared memory per block
};

// Everything the kernels read, flattened out of polymul::Plan.
struct Params {
    u32 N, L, k, B;
    const u64* primes;     // k
    const u64* pinv;       // k, -p^-1 mod 2^64
    const u64* r2;         // k, R^2 mod p
    const u64* root_fwd;   // k*N, index m*N + len + j
    const u64* root_inv;   // k*N
    const u64* twist_fwd;  // k*N, Montgomery
    const u64* twist_inv;  // k*N, plain
    const u64* limb_w;     // k*L, (2^(32l) mod p) * R^2
    const u64* inv_p;      // k*k, inv_p[j*k + m] = (p_j mod p_m)^-1
    const u32* P_half;     // np limbs
    const u32* P_mod_q;    // nq limbs
    const u32* qlimbs;     // L limbs
    const u32* mu;         // nmu limbs
    u32 np, nq, nmu, m_limbs;

    // weighted-sum CRT (see Plan): the same tables, in base 2^64.
    // crt_w is one row of 64 limbs per prime, columns past crt_nq are zero.
    const u64* crt_w;    // k * 64
    const u64* crt_c;    // k * 3
    const u64* q64;      // 64 limbs, q zero padded
    const u64* pmod64;   // 64, (P mod q) * 2^128 mod q
    const u64* qsh64;    // 64, q * 2^68
    u64 qneg_inv;        // -q^{-1} mod 2^64
    u32 crt_nq;          // 0 = no table, use k_crt
};

// ------------------------------------------------------------------ arithmetic
//
// Montgomery with R = 2^64, p < 2^62, written with the 64x64->128 high half
// instead of unsigned __int128, which is not guaranteed in device code.
__device__ inline u64 mont_mul(u64 a, u64 b, u64 pinv, u64 p) {
    const u64 lo = a * b;
    const u64 hi = __umul64hi(a, b);
    const u64 m = lo * pinv;
    const u64 mp_lo = m * p;
    const u64 mp_hi = __umul64hi(m, p);
    const u64 s = lo + mp_lo;
    const u64 carry = (s < lo) ? 1ull : 0ull;
    const u64 r = hi + mp_hi + carry;  // < 2p, so one conditional subtract
    return r >= p ? r - p : r;
}

__device__ inline u64 mod_add(u64 a, u64 b, u64 p) {
    const u64 s = a + b;
    return s >= p ? s - p : s;
}

// a*b mod p for plain (non-Montgomery) a, b. Two Montgomery products: the inner
// one leaves a*b/R, the outer multiplies by R^2 and leaves a*b. No division,
// which device code has no cheap way to do on 128 bits.
__device__ inline u64 raw_mul(u64 a, u64 b, u64 p, u64 pinv, u64 r2) {
    return mont_mul(mont_mul(a, b, pinv, p), r2, pinv, p);
}

// ----------------------------------------------------------------- big integer
//
// Base 2^32, little-endian, length carried alongside, trim on the way out. Same
// operations as polymul::Big, on fixed arrays.

__device__ inline u32 big_trim(const u32* x, u32 n) {
    while (n && x[n - 1] == 0) --n;
    return n;
}

__device__ inline int big_cmp(const u32* a, u32 na, const u32* b, u32 nb) {
    na = big_trim(a, na);
    nb = big_trim(b, nb);
    if (na != nb) return na < nb ? -1 : 1;
    for (u32 i = na; i-- > 0;)
        if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
    return 0;
}

// a - b, requires a >= b
__device__ inline u32 big_sub(const u32* a, u32 na, const u32* b, u32 nb, u32* out) {
    long long borrow = 0;
    for (u32 i = 0; i < na; ++i) {
        long long cur = static_cast<long long>(a[i]) - (i < nb ? static_cast<long long>(b[i]) : 0) - borrow;
        borrow = cur < 0 ? 1 : 0;
        if (cur < 0) cur += (1LL << 32);
        out[i] = static_cast<u32>(cur);
    }
    return big_trim(out, na);
}

__device__ inline u32 big_add(const u32* a, u32 na, const u32* b, u32 nb, u32* out) {
    const u32 n = na > nb ? na : nb;
    u64 carry = 0;
    for (u32 i = 0; i < n; ++i) {
        u64 cur = static_cast<u64>(i < na ? a[i] : 0u) + (i < nb ? b[i] : 0u) + carry;
        out[i] = static_cast<u32>(cur);
        carry = cur >> 32;
    }
    if (carry) out[n] = static_cast<u32>(carry);
    return big_trim(out, n + (carry ? 1u : 0u));
}

// out = a*b; out needs na+nb limbs
__device__ inline u32 big_mul(const u32* a, u32 na, const u32* b, u32 nb, u32* out) {
    const u32 n = na + nb;
    for (u32 i = 0; i < n; ++i) out[i] = 0;
    for (u32 i = 0; i < na; ++i) {
        u64 carry = 0;
        const u64 ai = a[i];
        for (u32 j = 0; j < nb; ++j) {
            const u64 cur = ai * b[j] + out[i + j] + carry;  // < 2^64 in base 2^32
            out[i + j] = static_cast<u32>(cur);
            carry = cur >> 32;
        }
        u32 t = i + nb;
        while (carry) {
            const u64 cur = static_cast<u64>(out[t]) + carry;
            out[t] = static_cast<u32>(cur);
            carry = cur >> 32;
            ++t;
        }
    }
    return big_trim(out, n);
}

// x = x*m + addend, in place. Mirrors polymul::mul_u64_add: m is up to 62 bits,
// so each step is a 64x64 -> 128 product and the carry stays below 2^63.
__device__ inline u32 big_mul_add(u32* x, u32 nx, u64 m, u64 addend) {
    u64 carry = addend;
    for (u32 i = 0; i < nx; ++i) {
        const u64 hi = __umul64hi(static_cast<u64>(x[i]), m);
        const u64 lo = static_cast<u64>(x[i]) * m;
        const u64 s = lo + carry;
        const u64 c = (s < lo) ? 1ull : 0ull;
        x[i] = static_cast<u32>(s);
        carry = ((hi + c) << 32) | (s >> 32);
    }
    u32 n = nx;
    while (carry) {
        x[n++] = static_cast<u32>(carry);
        carry >>= 32;
    }
    return big_trim(x, n);
}

// ------------------------------------------------------------------------ ntt
//
// Decimation in time, the same stage order and the same twiddle table layout as
// Engine::ntt_fwd: root[len + j] belongs to the stage whose half-length is len.
// Every thread owns whole butterflies, so the only cross-thread dependency is
// the barrier between stages.
__device__ inline void ntt_fwd(u64* a, u32 n, u64 p, u64 pinv, const u64* roots) {
    for (u32 len = n / 2; len >= 1; len >>= 1) {
        const u32 per = n / 2;
        for (u32 b = threadIdx.x; b < per; b += blockDim.x) {
            const u32 group = b / len;
            const u32 j = b - group * len;
            const u32 i = group * 2 * len + j;
            const u64 u = a[i], v = a[i + len];
            const u64 d = (u >= v) ? u - v : u + p - v;
            a[i] = mod_add(u, v, p);
            a[i + len] = mont_mul(d, roots[len + j], pinv, p);
        }
        __syncthreads();
        if (len == 1) break;  // u32: len >>= 1 would wrap
    }
}

__device__ inline void ntt_inv(u64* a, u32 n, u64 p, u64 pinv, const u64* roots) {
    for (u32 len = 1; len < n; len <<= 1) {
        const u32 per = n / 2;
        for (u32 b = threadIdx.x; b < per; b += blockDim.x) {
            const u32 group = b / len;
            const u32 j = b - group * len;
            const u32 i = group * 2 * len + j;
            const u64 u = a[i];
            const u64 v = mont_mul(a[i + len], roots[len + j], pinv, p);
            a[i] = mod_add(u, v, p);
            a[i + len] = (u >= v) ? u - v : u + p - v;
        }
        __syncthreads();
    }
}

// --------------------------------------------------------- ntt, two passes
//
// The sixteen stages of a length-65536 transform are not all alike. Read the
// array as 256 rows of 256, and the stages with half-length at least 256 pair
// two elements of the same column; the stages below 256 pair two elements of
// the same row. So a block that holds a strip of whole columns in shared memory
// can run its eight stages there, and one that holds a strip of whole rows can
// run the other eight — one read and one write per transform instead of one per
// stage.
//
// The stages, their order and their twiddles are unchanged: element (r, c) of
// the view is at r*STEP + c, and the twiddle is still roots[len + j] with
// j = (i mod len), which for a column stage with len = h*STEP is
// roots[len + j*STEP + c]. The two passes therefore produce the same values as
// ntt_fwd and ntt_inv above, stage for stage, which is what lets gpucheck
// compare them limb for limb.
//
// Only the length of a line and the twiddle's stride differ, so the four
// kernels below are the same butterfly with those two parameters fixed.
enum {
    STEP = 256,     // shared-memory line length, and the split of the stages
    LINES = 8,      // lines a block carries in shared memory
    MAXROWS = 256   // largest number of lines in the view, i.e. N <= STEP*MAXROWS
};

__device__ inline void bfly_fwd(u64& u, u64& v, u64 tw, u64 p, u64 pinv) {
    const u64 x = u, y = v;
    u = mod_add(x, y, p);
    v = mont_mul(x >= y ? x - y : x + p - y, tw, pinv, p);
}

__device__ inline void bfly_inv(u64& u, u64& v, u64 tw, u64 p, u64 pinv) {
    const u64 x = u, y = mont_mul(v, tw, pinv, p);
    u = mod_add(x, y, p);
    v = x >= y ? x - y : x + p - y;
}

// Forward stages that stay inside a column: half-lengths rows/2 .. 1, column c.
// Both operands, because the pointwise product comes after this pass.
__global__ void k_cols_fwd(const Params p, u64* __restrict__ a, u64* __restrict__ b) {
    const u32 m = blockIdx.y, pair = blockIdx.z;
    const u32 rows = p.N / STEP;
    const u32 col0 = blockIdx.x * LINES;
    const u64 pr = p.primes[m], pv = p.pinv[m];
    const u64* roots = p.root_fwd + static_cast<u64>(m) * p.N;
    const u64 base = (static_cast<u64>(m) * p.B + pair) * p.N;

    __shared__ u64 sa[LINES][MAXROWS];
    __shared__ u64 sb[LINES][MAXROWS];

    for (u32 t = threadIdx.x; t < LINES * rows; t += blockDim.x) {
        const u32 ci = t % LINES, rr = t / LINES;
        const u64 g = base + static_cast<u64>(rr) * STEP + col0 + ci;
        sa[ci][rr] = a[g];
        sb[ci][rr] = b[g];
    }
    __syncthreads();

    const u32 half = rows / 2;
    for (u32 h = rows >> 1; h; h >>= 1) {
        const u32 len = h * STEP;
        for (u32 bf = threadIdx.x; bf < LINES * half; bf += blockDim.x) {
            const u32 ci = bf / half;
            const u32 x = bf - ci * half;
            const u32 g = x / h, j = x - g * h;
            const u32 k = g * 2 * h + j;
            const u64 tw = roots[len + static_cast<u64>(j) * STEP + col0 + ci];
            bfly_fwd(sa[ci][k], sa[ci][k + h], tw, pr, pv);
            bfly_fwd(sb[ci][k], sb[ci][k + h], tw, pr, pv);
        }
        __syncthreads();
    }

    for (u32 t = threadIdx.x; t < LINES * rows; t += blockDim.x) {
        const u32 ci = t % LINES, rr = t / LINES;
        const u64 g = base + static_cast<u64>(rr) * STEP + col0 + ci;
        a[g] = sa[ci][rr];
        b[g] = sb[ci][rr];
    }
}

// Forward stages that stay inside a row, then the pointwise product. `b` is not
// written back: its residues are dead once the product exists.
__global__ void k_rows_fwd_dot(const Params p, u64* __restrict__ a, u64* __restrict__ b) {
    const u32 m = blockIdx.y, pair = blockIdx.z;
    const u32 rows = p.N / STEP;
    const u32 line0 = blockIdx.x * LINES;
    const u32 here = rows - line0 < static_cast<u32>(LINES) ? rows - line0 : static_cast<u32>(LINES);
    const u64 pr = p.primes[m], pv = p.pinv[m];
    const u64* roots = p.root_fwd + static_cast<u64>(m) * p.N;
    const u64 base = (static_cast<u64>(m) * p.B + pair) * p.N;

    __shared__ u64 sa[LINES][MAXROWS];
    __shared__ u64 sb[LINES][MAXROWS];

    for (u32 t = threadIdx.x; t < here * STEP; t += blockDim.x) {
        const u32 li = t / STEP, k = t - li * STEP;
        const u64 g = base + static_cast<u64>(line0 + li) * STEP + k;
        sa[li][k] = a[g];
        sb[li][k] = b[g];
    }
    __syncthreads();

    for (u32 h = STEP / 2; h; h >>= 1) {
        for (u32 bf = threadIdx.x; bf < here * (STEP / 2); bf += blockDim.x) {
            const u32 li = bf / (STEP / 2);
            const u32 x = bf - li * (STEP / 2);
            const u32 g = x / h, j = x - g * h;
            const u32 k = g * 2 * h + j;
            const u64 tw = roots[h + j];
            bfly_fwd(sa[li][k], sa[li][k + h], tw, pr, pv);
            bfly_fwd(sb[li][k], sb[li][k + h], tw, pr, pv);
        }
        __syncthreads();
    }

    for (u32 t = threadIdx.x; t < here * STEP; t += blockDim.x) {
        const u32 li = t / STEP, k = t - li * STEP;
        const u64 g = base + static_cast<u64>(line0 + li) * STEP + k;
        a[g] = mont_mul(sa[li][k], sb[li][k], pv, pr);
    }
}

// Inverse stages that stay inside a row: half-lengths 1 .. STEP/2.
__global__ void k_rows_inv(const Params p, u64* __restrict__ a) {
    const u32 m = blockIdx.y, pair = blockIdx.z;
    const u32 rows = p.N / STEP;
    const u32 line0 = blockIdx.x * LINES;
    const u32 here = rows - line0 < static_cast<u32>(LINES) ? rows - line0 : static_cast<u32>(LINES);
    const u64 pr = p.primes[m], pv = p.pinv[m];
    const u64* roots = p.root_inv + static_cast<u64>(m) * p.N;
    const u64 base = (static_cast<u64>(m) * p.B + pair) * p.N;

    __shared__ u64 sa[LINES][MAXROWS];

    for (u32 t = threadIdx.x; t < here * STEP; t += blockDim.x) {
        const u32 li = t / STEP, k = t - li * STEP;
        sa[li][k] = a[base + static_cast<u64>(line0 + li) * STEP + k];
    }
    __syncthreads();

    for (u32 h = 1; h < STEP; h <<= 1) {
        for (u32 bf = threadIdx.x; bf < here * (STEP / 2); bf += blockDim.x) {
            const u32 li = bf / (STEP / 2);
            const u32 x = bf - li * (STEP / 2);
            const u32 g = x / h, j = x - g * h;
            const u32 k = g * 2 * h + j;
            bfly_inv(sa[li][k], sa[li][k + h], roots[h + j], pr, pv);
        }
        __syncthreads();
    }

    for (u32 t = threadIdx.x; t < here * STEP; t += blockDim.x) {
        const u32 li = t / STEP, k = t - li * STEP;
        a[base + static_cast<u64>(line0 + li) * STEP + k] = sa[li][k];
    }
}

// Inverse stages that stay inside a column, then the untwist — twist_inv
// carries both psi^-i and N^-1, so it is the last thing that touches a residue.
__global__ void k_cols_inv_twist(const Params p, u64* __restrict__ a) {
    const u32 m = blockIdx.y, pair = blockIdx.z;
    const u32 rows = p.N / STEP;
    const u32 col0 = blockIdx.x * LINES;
    const u64 pr = p.primes[m], pv = p.pinv[m];
    const u64* roots = p.root_inv + static_cast<u64>(m) * p.N;
    const u64* ti = p.twist_inv + static_cast<u64>(m) * p.N;
    const u64 base = (static_cast<u64>(m) * p.B + pair) * p.N;

    __shared__ u64 sa[LINES][MAXROWS];

    for (u32 t = threadIdx.x; t < LINES * rows; t += blockDim.x) {
        const u32 ci = t % LINES, rr = t / LINES;
        sa[ci][rr] = a[base + static_cast<u64>(rr) * STEP + col0 + ci];
    }
    __syncthreads();

    const u32 half = rows / 2;
    for (u32 h = 1; h < rows; h <<= 1) {
        const u32 len = h * STEP;
        for (u32 bf = threadIdx.x; bf < LINES * half; bf += blockDim.x) {
            const u32 ci = bf / half;
            const u32 x = bf - ci * half;
            const u32 g = x / h, j = x - g * h;
            const u32 k = g * 2 * h + j;
            const u64 tw = roots[len + static_cast<u64>(j) * STEP + col0 + ci];
            bfly_inv(sa[ci][k], sa[ci][k + h], tw, pr, pv);
        }
        __syncthreads();
    }

    for (u32 t = threadIdx.x; t < LINES * rows; t += blockDim.x) {
        const u32 ci = t % LINES, rr = t / LINES;
        const u64 g = base + static_cast<u64>(rr) * STEP + col0 + ci;
        a[g] = mont_mul(sa[ci][rr], ti[static_cast<u64>(rr) * STEP + col0 + ci], pv, pr);
    }
}

// ---------------------------------------------------------------- kernel: rns
//
// One block covers TILE coefficients of one pair, and walks every prime, so the
// coefficient data is read from global memory once per tile instead of once per
// prime. blockIdx.x is the tile, blockIdx.y the pair. Layout out is
// [prime][pair][coefficient], which is what the next two kernels want to read
// consecutively.
__global__ void k_residues(const Params p, const u32* __restrict__ src,
                           u64* __restrict__ dst) {
    __shared__ u32 smem[TILE * MAXL];

    const u32 pair = blockIdx.y;
    const u32 base = blockIdx.x * TILE;
    const u32 room = p.N - base;
    const u32 cnt = room < static_cast<u32>(TILE) ? room : static_cast<u32>(TILE);
    const u32 L = p.L;

    for (u32 t = threadIdx.x; t < cnt * L; t += blockDim.x)
        smem[t] = src[(static_cast<u64>(pair) * p.N + base) * L + t];
    __syncthreads();

    // 1024 threads: 128 coefficient lanes x 8 prime lanes. Each thread owns one
    // (coefficient, prime) pair and walks only its limbs, so the serial work per
    // thread is 1/8 of the old 256-thread kernel while the staged coefficient
    // bytes are unchanged. blockDim.x must be 1024; the grid is unchanged.
    constexpr u32 PGROUPS = 8;
    if (blockDim.x < TILE * PGROUPS) {
        // One-thread-per-block host shim (gpucheck): keep the original serial
        // walk so the arithmetic stays checkable without a device.
        for (u32 m = 0; m < p.k; ++m) {
            const u64 pr = p.primes[m], pv = p.pinv[m];
            const u64* lw = p.limb_w + static_cast<u64>(m) * L;
            u64* out = dst + (static_cast<u64>(m) * p.B + pair) * p.N + base;
            for (u32 cc = 0; cc < cnt; ++cc) {
                const u32* coef = smem + cc * L;
                u64 r = 0;
                for (u32 l = 0; l < L; ++l)
                    r = mod_add(r, mont_mul(coef[l], lw[l], pv, pr), pr);
                out[cc] = mont_mul(r, p.twist_fwd[static_cast<u64>(m) * p.N + base + cc], pv, pr);
            }
        }
        return;
    }
    const u32 c = threadIdx.x % TILE;
    const u32 lane = threadIdx.x / TILE;
    if (c >= cnt) return;
    const u32* coef = smem + c * L;
    for (u32 m = lane; m < p.k; m += PGROUPS) {
        const u64 pr = p.primes[m], pv = p.pinv[m];
        const u64* lw = p.limb_w + static_cast<u64>(m) * L;
        u64 r = 0;
        for (u32 l = 0; l < L; ++l)
            r = mod_add(r, mont_mul(coef[l], lw[l], pv, pr), pr);
        u64* out = dst + (static_cast<u64>(m) * p.B + pair) * p.N + base;
        out[c] = mont_mul(r, p.twist_fwd[static_cast<u64>(m) * p.N + base + c], pv, pr);
    }
}

// ---------------------------------------------------------------- kernel: ntt
//
// One block per (pair, prime) carries both operands through the transform,
// multiplies pointwise, and untwists. Written back into `a`, so `a` becomes the
// product's residues and `b` is dead afterwards.
__global__ void k_ntt(const Params p, u64* __restrict__ a, u64* __restrict__ b) {
    const u32 m = blockIdx.x % p.k;
    const u32 pair = blockIdx.x / p.k;
    const u32 n = p.N;
    const u64 pr = p.primes[m], pv = p.pinv[m];

    u64* pa = a + (static_cast<u64>(m) * p.B + pair) * n;
    u64* pb = b + (static_cast<u64>(m) * p.B + pair) * n;
    const u64* rf = p.root_fwd + static_cast<u64>(m) * n;
    const u64* ri = p.root_inv + static_cast<u64>(m) * n;

    ntt_fwd(pa, n, pr, pv, rf);
    ntt_fwd(pb, n, pr, pv, rf);
    for (u32 i = threadIdx.x; i < n; i += blockDim.x)
        pa[i] = mont_mul(pa[i], pb[i], pv, pr);
    ntt_inv(pa, n, pr, pv, ri);
    const u64* ti = p.twist_inv + static_cast<u64>(m) * n;
    for (u32 i = threadIdx.x; i < n; i += blockDim.x)
        pa[i] = mont_mul(pa[i], ti[i], pv, pr);
}

// ----------------------------------------------------------------- kernel: crt
//
// One thread per coefficient, so the batch fills the machine: the Garner chain
// and the big integers are per-coefficient serial work, and the only way to hide
// them is to have many coefficients in flight. blockIdx.y is the pair,
// blockIdx.x the chunk of coefficients.
__global__ void k_crt(const Params p, const u64* __restrict__ res,
                      u32* __restrict__ out) {
    const u32 pair = blockIdx.y;
    const u32 n = p.N, L = p.L, k = p.k;
    const u32 np = p.np, nq = p.nq, nmu = p.nmu, mlimbs = p.m_limbs;
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    {
        u64 digit[MAXK];

        for (u32 m = 0; m < k; ++m) {
            const u64 pr = p.primes[m], pv = p.pinv[m], r2 = p.r2[m];
            const u64* invm = p.inv_p + m;  // inv_p[j*k + m]
            u64 t = res[(static_cast<u64>(m) * p.B + pair) * n + i];
            for (u32 j = 0; j < m; ++j) {
                u64 d = t + pr - digit[j];
                if (d >= pr) d -= pr;
                t = raw_mul(d, invm[static_cast<u64>(j) * k], pr, pv, r2);
            }
            digit[m] = t;
        }

        // The exact value, base 2^32.
        u32 X[MAXLIMB];
        u32 nx = 0;
        if (digit[k - 1]) {
            X[0] = static_cast<u32>(digit[k - 1]);
            nx = 1;
            const u32 hi = static_cast<u32>(digit[k - 1] >> 32);
            if (hi) X[nx++] = hi;
        }
        for (u32 m = k - 1; m-- > 0;)
            nx = big_mul_add(X, nx, p.primes[m], digit[m]);

        // Barrett: qhat = floor(X * mu / B^mlimbs), r = X - qhat*q.
        u32 wide[MAXLIMB * 2];
        const u32 nw = big_mul(X, nx, p.mu, nmu, wide);
        const u32 nq_hat = nw > mlimbs ? nw - mlimbs : 0;
        u32 t2[MAXLIMB];
        const u32 nt2 = big_mul(wide + mlimbs, nq_hat, p.qlimbs, L, t2);

        u32 red[MAXLIMB + 1];
        u32 nr = big_sub(X, nx, t2, nt2, red);
        while (big_cmp(red, nr, p.qlimbs, L) >= 0)
            nr = big_sub(red, nr, p.qlimbs, L, red);

        // The reconstruction is exact but signed: above P/2 the true value is
        // X - P, so reduce P mod q out of it.
        if (big_cmp(X, nx, p.P_half, np) > 0) {
            if (big_cmp(red, nr, p.P_mod_q, nq) >= 0)
                nr = big_sub(red, nr, p.P_mod_q, nq, red);
            else {
                const u32 ns = big_add(red, nr, p.qlimbs, L, red);
                nr = big_sub(red, ns, p.P_mod_q, nq, red);
            }
        }

        u32* dst = out + (static_cast<u64>(pair) * n + i) * L;
        for (u32 l = 0; l < L; ++l) dst[l] = l < nr ? red[l] : 0u;
    }
}

// ---------------------------------------------------------- kernel: crt, fast
//
// The weighted-sum reconstruction from Plan, one thread per coefficient. Two
// things make it fast and one makes it exact:
//
//   * the accumulator has a compile-time size and every index into it is a
//     constant after unrolling, so it lives in registers. A 30-limb array in
//     local memory is 32 transactions per warp access — that is not a
//     micro-optimisation, it is the difference between the CRT being free and
//     being the whole batch;
//   * the reduction is two single-limb Montgomery steps (R = 2^128, folded into
//     the weights at setup), so no Barrett divide and no 112-limb product;
//   * t = round(X/P) comes from the 192-bit fractional weights: exact to
//     2^-124 against a margin of (P - bound)/2P >= 2^-24, so the reconstruction
//     is exact for every input and not merely with high probability.
template <u32 NQ>
__device__ inline void mont_step(u64* x, const u64* q, u64 qneg_inv) {
    const u64 m = x[0] * qneg_inv;  // the limb that makes (x + m*q) end in zero
    u64 carry = 0;
#pragma unroll
    for (u32 l = 0; l < NQ; ++l) {
        const u64 lo = m * q[l];
        const u64 hi = __umul64hi(m, q[l]);
        const u64 s1 = x[l] + lo;
        const u64 s2 = s1 + carry;
        carry = hi + ((s1 < lo) ? 1ull : 0ull) + ((s2 < s1) ? 1ull : 0ull);
        if (l) x[l - 1] = s2;  // (x + m*q) >> 64, in place, one limb down
    }
    x[NQ - 1] = carry;
}

// S = sum_j res_j * crt_w[j], then the sign fix, then the two Montgomery steps.
template <u32 NQ>
__global__ void k_crt_w(const Params p, const u64* __restrict__ res,
                        u32* __restrict__ out) {
    const u32 pair = blockIdx.y;
    const u32 n = p.N, L = p.L, k = p.k;
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    __shared__ u64 sq[NQ], spq[NQ], sqs[NQ];
    for (u32 t = threadIdx.x; t < NQ; t += blockDim.x) {
        sq[t] = p.q64[t];
        spq[t] = p.pmod64[t];
        sqs[t] = p.qsh64[t];
    }
    __syncthreads();

    u64 S[NQ];
#pragma unroll
    for (u32 l = 0; l < NQ; ++l) S[l] = 0;
    u64 F[5];
#pragma unroll
    for (u32 l = 0; l < 5; ++l) F[l] = 0;

    const u64* rp = res + static_cast<u64>(pair) * n + i;
    const u64 stride = static_cast<u64>(p.B) * n;
    for (u32 j = 0; j < k; ++j) {
        const u64 a = rp[static_cast<u64>(j) * stride];
        const u64* w = p.crt_w + static_cast<u64>(j) * 64;
        u64 c = 0;
#pragma unroll
        for (u32 l = 0; l < NQ; ++l) {  // S += a * W'_j, loose carry
            const u64 lo = a * w[l];
            const u64 hi = __umul64hi(a, w[l]);
            const u64 s1 = S[l] + lo;
            const u64 s2 = s1 + c;
            S[l] = s2;
            c = hi + ((s1 < lo) ? 1ull : 0ull) + ((s2 < s1) ? 1ull : 0ull);
        }
        const u64* cf = p.crt_c + static_cast<u64>(j) * 3;
        u64 fc = 0;
#pragma unroll
        for (u32 l = 0; l < 3; ++l) {  // F += a * C_j, the 192-bit fraction
            const u64 lo = a * cf[l];
            const u64 hi = __umul64hi(a, cf[l]);
            const u64 s1 = F[l] + lo;
            const u64 s2 = s1 + fc;
            F[l] = s2;
            fc = hi + ((s1 < lo) ? 1ull : 0ull) + ((s2 < s1) ? 1ull : 0ull);
        }
        const u64 f3 = F[3] + fc;  // F < 2^260, so F[4] only takes the carry
        F[4] += (f3 < F[3]) ? 1ull : 0ull;
        F[3] = f3;
    }

    // t = round(F / 2^192) = round(X/P): two limbs, at most 2^68.
    const u64 t0 = F[3] + ((F[2] >> 63) & 1ull);
    const u64 t1 = F[4] + ((t0 < F[3]) ? 1ull : 0ull);

    // T = S + q*2^68 - t*(P mod q)*2^128: congruent to C*2^128 mod q, positive
    // because q*2^68 > t*(P mod q)*2^128, and below q*2^128 for the reduction.
    u64 cn = 0;
#pragma unroll
    for (u32 l = 0; l < NQ; ++l) {
        const u64 s1 = S[l] + sqs[l];
        const u64 s2 = s1 + cn;
        cn = ((s1 < S[l]) ? 1ull : 0ull) + ((s2 < s1) ? 1ull : 0ull);
        S[l] = s2;
    }
    u64 pc = 0, borrow = 0;
#pragma unroll
    for (u32 l = 0; l < NQ; ++l) {
        u64 lo = t0 * spq[l];
        u64 hi = __umul64hi(t0, spq[l]);
        if (l) {
            const u64 s1 = lo + t1 * spq[l - 1];
            hi += __umul64hi(t1, spq[l - 1]) + ((s1 < lo) ? 1ull : 0ull);
            lo = s1;
        }
        const u64 s2 = lo + pc;
        pc = hi + ((s2 < lo) ? 1ull : 0ull);
        const u64 d = S[l] - s2;
        const u64 e = d - borrow;
        borrow = ((S[l] < s2) ? 1ull : 0ull) + ((d < borrow) ? 1ull : 0ull);
        S[l] = e;
    }

    mont_step<NQ>(S, sq, p.qneg_inv);
    mont_step<NQ>(S, sq, p.qneg_inv);

    // Two steps leave the value below 3q; two conditional subtractions settle it.
    for (u32 pass = 0; pass < 2; ++pass) {
        u64 br = 0;
#pragma unroll
        for (u32 l = 0; l < NQ; ++l) {
            const u64 d = S[l] - sq[l];
            br = ((S[l] < sq[l]) ? 1ull : 0ull) + ((d < br) ? 1ull : 0ull);
        }
        if (br) break;
        u64 bw = 0;
#pragma unroll
        for (u32 l = 0; l < NQ; ++l) {
            const u64 d = S[l] - sq[l];
            const u64 b1 = (S[l] < sq[l]) ? 1ull : 0ull;
            const u64 e = d - bw;
            bw = b1 + ((d < bw) ? 1ull : 0ull);
            S[l] = e;
        }
    }

    u32* dst = out + (static_cast<u64>(pair) * n + i) * L;
#pragma unroll
    for (u32 l = 0; l < NQ; ++l) {
        if (2 * l < L) dst[2 * l] = static_cast<u32>(S[l]);
        if (2 * l + 1 < L) dst[2 * l + 1] = static_cast<u32>(S[l] >> 32);
    }
}

}  // namespace pgpu