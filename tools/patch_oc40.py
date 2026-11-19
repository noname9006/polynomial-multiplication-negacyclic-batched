import io

p = 'tc.cuh'
s = open(p, encoding='utf-8').read()

flag = '''// OC40: 64 rows/CTA CRT GEMM.  One B fragment feeds 4 m-tiles (instead of
// 2), halving the ~28 GB L2 B-traffic.  Simple 2-thread carry (no seam
// split).  Dynamic shared ~63 KB needs cudaFuncSetAttribute.
#ifndef OC40_R64
#define OC40_R64 0
#endif
'''.replace('\n', '\n')
anchor = '// OC26: parallel carry propagation helper (device-only).'
assert s.count(anchor) == 1
s = s.replace(anchor, flag + anchor)

kernel = r'''
#if OC40_R64
constexpr unsigned OC40_ROWS = 64;
__global__ void __launch_bounds__(128, 2)
crt_tc2_oc40(gpu31::CrtArgs pl, const u32* __restrict__ residues,
             u32* __restrict__ out, const uint2* __restrict__ frag, CrtTc cfg,
             const gpu31::Prime* __restrict__ primes,
             const u32* __restrict__ untw_inv, unsigned slot_logN,
             const u32* __restrict__ corr_m, unsigned ntiles) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    u32* sA = reinterpret_cast<u32*>(smem_raw);
    u64* sX = reinterpret_cast<u64*>(smem_raw +
        ((std::size_t(OC40_ROWS) * cfg.SA * 4 + 15) & ~std::size_t(15)));
    constexpr unsigned NG = 128 / OC40_ROWS;   // 2
    const unsigned n = pl.n, l = pl.limbs, K = pl.count;
    const unsigned pair = blockIdx.z, start = blockIdx.x * OC40_ROWS;
    const unsigned tid = threadIdx.x;
    const unsigned c = tid & (OC40_ROWS - 1), jg = tid / OC40_ROWS;
    const unsigned i = start + c;
    const bool active = i < n;
    const u64 rbase2 = (u64(pair) * ntiles + (start >> 9)) * K;
    auto RC = [&](unsigned j, unsigned coeff) -> u32 {
        return residues[(rbase2 + j) * 512u + (coeff & 511u)];
    };
    const unsigned W = cfg.KS * 8;

    TCP_T(t0);
    u64 fsum = 0;
#pragma unroll 1
    for (unsigned j0 = jg; j0 < K; j0 += 8 * NG) {
        u32 r[8];
#pragma unroll
        for (unsigned q = 0; q < 8; ++q) {
            const unsigned j = j0 + NG * q;
            r[q] = (j < K && active) ? RC(j, i) : 0u;
        }
#pragma unroll
        for (unsigned q = 0; q < 8; ++q) {
            const unsigned j = j0 + NG * q;
            if (j < K) {
                const u32 term = (OC32_PREWEIGHT && TC22_MERGED && slot_logN)
                    ? r[q]
                    : gpu31::mont(r[q], (TC22_MERGED && slot_logN) ? c_crt_slotc[j] : 0u,
                                  c_crt_prime[j]);
                sA[c * cfg.SA + j] = term;
                fsum += (u64(term) * c_crt_corr[j]) >> 30;
            }
        }
    }
    for (unsigned j = K + 1 + jg; j < W; j += NG) sA[c * cfg.SA + j] = 0;
    sA[c * cfg.SA + W + 2 * jg] = u32(fsum);
    sA[c * cfg.SA + W + 2 * jg + 1] = u32(fsum >> 32);
    __syncthreads();
    if (jg == 0) {
        const u32* rw = sA + c * cfg.SA + W;
        u64 s = 0;
#pragma unroll
        for (unsigned q = 0; q < NG; ++q)
            s += u64(rw[2 * q]) | (u64(rw[2 * q + 1]) << 32);
        sA[c * cfg.SA + K] = u32((s + (u64(1) << 31)) >> 32);
    }
    __syncthreads();

    TCP_T(t1);
    const unsigned warp = tid >> 5, lane = tid & 31;
    const unsigned g = lane >> 2, t = lane & 3;
    const bool odd = t & 1;
    const unsigned sh = (t & 1) * 16;
    TCP_T(t2);
    for (unsigned nt0 = warp; nt0 < cfg.NT; nt0 += 4) {
        int acc[4][2][4];
#pragma unroll
        for (unsigned mt = 0; mt < 4; ++mt)
#pragma unroll
            for (unsigned q = 0; q < 2; ++q)
                acc[mt][q][0] = acc[mt][q][1] = acc[mt][q][2] = acc[mt][q][3] = 0;
#pragma unroll 3
        for (unsigned ks = 0; ks < cfg.KS; ++ks) {
            const uint2 b0 = frag[(std::size_t(nt0) * cfg.KS + ks) * 32 + lane];
#pragma unroll
            for (unsigned mt = 0; mt < 4; ++mt) {
                const u32* r0 = sA + (mt * 16 + g) * cfg.SA + ks * 8 + t;
                const u32* r1 = r0 + 8 * cfg.SA;
                const u32 x0 = r0[0], x1 = r1[0], x2 = r0[4], x3 = r1[4];
                mma_u8(acc[mt][0], x0, x1, x2, x3, b0.x, b0.y);
            }
        }
#pragma unroll
        for (unsigned mt = 0; mt < 4; ++mt) {
            const u64 pg = (u64(u32(acc[mt][0][0])) << sh) +
                           (u64(u32(acc[mt][0][1])) << (sh + 8));
            const u64 pg8 = (u64(u32(acc[mt][0][2])) << sh) +
                            (u64(u32(acc[mt][0][3])) << (sh + 8));
            const u64 send = odd ? pg : pg8;
            const u64 mine = odd ? pg8 : pg;
            const u64 tot = mine + __shfl_xor_sync(0xffffffffu, send, 1);
            const unsigned row = mt * 16 + g + (odd ? 8 : 0);
            sX[row * cfg.SX + nt0 * 2 + (t >> 1)] = tot;
        }
    }
    __syncthreads();

    TCP_T(t3);
    const unsigned DS = 2 * cfg.SX;
    u32* digits = reinterpret_cast<u32*>(sA);
    u64* cout = reinterpret_cast<u64*>(digits + OC40_ROWS * DS);
    const unsigned seg = tid & 1, row = tid >> 1;
    const unsigned words = cfg.NT * 2;
    const unsigned halfw = words >> 1;
    const u64* xs = sX + row * cfg.SX;
    u32* dg = digits + row * DS;
    u64 run = 0, big = 0;
    if (row < OC40_ROWS) {
#pragma unroll
        for (unsigned j = 0; j < halfw; ++j) {
            const u64 x = xs[seg * halfw + j];
            const u64 tt = run + x;
            dg[seg * halfw + j] = u32(tt);
            run = tt >> 32;
            big |= x >> 48;
        }
    }
    cout[row * 2 + seg] = run | (big ? (u64(1) << 63) : u64(0));
    __syncthreads();
    if (row < OC40_ROWS && seg == 0) {
        u64 cin = 0;
        bool replay = (cout[row * 2] >> 63) != 0 || (cout[row * 2 + 1] >> 63) != 0;
        if (replay) {
            u64 carry = 0;
            for (unsigned w = 0; w < words; ++w) {
                carry += xs[w];
                dg[w] = u32(carry);
                carry >>= 32;
            }
            cin = carry;
        } else {
            u64 delta = cout[row * 2];
            for (unsigned j = 0; j < halfw && delta; ++j) {
                const u64 sum = u64(dg[halfw + j]) + u32(delta);
                dg[halfw + j] = u32(sum);
                delta = (delta >> 32) + (sum >> 32);
            }
            cin = cout[row * 2 + 1] + delta;
        }
        for (unsigned w = words; w < l + 2; ++w) {
            dg[w] = u32(cin);
            cin >>= 32;
        }
        gpu31::reduce_sum<1>(dg, l, pl.bits, pl.gap, pl.q);
    }
    __syncthreads();
    TCP_T(t4);
    const unsigned rows = (n - start < OC40_ROWS) ? (n - start) : OC40_ROWS;
    const u64 obase = u64(pair) * n * l;
    for (unsigned idx = tid; idx < rows * l; idx += 128) {
        const unsigned r = idx / l, cc = idx - r * l;
        const unsigned pos = __brev(start + r) >> (32 - slot_logN);
        out[obase + u64(pos) * l + cc] = digits[r * DS + cc];
    }
    TCP_T(t5);
    TCP_ADD(0, t0, t1); TCP_ADD(1, t1, t2); TCP_ADD(2, t2, t3);
    TCP_ADD(3, t3, t4); TCP_ADD(4, t4, t5);
}
#endif  // OC40_R64

'''
anchor2 = '// inverse DIF low stages in 512-tiles (output: bit-reversed slot order).'
assert s.count(anchor2) == 1
s = s.replace(anchor2, kernel + anchor2)
open(p, 'w', encoding='utf-8').write(s)
print('patched tc.cuh with OC40 kernel')
