p = 'gpu31.cuh'
s = open(p, encoding='utf-8').read()
anchor = '    // Read and print the stage times. Call after the stream has synchronised.\n    void report_profile() const {'
assert s.count(anchor) == 1
method = r'''    // P6 probe (bold H2-sparse): throughput of the 8-groups-per-mma sparse
    // radix-16 twiddle packing at the full NTT volume.  Init-time, untimed.
    void p6_probe() {
#if P6_PROBE
        if (!tc_r8f) return;
        const unsigned B = capacity_ < 128 ? capacity_ : 128;
        const unsigned n = plan_.N, K = plan_.K;
        const unsigned long long batches = (unsigned long long)B * 3u * K * n / 128u;  // 8 groups = 128 elements
        const unsigned grid_ctas = 4096;
        const unsigned reps = (unsigned)((batches + grid_ctas - 1ull) / grid_ctas);
        u32 *d_out = nullptr, *d_vin = nullptr, *d_btab = nullptr;
        if (cudaMalloc(&d_out, std::size_t(grid_ctas) * 256 * sizeof(u32)) != cudaSuccess ||
            cudaMalloc(&d_vin, 64 * sizeof(u32)) != cudaSuccess ||
            cudaMalloc(&d_btab, 32 * sizeof(u32)) != cudaSuccess) {
            cudaGetLastError();
            if (d_out) cudaFree(d_out);
            if (d_vin) cudaFree(d_vin);
            if (d_btab) cudaFree(d_btab);
            std::fprintf(stderr, "[P6] alloc failed\n");
            return;
        }
        std::vector<u32> hv(64), hb(32);
        for (unsigned i = 0; i < 64; ++i) hv[i] = 0x11223344u + i * 0x01010101u;
        const u32 p = plan_.primes[0].p;
        for (unsigned i = 0; i < 8; ++i) {
            const u64 w = (1ull << (i + 1)) % p;
            // hb[i*4+s] holds byte_s(W_j) for j=0..3
            for (unsigned s = 0; s < 4; ++s) {
                u32 packed = 0;
                for (unsigned j = 0; j < 4; ++j) {
                    const u64 Wj = (w << (8ull * j)) % p;
                    packed |= u32((Wj >> (8 * s)) & 0xffu) << (8 * j);
                }
                hb[i * 4 + s] = packed;
            }
        }
        cudaMemcpy(d_vin, hv.data(), hv.size() * sizeof(u32), cudaMemcpyHostToDevice);
        cudaMemcpy(d_btab, hb.data(), hb.size() * sizeof(u32), cudaMemcpyHostToDevice);
        cudaStream_t stm = cs_[0] ? cs_[0] : 0;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e9f;
        for (int r = 0; r < 3; ++r) {
            cudaEventRecord(e0, stm);
            tc::p6_sparse_probe<<<grid_ctas, 256, 0, stm>>>(d_out, d_vin, d_btab, reps);
            cudaEventRecord(e1, stm);
            cudaEventSynchronize(e1);
            float t = 0.f; cudaEventElapsedTime(&t, e0, e1);
            if (t < best) best = t;
        }
        std::fprintf(stderr,
            "[P6] sparse mma: batches=%llu grid=%u reps=%u best=%.3f ms "
            "(mma called %llu times; 4 planes x 8 groups per mma)\n",
            batches, grid_ctas, reps, best, batches * 4ull);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(d_out); cudaFree(d_vin); cudaFree(d_btab);
#endif
    }

'''
s = s.replace(anchor, method + anchor)
open(p, 'w', encoding='utf-8').write(s)
print('gpu31.cuh: p6_probe added')

p = 'solve.cu'
s = open(p, encoding='utf-8').read()
anchor = '                    if (d.tc_ntt && d.tc_r8f) d.p4_probe();   // P4: untimed IMMA throughput probe (bold H2)\n'
if anchor not in s:
    anchor = '                    std::fprintf(stderr, "[TC] selfcheck enc=%d crt=%d fuse_decode=%d ntt=%d efl=%d chunk=%d\\n",\n                                 int(d.tc_enc), int(d.tc_crt), int(d.tc_fuse_decode), int(d.tc_ntt),\n                                 int(d.tc_efl), int(d.tc_chunk));\n'
    assert s.count(anchor) == 1
    s = s.replace(anchor, anchor + '                    if (d.tc_ntt && d.tc_r8f) d.p6_probe();   // P6: untimed sparse IMMA probe (bold H2-sparse)\n')
else:
    s = s.replace(anchor, anchor + '                    if (d.tc_ntt && d.tc_r8f) d.p6_probe();   // P6: untimed sparse IMMA probe (bold H2-sparse)\n')
open(p, 'w', encoding='utf-8').write(s)
print('solve.cu: p6_probe call added')
