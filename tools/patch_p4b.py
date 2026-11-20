p = 'gpu31.cuh'
s = open(p, encoding='utf-8').read()

# insert probe method just before report_profile()
anchor = '    // Read and print the stage times. Call after the stream has synchronised.\n    void report_profile() const {'
assert s.count(anchor) == 1
method = r'''    // P4 probe (bold H2): raw u8 IMMA throughput at the full NTT volume.
    // Called at init (untimed).  Extrapolates the time needed for radix-16
    // DFTs as 16x16 byte-split GEMMs: 4 passes, 4 byte-planes, 2 n-halves.
    void p4_probe() {
#if P4_PROBE
        if (!tc_r8f) return;
        const unsigned B = capacity_ < 128 ? capacity_ : 128;
        const unsigned n = plan_.N, K = plan_.K;
        const unsigned long long dfts = (unsigned long long)B * 3u * K * n / 16u;
        const unsigned long long need_mma = dfts * 4ull * 4ull * 2ull;
        const unsigned gx = 16, gy = K, gz = B;
        const unsigned warps = gx * gy * gz * 8u;
        const unsigned reps = (unsigned)((need_mma + warps - 1ull) / warps);
        const std::size_t out_words = std::size_t(gx) * gy * gz * 256;
        const unsigned fwords = 1u << 16;
        u32* d_out = nullptr;
        uint2* d_frag = nullptr;
        if (cudaMalloc(&d_out, out_words * sizeof(u32)) != cudaSuccess ||
            cudaMalloc(&d_frag, std::size_t(fwords) * sizeof(uint2)) != cudaSuccess) {
            cudaGetLastError();
            if (d_out) cudaFree(d_out);
            if (d_frag) cudaFree(d_frag);
            std::fprintf(stderr, "[P4] alloc failed\n");
            return;
        }
        cudaMemset(d_frag, 0x37, std::size_t(fwords) * sizeof(uint2));
        cudaStream_t stm = cs_[0] ? cs_[0] : 0;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e9f;
        for (int r = 0; r < 3; ++r) {
            cudaEventRecord(e0, stm);
            tc::p4_mma_probe<<<dim3(gx, gy, gz), 256, 0, stm>>>(d_out, d_frag, fwords, reps);
            cudaEventRecord(e1, stm);
            cudaEventSynchronize(e1);
            float t = 0.f; cudaEventElapsedTime(&t, e0, e1);
            if (t < best) best = t;
        }
        std::fprintf(stderr,
            "[P4] mma-only: reps=%u warps=%u total_mma=%llu best=%.3f ms "
            "(B=%u K=%u dfts=%llu; 1.44e9 mma => %.2f ms if linear)\n",
            reps, warps, need_mma, best, B, K, dfts,
            best * 1440000000.0 / double(need_mma ? need_mma : 1ull));
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(d_out); cudaFree(d_frag);
#endif
    }

'''
s = s.replace(anchor, method + anchor)
open(p, 'w', encoding='utf-8').write(s)
print('gpu31.cuh patched: p4_probe')

p = 'solve.cu'
s = open(p, encoding='utf-8').read()
anchor = '                    std::fprintf(stderr, "[TC] selfcheck enc=%d crt=%d fuse_decode=%d ntt=%d efl=%d chunk=%d\\n",\n                                 int(d.tc_enc), int(d.tc_crt), int(d.tc_fuse_decode), int(d.tc_ntt),\n                                 int(d.tc_efl), int(d.tc_chunk));\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '                    if (d.tc_ntt && d.tc_r8f) d.p4_probe();   // P4: untimed IMMA throughput probe (bold H2)\n')
open(p, 'w', encoding='utf-8').write(s)
print('solve.cu patched: p4_probe call')
