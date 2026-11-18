// Your answer. Three functions, and nothing to declare.
//
// Two paths, one answer:
//
//   * the device path (polymul_gpu.cuh), used when the machine has a GPU and
//     the kernels proved themselves against the host path during fherma_init;
//   * the portable path (polymul.hpp), which is always built and is what runs
//     when there is no device, when an allocation fails, or when the untimed
//     self-check disagrees with the host by one bit.
//
// That is deliberate. The device path is written on a machine with no GPU, so
// the thing that decides whether to trust it is the one run that can compare it
// to the reference on the machine that will actually measure it — and the
// reference is compiled into the same binary.
#include "fherma.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "polymul.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>

#include "polymul_gpu.cuh"
#if !defined(PGPU_HOST_CHECK)
#include "gpu31.cuh"
#include "ntt_fast.cuh"
#include "crt_mma.cuh"
#include "crt_mma_fast.cuh"
#endif
#endif

// A kernel launch, spelled once. Under nvcc this is the launch itself; under
// tools/hostcheck.ps1 — an ordinary compiler with the one-thread shim standing
// in for the device — it is a plain call, which is what lets the host half of
// this file be type-checked on a machine with no CUDA toolkit. Nothing else in
// the file changes between the two.
#if defined(PGPU_HOST_CHECK)
#define PG_LAUNCH(kernel, grid, block, shared, stream, ...) \
    ((void)(grid), (void)(block), (void)(shared), (void)(stream), kernel(__VA_ARGS__))
#else
#define PG_LAUNCH(kernel, grid, block, shared, stream, ...) \
    kernel<<<grid, block, shared, stream>>>(__VA_ARGS__)
#endif

namespace {

#ifdef __CUDACC__
using pgpu::u32;
using pgpu::u64;

// The kernels' types have to be the host's types, not lookalikes: upload_tables
// assigns one's vectors to the other's, and that is a different type on Linux
// (unsigned long vs unsigned long long) even though it is the same on Windows.
// This is the machine-independent way to say so.
static_assert(std::is_same<pgpu::u32, polymul::u32>::value &&
                  std::is_same<pgpu::u64, polymul::u64>::value,
              "pgpu and polymul must spell their integers the same way");

// Pairs carried through the pipeline in one pass. Everything for a pair is
// k*N*8 bytes per operand, so this is what keeps the device working set near L2
// instead of streaming 8 GB from HBM.
constexpr u32 kPairsPerPass = 128;

void* device_alloc(std::size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        cudaGetLastError();  // clear, the caller decides what to do
        return nullptr;
    }
    return p;
}

template <class T>
bool upload(T*& dst, const std::vector<T>& src) {
    if (src.empty()) return true;
    dst = static_cast<T*>(device_alloc(src.size() * sizeof(T)));
    if (!dst) return false;
    return cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

// Deterministic, so a disagreement is reproducible from the point alone.
struct Rng {
    u64 s;
    explicit Rng(u64 seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
    u64 next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};
#endif

#if !defined(PGPU_HOST_CHECK)
struct NttHookCtx { ntt_fast::NttFastPlan* plan = nullptr; gpu31::Device* dev = nullptr; };
#endif

struct State {
    polymul::Plan plan;
    std::unique_ptr<polymul::Engine> engine;

#ifdef __CUDACC__
    // Fast 31-bit RNS path; built and checked in fherma_init. When it is used,
    // the older 62-bit pgpu path stays compiled as a fallback but is untouched.
#if !defined(PGPU_HOST_CHECK)
    std::unique_ptr<gpu31::Device> gpu31_dev;
    std::unique_ptr<ntt_fast::NttFastPlan> ntt_fast_plan;
    NttHookCtx ntt_hook_ctx;
    std::unique_ptr<crt_mma::CrtMmaPlan> crt_mma_plan;
    std::unique_ptr<crt_mma_fast::CrtMmaFastPlan> crt_fast_plan;
    bool gpu31_ok = false;
    bool crt_mma_ok = false;  // fixed CRT proved identical to the reference CRT
    bool crt_fast_ok = false; // pre-baked tensor CRT proved identical to the reference
#endif

    bool gpu_ok = false;           // the device path is trusted for this point
    bool result_on_device = false; // run left the answer in d_c
    bool inputs_on_device = false; // h2d has been called since the last run
    cudaStream_t stream = nullptr;

    u64* d_primes = nullptr;
    u64* d_pinv = nullptr;
    u64* d_r2 = nullptr;
    u64* d_root_fwd = nullptr;
    u64* d_root_inv = nullptr;
    u64* d_twist_fwd = nullptr;
    u64* d_twist_inv = nullptr;
    u64* d_limb_w = nullptr;
    u64* d_inv_p = nullptr;
    u32* d_p_half = nullptr;
    u32* d_p_mod_q = nullptr;
    u32* d_q = nullptr;
    u32* d_mu = nullptr;
    u64* d_crt_w = nullptr;
    u64* d_crt_c = nullptr;
    u64* d_q64 = nullptr;
    u64* d_pmod64 = nullptr;
    u64* d_qsh64 = nullptr;

    u32* d_a = nullptr;
    u32* d_b = nullptr;
    u32* d_c = nullptr;
    u64* d_resa = nullptr;
    u64* d_resb = nullptr;
    pgpu::Params par{};
    std::size_t in_elems = 0;  // B*N*L u32 per operand

    // Pre-sized output buffers, filled once in fherma_init (untimed). Moving one
    // into the answer in fherma_run is O(1); allocating and zero-filling 1.84 GB
    // inside the timed call costs ~0.9 s in page faults on the runner.
    std::vector<std::vector<std::uint32_t>> out_pool;

    // Per-stage timing, reported through fherma_metrics. Cumulative over the
    // passes of one case; reset by the metrics call after each case.
    static constexpr u32 kMaxPasses = 8;  // B=128 / 16 pairs per pass
    cudaEvent_t ev_b[kMaxPasses] = {}, ev_r[kMaxPasses] = {},
                ev_n[kMaxPasses] = {}, ev_c[kMaxPasses] = {};
    float ms_res = 0.f, ms_ntt = 0.f, ms_crt = 0.f;
    bool timing_ok = false;

    void release() {
        void* all[] = {d_primes, d_pinv, d_r2, d_root_fwd, d_root_inv, d_twist_fwd,
                       d_twist_inv, d_limb_w, d_inv_p, d_p_half, d_p_mod_q, d_q, d_mu,
                       d_crt_w, d_crt_c, d_q64, d_pmod64, d_qsh64,
                       d_a, d_b, d_c, d_resa, d_resb};
        for (void* p : all)
            if (p) cudaFree(p);
        for (u32 i = 0; i < kMaxPasses; ++i) {
            if (ev_b[i]) cudaEventDestroy(ev_b[i]);
            if (ev_r[i]) cudaEventDestroy(ev_r[i]);
            if (ev_n[i]) cudaEventDestroy(ev_n[i]);
            if (ev_c[i]) cudaEventDestroy(ev_c[i]);
        }
        if (stream) cudaStreamDestroy(stream);
        d_primes = d_pinv = d_r2 = d_root_fwd = d_root_inv = nullptr;
        d_twist_fwd = d_twist_inv = d_limb_w = d_inv_p = nullptr;
        d_p_half = d_p_mod_q = d_q = d_mu = nullptr;
        d_crt_w = d_crt_c = d_q64 = d_pmod64 = d_qsh64 = nullptr;
        d_a = d_b = d_c = nullptr;
        d_resa = d_resb = nullptr;
        stream = nullptr;
        gpu_ok = false;
        timing_ok = false;
    }
#endif
};

}  // namespace

// ---------------------------------------------------------------------------
// device setup and the self-check
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
namespace {

// Flatten the plan in place; the params are only meaningful once every upload
// has landed, which is why this is separate from the uploads themselves.
bool upload_tables(State& s) {
    const polymul::Plan& plan = s.plan;
    const std::size_t k = plan.k(), N = plan.N, L = plan.L;

    std::vector<u64> primes(k), pinv(k), r2(k), root_fwd, root_inv, twist_fwd, twist_inv,
        limb_w, inv_p(k * k, 0);
    for (std::size_t m = 0; m < k; ++m) {
        primes[m] = plan.primes[m];
        pinv[m] = plan.mods[m].pinv;
        r2[m] = plan.mods[m].r2;
        root_fwd.insert(root_fwd.end(), plan.root_fwd[m].begin(), plan.root_fwd[m].end());
        root_inv.insert(root_inv.end(), plan.root_inv[m].begin(), plan.root_inv[m].end());
        twist_fwd.insert(twist_fwd.end(), plan.twist_fwd[m].begin(), plan.twist_fwd[m].end());
        twist_inv.insert(twist_inv.end(), plan.twist_inv[m].begin(), plan.twist_inv[m].end());
        limb_w.insert(limb_w.end(), plan.limb_w[m].begin(), plan.limb_w[m].end());
    }
    for (std::size_t m = 1; m < k; ++m)
        for (std::size_t j = 0; j < m; ++j) inv_p[j * k + m] = plan.inv_p[j][m];

    std::vector<u32> p_half(plan.P_half.limbs()), p_mod_q(plan.P_mod_q.limbs()), q(L),
        mu(plan.mu.limbs());
    plan.P_half.to_le(p_half.data(), p_half.size());
    plan.P_mod_q.to_le(p_mod_q.data(), p_mod_q.size());
    plan.q.to_le(q.data(), L);
    plan.mu.to_le(mu.data(), mu.size());

    if (!upload(s.d_primes, primes) || !upload(s.d_pinv, pinv) || !upload(s.d_r2, r2) ||
        !upload(s.d_root_fwd, root_fwd) || !upload(s.d_root_inv, root_inv) ||
        !upload(s.d_twist_fwd, twist_fwd) || !upload(s.d_twist_inv, twist_inv) ||
        !upload(s.d_limb_w, limb_w) || !upload(s.d_inv_p, inv_p) ||
        !upload(s.d_p_half, p_half) || !upload(s.d_p_mod_q, p_mod_q) ||
        !upload(s.d_q, q) || !upload(s.d_mu, mu))
        return false;

    // The weighted-sum CRT is optional: no table (q wider than the kernels are
    // built for) means the Garner kernel runs instead, which is slower and just
    // as exact.
    std::vector<u64> crt_w, crt_c, q64, pmod64, qsh64;
    if (plan.crt_nq) {
        for (std::size_t j = 0; j < k; ++j)
            crt_w.insert(crt_w.end(), plan.crt_w[j].begin(), plan.crt_w[j].end());
        for (std::size_t j = 0; j < k; ++j)
            crt_c.insert(crt_c.end(), plan.crt_c[j].begin(), plan.crt_c[j].end());
        q64 = plan.q64;
        pmod64 = plan.pmod64;
        qsh64 = plan.qsh64;
        if (!upload(s.d_crt_w, crt_w) || !upload(s.d_crt_c, crt_c) || !upload(s.d_q64, q64) ||
            !upload(s.d_pmod64, pmod64) || !upload(s.d_qsh64, qsh64))
            return false;
    }

    pgpu::Params& p = s.par;
    p.N = static_cast<u32>(N);
    p.L = static_cast<u32>(L);
    p.k = static_cast<u32>(k);
    p.B = 0;  // set per pass: the kernels index [prime][pair][i] with this stride
    p.primes = s.d_primes;
    p.pinv = s.d_pinv;
    p.r2 = s.d_r2;
    p.root_fwd = s.d_root_fwd;
    p.root_inv = s.d_root_inv;
    p.twist_fwd = s.d_twist_fwd;
    p.twist_inv = s.d_twist_inv;
    p.limb_w = s.d_limb_w;
    p.inv_p = s.d_inv_p;
    p.P_half = s.d_p_half;
    p.P_mod_q = s.d_p_mod_q;
    p.qlimbs = s.d_q;
    p.mu = s.d_mu;
    p.np = static_cast<u32>(p_half.size());
    p.nq = static_cast<u32>(p_mod_q.size());
    p.nmu = static_cast<u32>(mu.size());
    p.m_limbs = static_cast<u32>(plan.m_limbs);
    p.crt_nq = plan.crt_nq;
    p.crt_w = s.d_crt_w;
    p.crt_c = s.d_crt_c;
    p.q64 = s.d_q64;
    p.pmod64 = s.d_pmod64;
    p.qsh64 = s.d_qsh64;
    p.qneg_inv = plan.qneg_inv;
    return true;
}

// Whether this length takes the two-pass transform: N has to be a whole number
// of STEP-long lines, and MAXROWS of them at most, which is every point of this
// challenge. Anything else keeps the stage-at-a-time kernel.
bool two_pass(u32 N) {
    return N >= pgpu::STEP && N % pgpu::STEP == 0 && N / pgpu::STEP <= pgpu::MAXROWS;
}

// The whole device pipeline for `pairs` pairs already resident in d_a/d_b.
// Returns false if any launch or the synchronisation reported an error.
bool gpu_multiply(State& s, u32 pairs) {
    const u32 N = s.par.N, L = s.par.L, k = s.par.k;
    const u32 tiles = (N + pgpu::TILE - 1) / pgpu::TILE;
    const u32 block = 256;

    u32 pass_idx = 0;
    for (u32 t0 = 0; t0 < pairs; t0 += kPairsPerPass) {
        const u32 tb = (pairs - t0 < kPairsPerPass) ? pairs - t0 : kPairsPerPass;
        if (s.timing_ok && pass_idx < State::kMaxPasses)
            cudaEventRecord(s.ev_b[pass_idx], s.stream);
        pgpu::Params p = s.par;
        p.B = tb;

        const pgpu::u32* pa = s.d_a + static_cast<std::size_t>(t0) * N * L;
        const pgpu::u32* pb = s.d_b + static_cast<std::size_t>(t0) * N * L;
        pgpu::u32* pc = s.d_c + static_cast<std::size_t>(t0) * N * L;

        const dim3 grid_rns(tiles, tb);
        PG_LAUNCH(pgpu::k_residues, grid_rns, 1024, 0, s.stream, p, pa, s.d_resa);
        PG_LAUNCH(pgpu::k_residues, grid_rns, 1024, 0, s.stream, p, pb, s.d_resb);
        if (s.timing_ok && pass_idx < State::kMaxPasses)
            cudaEventRecord(s.ev_r[pass_idx], s.stream);

        if (two_pass(N)) {
            const u32 rows = N / pgpu::STEP;
            const dim3 grid_cols(pgpu::STEP / pgpu::LINES, k, tb);
            const dim3 grid_rows((rows + pgpu::LINES - 1) / pgpu::LINES, k, tb);
            PG_LAUNCH(pgpu::k_cols_fwd, grid_cols, block, 0, s.stream, p, s.d_resa, s.d_resb);
            PG_LAUNCH(pgpu::k_rows_fwd_dot, grid_rows, block, 0, s.stream, p, s.d_resa,
                      s.d_resb);
            PG_LAUNCH(pgpu::k_rows_inv, grid_rows, block, 0, s.stream, p, s.d_resa);
            PG_LAUNCH(pgpu::k_cols_inv_twist, grid_cols, block, 0, s.stream, p, s.d_resa);
        } else {
            PG_LAUNCH(pgpu::k_ntt, tb * k, block, 0, s.stream, p, s.d_resa, s.d_resb);
        }
        if (s.timing_ok && pass_idx < State::kMaxPasses)
            cudaEventRecord(s.ev_n[pass_idx], s.stream);

        const u32 crt_block = 256;
        const dim3 grid_crt((N + crt_block - 1) / crt_block, tb);
        switch (p.crt_nq) {
            case 4: PG_LAUNCH(pgpu::k_crt_w<4>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 6: PG_LAUNCH(pgpu::k_crt_w<6>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 8: PG_LAUNCH(pgpu::k_crt_w<8>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 12: PG_LAUNCH(pgpu::k_crt_w<12>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 16: PG_LAUNCH(pgpu::k_crt_w<16>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 24: PG_LAUNCH(pgpu::k_crt_w<24>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 32: PG_LAUNCH(pgpu::k_crt_w<32>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 48: PG_LAUNCH(pgpu::k_crt_w<48>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            case 64: PG_LAUNCH(pgpu::k_crt_w<64>, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
            default: PG_LAUNCH(pgpu::k_crt, grid_crt, crt_block, 0, s.stream, p, s.d_resa, pc); break;
        }
        if (s.timing_ok && pass_idx < State::kMaxPasses)
            cudaEventRecord(s.ev_c[pass_idx], s.stream);
        ++pass_idx;
    }

    if (cudaGetLastError() != cudaSuccess) return false;
    if (cudaStreamSynchronize(s.stream) != cudaSuccess) return false;

    if (s.timing_ok) {
        const u32 np = pass_idx < State::kMaxPasses ? pass_idx : State::kMaxPasses;
        for (u32 q = 0; q < np; ++q) {
            float t0 = 0.f, t1 = 0.f, t2 = 0.f;
            cudaEventElapsedTime(&t0, s.ev_b[q], s.ev_r[q]);
            cudaEventElapsedTime(&t1, s.ev_r[q], s.ev_n[q]);
            cudaEventElapsedTime(&t2, s.ev_n[q], s.ev_c[q]);
            s.ms_res += t0;
            s.ms_ntt += t1;
            s.ms_crt += t2;
        }
        std::fprintf(stderr,
                     "polymul stages (ms): residues %.3f  ntt %.3f  crt %.3f  total %.3f\n",
                     s.ms_res, s.ms_ntt, s.ms_crt, s.ms_res + s.ms_ntt + s.ms_crt);
    }
    return true;
}

// Two random pairs, both ways, compared coefficient by coefficient. Runs in
// fherma_init, which is not timed, and decides whether fherma_run may use the
// device at all. Two rather than one on purpose: with a single pair every
// [pair] index in the kernels is zero, which would hide a wrong pair stride.
// Nothing about the case data is involved: the pairs are generated here from the
// point's own constants.
bool gpu_selfcheck(State& s, u32 pairs) {
    const polymul::Plan& plan = s.plan;
    const std::size_t N = plan.N, L = plan.L, per = N * L;

    std::vector<u32> a(pairs * per), b(pairs * per), want(pairs * per, 0),
        got(pairs * per, 0);
    Rng rng(plan.primes[0] ^ (static_cast<u64>(N) << 32) ^ plan.q.limb(0));
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<u32>(rng.next());
        b[i] = static_cast<u32>(rng.next());
    }
    // Canonical inputs: the specification promises coefficients in [0, q).
    for (u32 i = 0; i < pairs * static_cast<u32>(N); ++i) {
        for (u32* c : {a.data() + static_cast<std::size_t>(i) * L,
                       b.data() + static_cast<std::size_t>(i) * L}) {
            polymul::Big x = polymul::Big::from_le(c, L);
            if (polymul::cmp(x, plan.q) < 0) continue;
            polymul::Big quot, rem;
            polymul::divmod(x, plan.q, quot, rem);
            rem.to_le(c, L);
        }
    }

    for (u32 j = 0; j < pairs; ++j)
        s.engine->multiply(a.data() + static_cast<std::size_t>(j) * per,
                           b.data() + static_cast<std::size_t>(j) * per,
                           want.data() + static_cast<std::size_t>(j) * per);

    if (cudaMemcpy(s.d_a, a.data(), a.size() * sizeof(u32), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(s.d_b, b.data(), b.size() * sizeof(u32), cudaMemcpyHostToDevice) !=
            cudaSuccess)
        return false;
    if (!gpu_multiply(s, pairs)) return false;
    if (cudaMemcpy(got.data(), s.d_c, got.size() * sizeof(u32), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        return false;

    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) ++bad;
    if (bad) {
        std::fprintf(stderr,
                     "polymul: device path disagrees with the host on %zu of %zu limbs; "
                     "using the host path\n",
                     bad, got.size());
        return false;
    }
    return true;
}

// Verify the 31-bit path against the portable host engine on deterministic,
// canonical random pairs. Same idea as gpu_selfcheck, but through gpu31::Device.
#if !defined(PGPU_HOST_CHECK)
bool gpu31_selfcheck(State& s, gpu31::Device& dev, u32 pairs) {
    const polymul::Plan& plan = s.plan;
    const std::size_t N = plan.N, L = plan.L, per = N * L;
    std::vector<u32> a(pairs * per), b(pairs * per), want(pairs * per, 0),
        got(pairs * per, 0);
    Rng rng(plan.primes[0] ^ (static_cast<u64>(N) << 31) ^ plan.q.limb(0) ^ 0xa5a5a5a5u);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<u32>(rng.next());
        b[i] = static_cast<u32>(rng.next());
    }
    for (u32 i = 0; i < pairs * static_cast<u32>(N); ++i) {
        for (u32* c : {a.data() + static_cast<std::size_t>(i) * L,
                       b.data() + static_cast<std::size_t>(i) * L}) {
            polymul::Big x = polymul::Big::from_le(c, L);
            if (polymul::cmp(x, plan.q) < 0) continue;
            polymul::Big quot, rem;
            polymul::divmod(x, plan.q, quot, rem);
            rem.to_le(c, L);
        }
    }
    for (u32 j = 0; j < pairs; ++j)
        s.engine->multiply(a.data() + static_cast<std::size_t>(j) * per,
                           b.data() + static_cast<std::size_t>(j) * per,
                           want.data() + static_cast<std::size_t>(j) * per);

    if (cudaMemcpy(s.d_a, a.data(), a.size() * sizeof(u32), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(s.d_b, b.data(), b.size() * sizeof(u32), cudaMemcpyHostToDevice) !=
            cudaSuccess)
        return false;
    if (dev.run(s.d_a, s.d_b, s.d_c, pairs, s.stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(s.stream) != cudaSuccess) return false;
    if (cudaMemcpy(got.data(), s.d_c, got.size() * sizeof(u32), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        return false;

    std::size_t bad = 0;
    for (std::size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) ++bad;
    if (bad) {
        std::fprintf(stderr,
                     "polymul: gpu31 disagrees with host on %zu of %zu limbs; keeping fallback\n",
                     bad, got.size());
        return false;
    }
    return true;
}

// NTT hook: dispatched from gpu31::Device::run for stages 3/5. The context is
// the NttFastPlan held in State; the bool selects forward vs inverse.
#if !defined(PGPU_HOST_CHECK)
static cudaError_t ntt_hook_fn(u32* a, u32* b, unsigned B, cudaStream_t stream,
                               bool fwd, void* ctx) {
    auto* h = static_cast<NttHookCtx*>(ctx);
    if (!h || !h->plan || !h->dev) return cudaErrorInvalidValue;
    return fwd ? h->plan->forward(a, b, *h->dev, B, stream)
               : h->plan->inverse(a, b, *h->dev, B, stream);
}
#endif

// Generic CRT-vs-reference check: runs `runner` on random residues and compares
// its output byte-for-byte against gpu31's reference CRT. Used for both the
// cuPQC and MMA candidates.
template <class Runner>
bool crt_vs_reference(State& s, Runner runner, u32 pair_count, const char* name) {
    const unsigned n = s.plan.N, k = s.gpu31_dev->plan().K;
    const u32 pairs = pair_count < 2 ? pair_count : 2;
    cudaStream_t stm = s.stream;
    const std::size_t nres = std::size_t(pairs) * k * n;
    std::vector<u32> host(nres);
    Rng rng(0x5eed1234ull ^ (u64(n) << 20) ^ k);
    for (auto& v : host) v = static_cast<u32>(rng.next()) & 0x7fffffffu;
    if (cudaMemcpy(s.d_a, host.data(), nres * sizeof(u32),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    u32* src = s.d_a;
    const auto& args = s.gpu31_dev->crt_args();
    if (s.gpu31_dev->crt_reference(src, s.d_c, pairs, stm) != cudaSuccess) return false;
    const cudaError_t e_run = runner(src, s.d_b, args, pairs, stm);
    if (e_run != cudaSuccess) {
        std::fprintf(stderr, "polymul: %s runner returned CUDA error %d\n", name, (int)e_run);
        return false;
    }
    if (cudaStreamSynchronize(stm) != cudaSuccess) return false;
    const std::size_t nout = std::size_t(pairs) * n * s.plan.L;
    std::vector<u32> want(nout), got(nout);
    if (cudaMemcpy(want.data(), s.d_c, nout * sizeof(u32), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(got.data(), s.d_b, nout * sizeof(u32), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        return false;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < nout; ++i)
        if (want[i] != got[i]) ++bad;
    if (bad) {
        std::fprintf(stderr,
                     "polymul: %s CRT disagrees with reference on %zu of %zu limbs; "
                     "keeping reference CRT\n",
                     name, bad, nout);
        return false;
    }
    return true;
}

// Check the tensor-core CRT against the reference CRT on random residue data,
// using the same device buffers. If even one limb differs, we keep the reference.
bool crt_mma_selfcheck(State& s, crt_mma::CrtMmaPlan& plan, u32 pair_count) {
    // Only 2 pairs: d_a/d_b are sized B*N*L u32, far more than 2*K*N residues.
    const unsigned n = s.plan.N, k = s.gpu31_dev->plan().K;
    const u32 pairs = pair_count < 2 ? pair_count : 2;
    cudaStream_t stm = s.stream;
    const std::size_t nres = std::size_t(pairs) * k * n;
    std::vector<u32> host(nres);
    Rng rng(0x5eed1234ull ^ (u64(n) << 20) ^ k);
    for (auto& v : host) v = static_cast<u32>(rng.next()) & 0x7fffffffu;
    if (cudaMemcpy(s.d_a, host.data(), nres * sizeof(u32),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    u32* src = s.d_a;
    const auto& args = s.gpu31_dev->crt_args();
    // reference output -> d_c, MMA output -> d_b
    if (s.gpu31_dev->crt_reference(src, s.d_c, pairs, stm) != cudaSuccess) return false;
    if (plan.run(src, s.d_b, args, pairs, stm) != cudaSuccess) return false;
    if (cudaStreamSynchronize(stm) != cudaSuccess) return false;
    const std::size_t nout = std::size_t(pairs) * n * s.plan.L;
    std::vector<u32> want(nout), got(nout);
    if (cudaMemcpy(want.data(), s.d_c, nout * sizeof(u32), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(got.data(), s.d_b, nout * sizeof(u32), cudaMemcpyDeviceToHost) !=
            cudaSuccess)
        return false;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < nout; ++i)
        if (want[i] != got[i]) ++bad;
    if (bad) {
        std::fprintf(stderr,
                     "polymul: MMA CRT disagrees with reference on %zu of %zu limbs; "
                     "keeping reference CRT\n",
                     bad, nout);
        return false;
    }
    return true;
}
#endif  // !PGPU_HOST_CHECK

}  // namespace
#endif  // __CUDACC__

// Setup. Once per point, and not measured. It sees only (B, N, W, L, q), so every
// parameter-dependent table is built here: primes, twiddles, CRT and Barrett
// constants, and — on a device — the same tables in device memory, plus the
// self-check that decides whether they are used.
void* fherma_init(const fherma::Point& p) {
    if (p.q.data.size() != p.L) throw std::runtime_error("point: q has the wrong limb count");
    auto state = std::make_unique<State>();
    state->plan = polymul::build_plan(p.N, p.W, p.L, p.q.data.data());

    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned threads = hw == 0 ? 1u : (hw > 64u ? 64u : hw);
    state->engine = std::make_unique<polymul::Engine>(state->plan, threads);

    // One output buffer per case (warm-up + measured seeds), sized and zeroed
    // here, where setup is free. fherma_run moves one out in O(1); allocating
    // and zero-filling 1.84 GB inside the timed call costs ~0.9 s in page faults.
    const std::size_t out_words = static_cast<std::size_t>(p.B) * state->plan.N * state->plan.L;
    state->out_pool.reserve(6);
    for (int i = 0; i < 6; ++i) state->out_pool.emplace_back(out_words, 0u);

#ifdef __CUDACC__
    const polymul::Plan& plan = state->plan;
    const std::size_t per = static_cast<std::size_t>(plan.N) * plan.L;
    state->in_elems = static_cast<std::size_t>(p.B) * per;

    bool fits = plan.N > 0 && p.B > 0 && plan.L <= pgpu::MAXL && plan.k() <= pgpu::MAXK &&
                plan.P.limbs() <= pgpu::MAXLIMB &&
                plan.P.limbs() + plan.mu.limbs() <= 2 * pgpu::MAXLIMB;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) fits = false;
    if (fits && cudaSetDevice(0) != cudaSuccess) fits = false;

    if (fits) {
        const u32 k = static_cast<u32>(plan.k()), N = plan.N;
        const u32 pass = p.B < kPairsPerPass ? p.B : kPairsPerPass;
        const std::size_t pairs_bytes = static_cast<std::size_t>(pass) * k * N * 8;

        if (cudaStreamCreate(&state->stream) != cudaSuccess) fits = false;
        if (fits) fits = upload_tables(*state);
        if (fits) state->d_a = static_cast<u32*>(device_alloc(state->in_elems * sizeof(u32)));
        if (fits) state->d_b = static_cast<u32*>(device_alloc(state->in_elems * sizeof(u32)));
        if (fits) state->d_c = static_cast<u32*>(device_alloc(state->in_elems * sizeof(u32)));
        if (fits) state->d_resa = static_cast<u64*>(device_alloc(pairs_bytes));
        if (fits) state->d_resb = static_cast<u64*>(device_alloc(pairs_bytes));
        if (fits && (!state->d_a || !state->d_b || !state->d_c || !state->d_resa || !state->d_resb))
            fits = false;
        if (fits) fits = gpu_selfcheck(*state, p.B < 2 ? p.B : 2);
        if (fits) {
            for (u32 i = 0; i < State::kMaxPasses; ++i) {
                if (cudaEventCreate(&state->ev_b[i]) != cudaSuccess ||
                    cudaEventCreate(&state->ev_r[i]) != cudaSuccess ||
                    cudaEventCreate(&state->ev_n[i]) != cudaSuccess ||
                    cudaEventCreate(&state->ev_c[i]) != cudaSuccess) {
                    fits = false;
                    break;
                }
            }
        }
        state->timing_ok = fits;
        if (!fits) state->release();
        state->gpu_ok = fits;
    }

    // Prefer the batched 31-bit RNS path when it proves itself against the host.
    // It reuses the device buffers and stream set up above; the 62-bit path stays
    // compiled and ready if construction or the self-check fails.
#if !defined(PGPU_HOST_CHECK)
    if (state->gpu_ok) {
        try {
            auto dev = std::make_unique<gpu31::Device>(p);
            if (gpu31_selfcheck(*state, *dev, p.B < 2 ? p.B : 2)) {
                state->gpu31_dev = std::move(dev);
                state->gpu31_ok = true;

                // Install the optimized NTT if it proves byte-identical via the
                // gpu31 self-check (we re-run it with the hook below).
                try {
                    state->ntt_fast_plan = std::make_unique<ntt_fast::NttFastPlan>(*state->gpu31_dev);
                    state->ntt_hook_ctx.plan = state->ntt_fast_plan.get();
                    state->ntt_hook_ctx.dev = state->gpu31_dev.get();
                    state->gpu31_dev->set_ntt_hook(ntt_hook_fn, &state->ntt_hook_ctx);
                    // Re-verify the whole pipeline with the optimized NTT. On any
                    // mismatch, drop the hook and keep the proven built-in NTT.
                    if (!gpu31_selfcheck(*state, *state->gpu31_dev, p.B < 2 ? p.B : 2)) {
                        state->gpu31_dev->set_ntt_hook(nullptr, nullptr);
                        state->ntt_fast_plan.reset();
                    }
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "polymul: NTT fast setup failed: %s\n", e.what());
                    state->gpu31_dev->set_ntt_hook(nullptr, nullptr);
                    state->ntt_fast_plan.reset();
                }

                // The 62-bit path's scratch buffers are now dead weight: gpu_multiply()
                // is their only consumer, and fherma_run reaches it only when gpu31_ok
                // is false (a failed copy/sync falls straight to the host path). Free
                // ~7.6 GB here. Leave the pointers nullptr so State::release(), which
                // frees each pointer only `if (p)`, cannot double-free them.
                cudaFree(state->d_resa);
                cudaFree(state->d_resb);
                state->d_resa = nullptr;
                state->d_resb = nullptr;

                // Tensor-core encode and CRT (tc.cuh): each is kept only if the
                // whole pipeline still matches the host answer exactly.
                {
                    gpu31::Device& d = *state->gpu31_dev;
                    if (d.tc_enc_ready) {
                        d.tc_enc = true;
                        if (!gpu31_selfcheck(*state, d, p.B < 2 ? p.B : 2)) d.tc_enc = false;
                    }
                    if (d.tc_crt_ready) {
                        d.tc_crt = true;
                        if (!gpu31_selfcheck(*state, d, p.B < 2 ? p.B : 2)) d.tc_crt = false;
                    }
                    if (d.tc_crt && d.d_untw_inv) {
                        d.tc_fuse_decode = true;
                        if (!gpu31_selfcheck(*state, d, p.B < 2 ? p.B : 2)) d.tc_fuse_decode = false;
                    }
                    std::fprintf(stderr, "[TC] selfcheck enc=%d crt=%d fuse_decode=%d
",
                                 int(d.tc_enc), int(d.tc_crt), int(d.tc_fuse_decode));
                }

                // Try the conflict-free CRT, keeping it only if bit-identical to
                // the reference CRT on deterministic residue data.
                const auto& pl = state->gpu31_dev->plan();
                if (!state->gpu31_dev->tc_crt) try {
                    state->crt_mma_plan = std::make_unique<crt_mma::CrtMmaPlan>(
                        pl.crt_weights, pl.L, pl.K + 1);
                    crt_mma::CrtMmaPlan* p_plan = state->crt_mma_plan.get();
                    state->crt_mma_ok = crt_vs_reference(
                        *state,
                        [p_plan](const u32* r, u32* o, const auto& a, u32 b, cudaStream_t s) {
                            return p_plan->run(r, o, a, b, s);
                        },
                        2, "CRT");
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "polymul: CRT setup failed: %s\n", e.what());
                }

                if (false) // H15 CRT-fast: measured slower (56 ms); disabled
                // H15: try the pre-baked tensor-core CRT.  It replaces the
                // ~121 scalar packing instructions per MMA of crt_mma_diag
                // with 6 loads, by baking the B operand on the host and the
                // A byte-planes in shared.  Keep it only if byte-identical.
                try {
                    state->crt_fast_plan = std::make_unique<crt_mma_fast::CrtMmaFastPlan>(
                        pl.crt_weights, pl.L, pl.K + 1);
                    auto* pf = state->crt_fast_plan.get();
                    std::fprintf(stderr, "[CRTF] plan L=%u pitch=%u smem=%zu\n",
                                 pf->L, pf->pitch, crt_mma_fast::shared_bytes(pf->L, pf->pitch));
                    state->crt_fast_ok = crt_vs_reference(
                        *state,
                        [pf](const u32* r, u32* o, const auto& a, u32 b, cudaStream_t s) {
                            return pf->run(r, o, a, b, s);
                        },
                        2, "CRT-fast");
                    std::fprintf(stderr, "[CRTF] selfcheck=%d\n", (int)state->crt_fast_ok);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "polymul: CRT-fast setup failed: %s\n", e.what());
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "polymul: gpu31 setup failed: %s\n", e.what());
        }
    }
#endif
#endif

    return state.release();
}

// The answer, and the only thing measured.
//
//   in.a — tensor<B x N x L x u32>, coefficients in [0, q), little-endian limbs
//   in.b — same
//   out.c — same layout, the negacyclic product
fherma::Outputs fherma_run(void* state, const fherma::Inputs& in) {
    auto* s = static_cast<State*>(state);
    const polymul::Plan& plan = s->plan;
    s->ms_res = s->ms_ntt = s->ms_crt = 0.f;

    if (in.a.shape.size() != 3 || in.b.shape != in.a.shape)
        throw std::runtime_error("run: expected a and b of shape {B, N, L}");
    if (static_cast<std::uint32_t>(in.a.shape[1]) != plan.N ||
        static_cast<std::uint32_t>(in.a.shape[2]) != plan.L)
        throw std::runtime_error("run: the tensor shape does not match the point");

    const std::int64_t B = in.a.shape[0];
    const std::size_t per = static_cast<std::size_t>(plan.N) * plan.L;
    if (static_cast<std::size_t>(in.a.data.size()) != static_cast<std::size_t>(B) * per)
        throw std::runtime_error("run: a has the wrong number of elements");
    if (static_cast<std::size_t>(in.b.data.size()) != static_cast<std::size_t>(B) * per)
        throw std::runtime_error("run: b has the wrong number of elements");

    fherma::Outputs out;
    out.c.shape = in.a.shape;

#ifdef __CUDACC__
    s->result_on_device = false;
    if (s->gpu_ok && B > 0 && static_cast<std::size_t>(B) * per <= s->in_elems) {
        const std::size_t bytes = static_cast<std::size_t>(B) * per * sizeof(std::uint32_t);
        bool ok = true;
        if (!s->inputs_on_device)
            ok = cudaMemcpyAsync(s->d_a, in.a.data.data(), bytes, cudaMemcpyHostToDevice,
                                 s->stream) == cudaSuccess &&
                 cudaMemcpyAsync(s->d_b, in.b.data.data(), bytes, cudaMemcpyHostToDevice,
                                 s->stream) == cudaSuccess;
        if (ok) {
#if !defined(PGPU_HOST_CHECK)
            if (s->gpu31_ok) {
                const u32 Bc = static_cast<u32>(B);
                const bool use_fast = s->crt_fast_ok && s->crt_fast_plan != nullptr;
                const bool use_mma = use_fast || (s->crt_mma_ok && s->crt_mma_plan != nullptr);
                ok = s->gpu31_dev->run(s->d_a, s->d_b, s->d_c, Bc, s->stream,
                                       /*skip_crt=*/use_mma) == cudaSuccess;
                if (ok && use_fast)
                    ok = s->crt_fast_plan->run(s->gpu31_dev->residues_device(), s->d_c,
                                               s->gpu31_dev->crt_args(), Bc, s->stream)
                         == cudaSuccess;
                else if (ok && use_mma)
                    ok = s->crt_mma_plan->run(s->gpu31_dev->residues_device(), s->d_c,
                                              s->gpu31_dev->crt_args(), Bc, s->stream)
                         == cudaSuccess;
                // The rules require fherma_run to return only after its device
                // work has finished. Without this, the kernels would run during
                // fherma_d2h, outside the ranked window - a violation.
                if (ok) {
                    ok = cudaStreamSynchronize(s->stream) == cudaSuccess;
                    s->gpu31_dev->report_profile();
                }
            } else
#endif
                ok = gpu_multiply(*s, static_cast<u32>(B));
        }
        if (ok) {
            if (!s->out_pool.empty()) {
                out.c.data = std::move(s->out_pool.back());
                s->out_pool.pop_back();
            } else {
                out.c.data.resize(static_cast<std::size_t>(B) * per);
            }
            s->result_on_device = true;
            return out;
        }
        // A launch or a synchronisation failed: stop trusting the device for the
        // rest of the run rather than leaving the answer half written.
        s->gpu_ok = false;
        s->release();
    }
#endif

    if (!s->out_pool.empty()) {
        out.c.data = std::move(s->out_pool.back());
        s->out_pool.pop_back();
    } else {
        out.c.data.resize(static_cast<std::size_t>(B) * per);
    }
    for (std::int64_t j = 0; j < B; ++j) {
        s->engine->multiply(in.a.data.data() + static_cast<std::size_t>(j) * per,
                            in.b.data.data() + static_cast<std::size_t>(j) * per,
                            out.c.data.data() + static_cast<std::size_t>(j) * per);
    }
    return out;
}

// ---------------------------------------------------------------------------
// the optional transfers
// ---------------------------------------------------------------------------
//
// Both may only copy, byte for byte, into and out of buffers prepared in
// fherma_init — no packing, no conversion, no arithmetic. When the device path
// is not in use they do nothing at all, so the harness reports no transfer.
#ifdef __CUDACC__
void fherma_h2d(void* state, const fherma::Inputs& in) {
    auto* s = static_cast<State*>(state);
    s->inputs_on_device = false;
    if (!s->gpu_ok || in.a.data.empty()) return;
    if (in.a.data.size() != s->in_elems || in.b.data.size() != s->in_elems) return;

    const std::size_t bytes = s->in_elems * sizeof(std::uint32_t);
    if (cudaMemcpyAsync(s->d_a, in.a.data.data(), bytes, cudaMemcpyHostToDevice, s->stream) !=
        cudaSuccess)
        return;
    if (cudaMemcpyAsync(s->d_b, in.b.data.data(), bytes, cudaMemcpyHostToDevice, s->stream) !=
        cudaSuccess)
        return;
    s->inputs_on_device = true;
}

fherma::Outputs fherma_d2h(void* state, fherma::Outputs answer) {
    auto* s = static_cast<State*>(state);
    if (s->gpu_ok && s->result_on_device && !answer.c.data.empty()) {
        const std::size_t bytes = answer.c.data.size() * sizeof(std::uint32_t);
        if (cudaMemcpyAsync(answer.c.data.data(), s->d_c, bytes, cudaMemcpyDeviceToHost,
                            s->stream) == cudaSuccess)
            cudaStreamSynchronize(s->stream);
        else
            cudaGetLastError();
    }
    s->result_on_device = false;
    s->inputs_on_device = false;
    return answer;
}
#endif

// Optional: per-stage device timings, accumulated over the passes of one case.
// The harness calls this after each case, outside the timed window.
std::vector<std::pair<std::string, double>> fherma_metrics(void* state) {
    auto* s = static_cast<State*>(state);
    std::vector<std::pair<std::string, double>> out;
    if (s && s->timing_ok) {
        out.push_back({"residue_ms", s->ms_res});
        out.push_back({"ntt_ms", s->ms_ntt});
        out.push_back({"crt_ms", s->ms_crt});
        s->ms_res = s->ms_ntt = s->ms_crt = 0.f;
    }
    return out;
}

// Optional.
void fherma_free(void* state) {
    auto* s = static_cast<State*>(state);
#ifdef __CUDACC__
    if (s) s->release();
#endif
    delete s;
}
