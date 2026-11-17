# cuPQC-NTT 0.6.0 vs. our tiled DIT NTT (`gpu31.cuh`)

Scope: can NVIDIA's cuPQC-NTT (shipped in `fherma/cupqc`, SDK 0.6.0) replace the
hand-written tiled DIT NTT that costs ~38 ms of our ~128 ms total? Sources:
[feature page](https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_ntt/cupqc_ntt_feature.html),
[usage guide](https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_ntt/cupqc_ntt_usage.html),
[API](https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_ntt/cupqc_ntt_api.html),
[examples](https://github.com/NVIDIA/cuPQC/tree/main/examples/ntt),
[release notes](https://docs.nvidia.com/cuda/cupqc/additional/release_notes.html).

## Short answers to the three questions

1. **Custom 31-bit primes / own roots: YES.** The API takes the modulus `p` and a
   primitive root `g` as *runtime* arguments: `NTT().make_twiddles(twiddles, p, g)`
   and `cupqc::nttConst<uint32_t>(p)` (computes `p`, `-p^-1 mod 2^32`, `R^2 mod p`).
   BabyBear/KoalaBear are only convenience constants; any prime is accepted
   (up to 62 bits with `uint64_t`; with `uint32_t`, our `p < 2^31` is fine).
   We would pass `g = ω = ψ²` from our existing `Plan` (we already compute ψ, ω,
   ψ^-1, ω^-1, N^-1 and all twist tables). **Caveat:** cuPQC only implements the
   **cyclic** transform (mod `x^N - 1`); we are negacyclic (mod `x^N + 1`).
   The standard ψ-twist reduction must be done by us at the load/store boundary
   (we already have the tables; see sketch). Twiddles must be converted to
   Montgomery form once (`transform_twiddles_to_mont` or per-element `to_mont`);
   our coefficients are already `a·2^32 mod p`, so `load()`/`store()` (raw copies)
   are the right boundary calls.

2. **Exact API / staged path for N=65536:** header `#include <ntt.hpp>` (namespace
   `cupqc`), descriptor composed with operators; sizes must be compile-time:
   - Single-stage (`uint32_t`, custom p) supports only `2 ≤ d ≤ 13`, i.e. N ≤ 8192.
     **N=65536 (d=16) requires staged mode** via `SubSize<M>`; allowed `M` for
     d=16: 256/512/1024/2048/4096/8192, with `K = N/M`.
   - Descriptor: `decltype(Algorithm<algorithm::NTT>() + Direction<nttDirection::FORWARD>()
     + Precision<uint32_t>() + Size<65536>() + SubSize<256>() + Block() + BlockDim<128>())`.
   - Forward = 2 kernels: stage 1 `stage_1_load_to_mont → stage_1_execute →
     stage_1_store` (grid `M` blocks, smem `fwd_stage_1_ntt_shared_workspace_size<65536,256,uint32_t>()`
     = K elements); stage 2 `stage_2_load → stage_2_execute → stage_2_store_from_mont`
     (grid `K` blocks, smem `fwd_stage_2_...` = M elements).
   - Inverse reverses the stages (stage 1 grid `K`, stage 2 grid `M`) and applies
     `N_inv` in `stage_2_execute(..., N_inv)`; `n_inv<65536>(p)` computes it.
   - `execute` operates on a **shared-memory** buffer only; never pass a global
     pointer. `__syncthreads()` between load/execute/store. Twiddles are generated
     by a single-thread kernel `<<<1,1>>>`; `transform_twiddles_to_mont` is blockwise.

3. **Linking / LTO risk: YES, a second LTO static library is required.** Add
   `-lcupqc-ntt` (CMake: `cupqc-ntt_static`, or `$ENV{CUPQC_SDK_DIR}/lib/libcupqc-ntt.a`)
   on top of `libcupqc-bigint.a`, with `-dlto` at compile *and* device-link
   (`$<DEVICE_LINK:...>` in `CMakeLists.txt`, which is generated and must not be
   edited locally). Expect the device-link step to get heavier; we already hit a
   **600 s device-link timeout** with BigInt alone, so adding another LTO archive
   is a concrete build-failure risk, not just a slowdown. Additional trap from the
   examples' README: *"a configuration the type system accepts but the shipped
   library does not provide compiles cleanly and then fails to link"* — the exact
   `(N=65536, SubSize<M>, custom prime)` specialisation must be link-tested.

## Other hard constraints / risks

- **Cyclic only** (docs, Transform type). Negacyclic needs the twist dance:
  `â_i = a_i·ψ^i`, cyclic NTT, pointwise multiply, inverse NTT, then
  `c_i = (INTT(â̂·b̂)_i)·ψ^{-i}·N^{-1}`. Our `encode` currently applies a
  *bit-reversed* twist and `multiply_reverse` reorders for our inverse; with cuPQC
  we must determine experimentally whether `execute` consumes natural or
  bit-reversed order (docs do not say; the SDK header is not in the public GitHub
  repo). Expect to rework `reindex`/`encode`/`multiply_reverse`/`decode`.
- Staged NTT writes an **intermediate global-memory buffer in Montgomery form**
  between its two passes, just like our `low_stages`/`high_stages`. The traffic
  advantage of our current design (5.5 MB per (pair,prime), README ~261) is not
  inherently improved; cuPQC adds up to 4 launches per fwd+inv pair instead of 2
  in `transform()`, and cannot fuse the twist into its butterflies.
- No local GPU/SDK in this workspace (`CUPQC_SDK_DIR` unset), so neither a link
  test nor a microbenchmark can be run here.

## Code sketch (how we would call it for N=65536, our primes)

```cpp
#include <ntt.hpp>
using namespace cupqc;

constexpr uint32_t NTT_N = 65536, NTT_M = 256, NTT_K = NTT_N / NTT_M; // 256

template <nttDirection Dir>
using Staged = decltype(Algorithm<algorithm::NTT>() + Direction<Dir>()
                       + Precision<uint32_t>() + Size<NTT_N>()
                       + SubSize<NTT_M>() + Block() + BlockDim<128>());
using Fwd = Staged<nttDirection::FORWARD>;
using Inv = Staged<nttDirection::INVERSE>;

// One-time, per prime k (ψ, ω=ψ² from our existing Plan):
//   twiddles F: make_twiddles<Fwd><<<1,1>>>(d_tw, p_k, omega_k);
//   twiddles I: make_twiddles<Inv><<<1,1>>>(d_itw, p_k, omega_inv_k);
//   transform_twiddles_to_mont<Fwd><<<1,128>>>(d_tw, p_k);  // same for d_itw
//   N_inv = n_inv<NTT_N>(p_k);  twist[k][i] = ψ_k^i·2^32 mod p (ours, Montgomery);
//   untwist[k][i] = ψ_k^{-i}·2^32 mod p  (do NOT fold N_inv in: INTT applies it).

__global__ void fwd_s1(const u32* a, u32* out, const u32* tw, const u32* psi,
                       const Prime* primes, u32 Kprimes, u32 B) {
    extern __shared__ u32 s[];
    const int m = blockIdx.y, b = blockIdx.z;            // prime, pair
    const u64 off = (u64(b) * Kprimes + m) * NTT_N;
    Fwd ntt;
    ntt.stage_1_load_to_mont(s, a + off, blockIdx.x, nttConst<u32>(primes[m].p));
    __syncthreads();
    // negacyclic twist in-place: s[i] = mont(s[i], psi[m*NTT_N + i], p)
    ntt.stage_1_execute(s, tw, primes[m].p);
    __syncthreads();
    ntt.stage_1_store(s, out + off, blockIdx.x);
}
// fwd_s2: stage_2_load -> stage_2_execute -> stage_2_store_from_mont
// inv_s1: stage_1_load(_to_mont) -> stage_1_execute(d_itw) -> stage_1_store  (grid K)
// inv_s2: stage_2_load -> untwist in-place -> stage_2_execute(d_itw, p, N_inv)
//         -> stage_2_store_from_mont                                      (grid M)
// Host: fwd_s1<<<dim3(M,k,B),128,fwd_s1_smem>>>; fwd_s2<<<dim3(K,k,B),128,fwd_s2_smem>>>;
//       inv_s1<<<dim3(K,k,B),128,inv_s1_smem>>>; inv_s2<<<dim3(M,k,B),128,inv_s2_smem>>>;
// Pointwise multiply across primes stays as `multiply_reverse`/reconstruct;
// ordering convention must be validated against a round-trip first.
```

CMake (if pursued, on a **branch**, not this workspace): add
`$ENV{CUPQC_SDK_DIR}/lib/libcupqc-ntt.a` to `target_link_libraries` next to
`libcupqc-bigint.a`. Includes already cover `$ENV{CUPQC_SDK_DIR}/include/cupqc`
where `ntt.hpp` lives.

## Verdict: **TOO RISKY** (as a replacement in the ranked build)

Reasons, in priority order:
1. Adding a second device-LTO archive to a build that already hits a 600 s
   device-link timeout can turn a timing problem into a hard build failure.
2. Cyclic-only transform: negacyclic requires re-plumbing twist/untwist and
   possibly the bit-reversal layout; correctness risk is high without a GPU to
   test on, and a wrong answer silently falls back to the ~55 s host path.
3. N=65536 forces staged mode: 4 launches per fwd+inv, global intermediate,
   traffic comparable to our already traffic-tuned tiled DIT — the theoretical
   floor from README (~16 ms for this decomposition) is not lowered by cuPQC.
4. Specialisation-at-link trap: `(N, M)` must exist in the shipped LTO archive.
5. Payoff is bounded: NTT is 38 ms/128 ms; even a 2× NTT speedup saves 19 ms
   (~15 % total); a realistic 1.0–1.3× saves 0–11 ms (0–9 %).

**Predicted speedup range:** NTT portion ~0.8×–2.0× (most likely ~1.0–1.3×);
total run ~1.0×–1.15× (likely ~1.0×–1.09×, i.e. 0–11 ms of 128 ms).

**Only revisit as an isolated microbenchmark** (separate .cu, link NTT library
*without* BigInt to measure link time and kernel time) if the device-link timeout
is first solved (e.g. splitting libraries or raising the limit). Do not swap it
into `gpu31.cuh` based on docs alone.
