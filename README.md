# FHERMA — negacyclic batched polynomial multiplication

A working, exact implementation of
[`polynomial-multiplication/negacyclic-batched@1.0.0`](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic-batched),
plus the local harness needed to test it on a machine that has no GPU.

It is written as a portable C++ core, so the same files

* build with `g++` here, where every answer can be checked against an
  independent Python oracle, and
* build with `nvcc` through the generated `CMakeLists.txt`, which is what the
  platform does.

The CPU path is therefore two things at once: the correctness reference for a
CUDA implementation, and a legal submission on its own.

## Measured so far

| point | B | N | W | local CPU (6 cores / 12 threads) | leaderboard |
| --- | --- | --- | --- | --- | --- |
| ranked | 128 | 65,536 | 1,747 | one pair 0.37 s, so ≈ 55 s for a batch | 21.1 ms |
| #1 | 1 | 65,536 | 1,747 | 0.37 s | 828 µs |
| #2 | 1 | 32,768 | 868 | 0.07 s | 248 µs |

The CPU path is roughly three orders of magnitude off the GPU field, which is
the expected outcome — it exists to be *right*, and to be the reference the
CUDA kernels are diffed against.

## Layout

This directory is the repository: its root is the implementation the platform
builds (`fherma.toml` sits here, and the platform runs
`cmake -B build && cmake --build build` then `./build/solution`).

```
.
  polymul.hpp     the whole algorithm: big integers, RNS, NTT, Garner, Barrett
  solve.cu        fherma_init / fherma_run / fherma_free  (the answer)
  main.cpp        generated harness — do not edit
  fherma.h        generated types — `fherma implementation init <spec> --update` rewrites them
  CMakeLists.txt  generated CUDA build — do not edit
  fherma.toml     generated manifest; the build command lives here
  tools/
    bundle.py         make point directories, verify answers against the definition
    run_local.ps1     build + make + run + verify, one command
    selftest.ps1      the eight-point regression matrix
  build-cpu/          local g++ build (disposable, gitignored)
  work/               generated case data and outputs (disposable, gitignored)
```

## Build and test

Build and run **in one command**: Windows Defender deletes the freshly linked
`.exe` between shell invocations here, so a build that succeeds and a later
`./solution.exe` in a different shell is a coin flip. The scripts already do
the right thing.

```powershell
# one point, end to end
./tools/run_local.ps1 -Point '{"B":2,"N":1024,"W":128}' -Cases 3

# the whole matrix, including the ranked point
./tools/selftest.ps1
```

`run_local.ps1` takes `-Python` if `python` is not the interpreter with the
right version (here it is `.venv-fherma/Scripts/python.exe`).

Verification of a big point does not use schoolbook multiplication; see
[Verification](#verification).

## Algorithm

Everything happens in `polymul.hpp`, which is deliberately the same algorithm a
GPU version must implement, just on one core.

1. **Parameters.** `q` is handed in at setup: the largest prime below `2^W` with
   `q ≡ 1 (mod 2N)`. At the ranked point `q` is 1,747 bits, `L = 55` limbs, and
   the top limb is `0x7ffff`.

2. **Bound.** A product coefficient is `Σ ± a_i b_j` over at most `N` terms, so
   `|c_k| ≤ N (q-1)^2`. Per coefficient the implementation works with the signed
   value, so it is enough that the RNS product exceeds `2N (q-1)^2` — **3,511
   bits** at the ranked point.

3. **RNS primes.** Descend from `2^62 - 1` in steps of `2N`, keeping primes
   `p ≡ 1 (mod 2N)`. That yields **k = 57** primes, `P = Π p_m` of **3,534 bits**,
   comfortably over the bound. `p < 2^62` keeps `a·b` inside 124 bits, so
   Montgomery reduction with `unsigned __int128` is exact and needs no
   multi-limb fallback. Primality is proven with a Miller–Rabin base set that is
   deterministic below `3.3e24`.

4. **Negacyclic NTT.** With `ψ` of exact order `2N` (found by raising a candidate
   to `(p-1)/2N` and checking `ψ^N = -1`), the twist `a_i ψ^i` turns
   `Z_p[X]/(X^N+1)` into a cyclic ring, and a length-`N` NTT with `ω = ψ²`
   diagonalises it. Twiddles are precomputed per prime and stored in Montgomery
   form, indexed `len + j` for the stage of half-length `len`.

5. **Reconstruction.** Garner digits `d_0..d_{k-1}` from the residues, then one
   `Σ d_m Π_{j<m} p_j` evaluation per coefficient, then Barrett reduction modulo
   `q` with `μ = floor(B^m / q)`, `B = 2^32`, `m = limbs(P) + 1`. Because the
   reconstruction is exact and the true value may be negative, a second branch
   subtracts `P mod q` when the value exceeds `P/2`.

All of it — primes, roots, twists, limb weights, `μ` — depends only on
`(N, W, L, q)`, so it is built once in `fherma_init`, which the harness does not
time. That is the same place a CUDA version allocates its buffers and uploads
its tables.

Parallelism is a plain fork-join over two independent axes: the k primes in the
NTT stage, and the N coefficients in the reconstruction stage. Chunks own
disjoint output ranges, so the answer is bit-identical to the serial loop.

### Table sizes for the ranked point

| thing | size |
| --- | --- |
| `P` | 3,534 bits = 111 limbs |
| bound `2N(q-1)^2` | 3,511 bits |
| one poly in RNS domain, per prime | `N · u64` = 512 KiB |
| all k primes, one polynomial | `k · N · u64` = 29.9 MB |
| twiddles and twists, per prime | 4 tables × `N · u64` = 2.1 MB |
| the same for all k primes | 119 MB |
| one pair, a and b in and c out | 43 MB |

A and B alone are 60 MB per pair, and the precomputed tables add 119 MB that
every pair reads. At B = 128 that is 3.69 GB of inputs — so a GPU version wants
to keep the tables resident, process pairs in blocks that fit in L2 (126 MB on
the RTX PRO 6000), and stream `a`/`b` through the primes rather than
materialising all `k` residues for a whole batch.

## Verification

`tools/bundle.py` writes point directories in the on-disk protocol `main.cpp`
speaks, and checks the answers independently of the C++:

* every returned coefficient must be canonical, `0 ≤ c_k < q`;
* for `N ≤ 2048` (`--full-limit`), full schoolbook negacyclic multiplication;
* for larger `N`, compensation for a verifier that would otherwise be as slow as
  the thing it checks:
  * `--algebraic` (on by default): evaluate `a`, `b` and `c` at an element `z`
    with `z^N = -1 (mod q)`. `X ↦ z` is a ring homomorphism
    `Z_q[X]/(X^N+1) → Z_q`, so `A(z)·B(z) = C(z)` must hold. A wrong coefficient
    anywhere breaks it with probability `1 - N/q`, and it costs one polynomial
    evaluation per operand.
  * `--spot COUNT`: `COUNT` random coefficients recomputed directly from the
    definition, `c_k = Σ_{i+i'=k} a_i b_i' - Σ_{i+i'=k+N} a_i b_i'`.

`--no-algebraic` turns the first off, `--spot 0` the second.

This is a test aid, not the official verifier: the platform runs the
specification's own bundle, and it is stricter than anything here.

## Submitting

The platform takes a **repository plus a commit**, not an upload: you push this
directory, then register the repository and the commit SHA on the implementation
and declare its coverage. A branch name is not enough — the commit is what gets
measured.

```powershell
# from this directory, if you want the platform's pipeline shape locally
$env:PYTHONIOENCODING = "utf-8"   # the CLI crashes printing ✓ on a cp1252 console
..\..\.venv-fherma\Scripts\fherma.exe implementation run --bundle <dir> --point '{"B":128,"N":65536,"W":1747}'
```

(`implementation check` is for specifications judged by a reference
implementation — it validates that repository's `[harness]` table, and this is
not one of those.)

Then, on the challenge page, "Submit an implementation" → point it at the repo
and commit, declare coverage, and let it build and measure. Private repos are
allowed (a read-only clone token goes beside the repo reference), and the source
need not be published.

Checklist:

* Coverage must include `B=128, N=65536, W=1747` — that point decides the
  placing. `fherma.toml` already targets the batched spec.
* `fherma_init` may not look at `a` or `b`; it does not, and must not.
* `fherma_run` must return only after all its work is done. The CPU path is
  synchronous by construction; a CUDA path needs `cudaStreamSynchronize`.
* `N`, `W`, `L`, `q` are parameters, not constants. Handled: every table is
  built from them at setup, and the regression matrix includes `N = 1`,
  `N = 2`, `W = 2`, `W = 254` (not a multiple of 32) and the two real widths.
* Do not edit `main.cpp`, `fherma.h` or `CMakeLists.txt`. The platform
  overwrites the generated files with its own copies before building, so local
  edits there cannot change how the submission is measured anyway.
* `build/`, `build-cpu/` and `work/` are gitignored. The platform makes its own
  `build/`.

The official test bundle is not distributed, so local verification is
`tools/bundle.py`. If you want the platform's own harness shape instead,
`fherma testing init polynomial-multiplication/negacyclic-batched@1.0.0`
scaffolds the bundle template (`generate.py`, `oracle.py`, `derived.py` to fill
in) and `fherma implementation run --bundle <dir>` will drive the real runner
pipeline over it, untimed setup and all.

### The CUDA architecture question

The generated `CMakeLists.txt` defaults to `-arch=sm_89`, but the runner on the
leaderboard is `fherma-gpu-rtx6000` — an RTX PRO 6000, i.e. Blackwell,
`sm_120`. Two consequences:

* A pure host answer (what is here today) has no device code, so the arch does
  not matter.
* Once kernels are added, the default build relies on the PTX in the fatbin
  being JIT-compiled for `sm_120` at first launch. That works, but the JIT cost
  lands wherever the first kernel launch happens — so launch a trivial kernel
  from `fherma_init` to pay it outside the timed window, or build natively:

  ```toml
  [build]
  command = "cmake -B build -DCMAKE_BUILD_TYPE=Release -DFHERMA_CUDA_ARCH=sm_120 && cmake --build build"
  ```

  Native `sm_120` is the faster option, but only try it if you can also verify
  that cuPQC's device-LTO objects link for that architecture — `-dlto` is on, and
  cuPQC 0.6.0 in the challenge image was most likely built for `sm_89`.

## What the leaderboard numbers say

Ranked time is 21.1 ms for all 128 products, with host↔device copies excluded
since 13 November. 3.69 GB of inputs and 1.84 GB of output would take ~90 ms at
PCIe Gen5 rates, so the field is not transfer-bound — the ranked figure is
device arithmetic, and the top entries describe themselves as RNS NTT plus a
*tensor-core CRT*. That matches the shape of the work: the NTT stages are
`128 · 57 · 3 · (N/2)·log₂N ≈ 1.1e10` butterflies, and Garner reconstruction is
`128 · N · k²/2 ≈ 1.3e10` small modular reductions, so the reconstruction is not
a rounding error to leave on CUDA cores.

## Where the GPU path goes

`polymul.hpp` fixes the algorithm; only the executor changes. The stage order,
the per-stage sizes and the chunking hint are written out at the bottom of
`solve.cu`. In payoff order: NTT over primes, pointwise product, inverse NTT,
the `L × k` decomposition matmul, then the CRT/Barrett reconstruction.
