# FHERMA — negacyclic batched polynomial multiplication

A working, exact implementation of
[`polynomial-multiplication/negacyclic-batched@1.0.0`](https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic-batched),
plus the local harness needed to test it on a machine that has no GPU.

It is written as a portable C++ core with a CUDA path over it, so the same files

* build with `g++` here, where every answer can be checked against an
  independent Python oracle, and
* build with `nvcc` through the generated `CMakeLists.txt`, which is what the
  platform does.

The CPU path is therefore three things at once: the correctness reference for
the CUDA kernels, the fallback that runs when the device self-check disagrees,
and a legal submission on its own.

## Measured so far

| point | B | N | W | local CPU (6 cores / 12 threads) | leaderboard |
| --- | --- | --- | --- | --- | --- |
| ranked | 128 | 65,536 | 1,747 | one pair 0.37 s, so ≈ 55 s for a batch | 21.1 ms |
| #1 | 1 | 65,536 | 1,747 | 0.37 s | 828 µs |
| #2 | 1 | 32,768 | 868 | 0.07 s | 248 µs |

The CPU path is roughly three orders of magnitude off the GPU field, which is
the expected outcome — it exists to be *right*, and to be the reference the
CUDA kernels are diffed against.

The CUDA path has not been measured here: this machine has neither `nvcc` nor a
GPU. What is checked locally is its arithmetic, bit for bit against the host
path, and its host half; the device self-check described below is what decides
on the measuring machine whether it is used at all.

## Layout

This directory is the repository: its root is the implementation the platform
builds (`fherma.toml` sits here, and the platform runs
`cmake -B build && cmake --build build` then `./build/solution`).

```
.
  polymul.hpp     the whole algorithm: big integers, RNS, NTT, Garner, Barrett
  polymul_gpu.cuh the same arithmetic and stage order as device kernels
  solve.cu        fherma_init / fherma_run / fherma_free  (the answer)
  main.cpp        generated harness — do not edit
  fherma.h        generated types — `fherma implementation init <spec> --update` rewrites them
  CMakeLists.txt  generated CUDA build — do not edit
  fherma.toml     generated manifest; the build command lives here
  tools/
    bundle.py         make point directories, verify answers against the definition
    run_local.ps1     build + make + run + verify, one command
    selftest.ps1      the eight-point regression matrix
    gpucheck.cpp      the device kernels against the host path, one thread per block
    crtcheck.cpp      the weighted-sum CRT's integer identities, checked with Big
    hostcheck.ps1     type-check solve.cu and the kernels with g++ (no nvcc needed)
    qtable.py         writes qtable.hpp: the q each gpucheck point uses
    shim/             a one-thread stand-in for the CUDA execution model
    fakecuda/         declarations of <cuda_runtime.h> for hostcheck
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

### The device half, without a device

None of the CUDA work can be compiled or run here: there is no `nvcc` and no
GPU. Two checks stand in for that, and both are cheap:

```powershell
# the kernels' arithmetic against the host path, bit for bit
g++ -O2 -std=c++17 -o build-cpu/gpucheck.exe tools/gpucheck.cpp
./build-cpu/gpucheck.exe

# every name, type and call in solve.cu, plus the kernels, type-checked
./tools/hostcheck.ps1
```

`gpucheck` includes `polymul_gpu.cuh` under `tools/shim/cuda_shim.hpp`, which
maps the execution model to one block of one thread. That is faithful for these
kernels — every thread owns whole elements and the only cross-thread dependency
is the barrier between NTT stages — so it exercises the Montgomery products, the
twiddles, the weighted-sum CRT and the two Montgomery steps that finish it, and
compares the finished coefficients with `Engine::multiply`. It runs
the ranked point's arithmetic (k = 57, L = 55, P = 3,534 bits) at N = 1,024, the
ranked point's length (N = 65,536, all sixteen stages of the two-pass split) at
a narrow q, and several smaller points end to end — every point through both
paths, in 0.6 s.

`crtcheck` is the same idea one level lower: it takes the reconstruction's
tables and its claim — `S ≡ X·2^128 (mod q)`, `t = round(X/P)` from the 192-bit
weights, `0 ≤ T < q·2^128` — and verifies each one with `Big` arithmetic over
random residue sets, so a mistake in the number theory fails on this machine
instead of on the run page. It also reads the ranked point's 1,747-bit `q` from
a point directory (`crtcheck q.bin 65536 1747`).

What the shim cannot check is the parallel decomposition: whether block `b` and
thread `t` are handed the right elements. That is what the self-check below does,
on the real device.

`hostcheck.ps1` compiles `solve.cu` as C++ with `__CUDACC__` defined and
`<cuda_runtime.h>` declared by `tools/fakecuda`, and turns each kernel launch
into a plain call. It cannot check launch geometry, the register budget or the
generated code — only that the file is well-formed, which is otherwise a
platform round trip to discover.

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

## The CUDA path

Three kernels, one per stage, in `polymul_gpu.cuh`; `solve.cu` owns the device
memory, the stream, and the decision whether to use them at all.

1. **`k_residues`** — one block per (pair, tile of 128 coefficients):
   `blockIdx.y` is the pair, `blockIdx.x` the tile. The tile's raw coefficients
   (128 × 55 u32 = 28 KiB) are staged in shared memory once, and the block then
   walks all 57 primes, writing `Σ_l a_l · 2^{32l} mod p_m` in Montgomery form,
   twisted. Reading the coefficient once per tile instead of once per prime is
   what makes the stage cheap: 14 MiB per operand instead of 57 × 14 MiB, plus
   29.9 MiB of residues out.

2. **The NTT** — two passes, four kernels, one launch per pass. Read the array
   as `N/256` rows of 256: the stages with half-length ≥ 256 pair two elements
   of the same column, the rest pair two elements of the same row. So a block
   holds a strip of eight whole columns (or rows) in shared memory and runs its
   eight stages there, then writes the strip back. The forward column pass
   carries both operands; the forward row pass ends with the pointwise product
   and does not write `b` back, since it is dead; the inverse row pass follows;
   the inverse column pass untwists on the way out (`twist_inv` carries both
   ψ^-i and N^-1). The stages, their order and their twiddles are the same as
   the host's — only where the data sits between them has changed — so the
   values are identical stage for stage.

3. **`k_crt_w`** — one thread per coefficient, `blockIdx.y` the pair. Not the
   Garner chain: the same reconstruction as one weighted sum,
   `S = Σ_j res_j·W'_j` with `W'_j = e_j·2^128 mod q` (`e_j` the CRT idempotent),
   which is 57×28 multiply-adds instead of a 57²/2-digit chain over 112-limb
   numbers — about 2k operations per coefficient instead of 20k. The sign comes
   from a second, 192-bit sum over `C_j = ⌊e_j·2^192/P⌋`:
   `t = round(Σ res_j·C_j / 2^192)` is `round(X/P)` exactly, because the
   estimate is off by less than `2^-124` while the true fractional part is at
   least `(P - bound)/2P ≈ 2^-24` from one half. Then
   `T = S + q·2^68 - t·(P mod q)·2^128` is positive and below `q·2^128`, so
   two single-limb Montgomery steps plus two conditional subtractions finish it.
   The accumulator is a compile-time-size register array (one kernel instance per
   size, `NQ` from 4 to 64), so no coefficient touches local memory.

   `k_crt`, the Garner chain, is still there and still exact: it runs when `q` is
   wider than the tables are built for, and it is what the host path uses.

Pairs go through in passes of 16: the two residue buffers are 478 MB each, and
each NTT launch is 29,184 blocks of 256 threads (32 column tiles × 57 primes ×
16 pairs) — about 52 waves on the runner's 188 SMs if three blocks fit per SM.
The ranked batch is eight such passes, queued on one stream.

Device arithmetic is Montgomery with `__umul64hi` only — device code has no
guaranteed `unsigned __int128` — so a non-Montgomery `a·b mod p` is two
Montgomery products rather than a 128-bit division.

**Trust.** `fherma_init` sends two random pairs through the whole device
pipeline and compares them, limb for limb, with the host path — the same
buffers, the same kernels `fherma_run` will use. Two rather than one because a
single pair makes every `pair` index zero, which would hide a wrong stride. That
is not hypothetical: the first run on the card had both CRT kernels writing
`out + i*L` instead of `out + (pair*N + i)*L`, so pair 1 overwrote pair 0 and
everything past the first pair stayed as the driver left it — 7,208,959 of
7,208,960 limbs wrong, and the self-check said so in one line before the host
answer still passed the case. `gpucheck` had missed it because it handed the
kernel a per-pair pointer; it now passes the whole buffer and lets
`blockIdx.y` do the offset, which is what `solve.cu` does. Any mismatch,
allocation failure, launch failure, or a point outside the built-in limits
(k ≤ 64, L ≤ 64, P ≤ 192 limbs) leaves the host path in place, and the host path
is in the same binary, so there is nothing to fall back *to*. Init is not timed,
so this costs the ranking nothing.

**Transfers.** `fherma_h2d` and `fherma_d2h` copy byte for byte, nothing else —
as the rules require since 13 November, when they moved outside the ranked time
(limits 0.45 s in, 0.25 s out). `fherma_run` still copies if they were not
called, and is then measured with the copy.

**Cost before measuring.** The card is an RTX PRO 6000 Blackwell: 96 GB,
188 SMs, 1.79 TB/s — which is what makes this shape of work affordable. Per
(pair, prime) the two NTT passes move 5.5 MB — read a+b, write a+b, then read
a+b and write a, then two read-writes of a — against 52 MB for the
stage-at-a-time version it replaced. At 57 primes and 128 pairs that is 40 GB
per batch, about 22 ms of traffic; the floor for this decomposition is 4 MB per
(pair, prime), ~16 ms, because a 512 KB transform cannot stay in a 100 KB shared
memory. The residue stage adds 3.8 GB written and 3.6 GB read (the inputs
themselves), and the reconstruction reads 30 MB per pair. So the honest
expectation is the low tens of milliseconds, and the ranked figure will be set
by how well the kernels overlap with the traffic — which is exactly what cannot
be answered on a machine with no GPU. The residue stage is the one to watch for
arithmetic, at 55 Montgomery products per (coefficient, prime).

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
  synchronous by construction; the CUDA path synchronises the stream before it
  returns, and `fherma_d2h` only copies.
* `fherma_h2d` and `fherma_d2h` may only copy, byte for byte, and may not launch
  a kernel or change the representation. Ours copy and nothing else; the limits
  are 0.45 s in and 0.25 s out for 3.69 GB and 1.84 GB.
* If the run page reports seconds rather than milliseconds, the device self-check
  refused and the host path ran: stderr says which (a disagreement, a failed
  launch, an allocation). That is the switch to look at first.
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

`CMakeLists.txt` builds with `-arch=sm_120`, and that is read off the platform
rather than guessed: the challenge's own runner page lists
`fherma-gpu-rtx6000` as an NVIDIA RTX PRO 6000 Blackwell Server Edition —
compute capability 12.0, 96 GB, 1.79 TB/s. It does not match the `sm_89` the
generated file shipped with (the Ada generation, which the file's comment names
as an L40S), and `-arch=sm_89` emits SASS for 8.9 and no PTX, so those kernels
would not load on the runner at all. That is why nothing here depends on it
working.

If the card changed again, the failure would not be silent and would not be a
wrong answer: the self-check in `fherma_init` launches the real kernels, the
launch fails, and the host path takes over — ~55 s per case instead of
milliseconds on the run page. Then, and only then:

```toml
[build]
command = "cmake -B build -DCMAKE_BUILD_TYPE=Release -DFHERMA_CUDA_ARCH=sm_89 && cmake --build build"
```

The arch has to be one the image's `nvcc` knows, and the image is CUDA 12.8 +
cuPQC 0.6.0's.

## What the leaderboard numbers say

Ranked time is 21.1 ms for all 128 products, with host↔device copies excluded
since 13 November. 3.69 GB of inputs and 1.84 GB of output would take ~90 ms at
PCIe Gen5 rates, so the field is not transfer-bound — the ranked figure is
device arithmetic, and the top entries describe themselves as RNS NTT plus a
*tensor-core CRT*. That matches the shape of the work: the NTT stages are
`128 · 57 · 3 · (N/2)·log₂N ≈ 1.1e10` butterflies, and a Garner chain would be
`128 · N · k²/2 ≈ 1.3e10` serially dependent reductions — which is why the
reconstruction here is a weighted sum instead (2k per coefficient), and why the
leaders put that sum on matrix units.

## Where the GPU path goes

The stage order and the per-stage sizes are fixed by `polymul.hpp`; only the
executor changes. In payoff order:

1. **The residue stage.** Each coefficient is 55 limbs, and each of the 57
   primes walks all of them: `128 · 57 · 65536 · 55 ≈ 2.6e10` limb multiply-adds
   per batch, twice the NTT's butterfly count. It is the largest stage left, and
   the fix is the usual one — keep the coefficient in registers and accumulate
   all 57 residues in one pass over its limbs, so each limb is loaded once
   instead of 57 times and the carry chain is shared.
2. **Fewer passes.** Sixteen pairs at a time is a compromise; the residue
   buffers could be sized to the whole batch once the NTT no longer thrashes
   them.
3. **The NTT itself.** Four launches per pass is close to the traffic floor at
   these sizes, but the butterflies still run one per thread with a
   shared-memory twiddle read. Four elements per thread would halve the
   twiddle traffic, and the top entries' *tensor-core* CRT suggests they moved
   the reconstruction into matrix units rather than CUDA cores.

Measured against the field: 21.1 ms is the winning figure, so a first device
run in the low tens of milliseconds is a contender and one in the hundreds is a
place on the board.
