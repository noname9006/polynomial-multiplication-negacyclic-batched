// Your answer. Three functions, and nothing to declare.
//
// This file is deliberately portable C++: it builds with nvcc through the
// generated CMakeLists (the submission path) and with g++ as a plain
// translation unit (the local test path, see ../../tools/run_local.ps1). The
// GPU path is added below under `#ifdef __CUDACC__` without disturbing it, so
// the CPU implementation stays available as the reference and as a fallback.
#include "fherma.h"

#include <memory>
#include <stdexcept>
#include <thread>

#include "polymul.hpp"

namespace {

struct State {
    polymul::Plan plan;
    std::unique_ptr<polymul::Engine> engine;
};

}  // namespace

// Setup. Once per point, not measured. It sees only (B, N, W, L, q), so every
// parameter-dependent table is built here: primes, twiddles, CRT and Barrett
// constants. On a GPU this is also where the device buffers, the streams and
// the copies of those tables would be allocated.
void* fherma_init(const fherma::Point& p) {
    if (p.q.data.size() != p.L) throw std::runtime_error("point: q has the wrong limb count");
    auto state = std::make_unique<State>();
    state->plan = polymul::build_plan(p.N, p.W, p.L, p.q.data.data());
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned threads = hw == 0 ? 1u : (hw > 64u ? 64u : hw);
    state->engine = std::make_unique<polymul::Engine>(state->plan, threads);
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

    if (in.a.shape.size() != 3 || in.b.shape != in.a.shape)
        throw std::runtime_error("run: expected a and b of shape {B, N, L}");
    if (static_cast<std::uint32_t>(in.a.shape[1]) != plan.N ||
        static_cast<std::uint32_t>(in.a.shape[2]) != plan.L)
        throw std::runtime_error("run: the tensor shape does not match the point");

    const std::int64_t B = in.a.shape[0];
    const std::size_t per = static_cast<std::size_t>(plan.N) * plan.L;
    if (static_cast<std::size_t>(in.a.data.size()) != static_cast<std::size_t>(B) * per)
        throw std::runtime_error("run: a has the wrong number of elements");

    fherma::Outputs out;
    out.c.shape = in.a.shape;
    out.c.data.resize(in.a.data.size());

    for (std::int64_t j = 0; j < B; ++j) {
        s->engine->multiply(in.a.data.data() + static_cast<std::size_t>(j) * per,
                            in.b.data.data() + static_cast<std::size_t>(j) * per,
                            out.c.data.data() + static_cast<std::size_t>(j) * per);
    }
    return out;
}

// Optional.
void fherma_free(void* state) {
    delete static_cast<State*>(state);
}

// ---------------------------------------------------------------------------
// Where the GPU path goes
// ---------------------------------------------------------------------------
//
// The shape of the work is fixed by the CPU path above; only the executor
// changes. In order of payoff:
//
//   1. decompose + twist   (N*L*k modular multiply-adds — a (L x k) matrix per
//                           coefficient; tensor cores if you split into bytes)
//   2. NTT per prime       (k * 3 * N log N butterflies; cuPQC-NTT ships a
//                           staged length-N transform, or write your own
//                           shared-memory kernel)
//   3. pointwise product   (k * N Montgomery multiplies)
//   4. inverse NTT+untwist (k * N log N + k * N)
//   5. Garner/CRT+Barrett  (k^2/2 small reductions and one 111x55-limb product
//                           per coefficient; the "tensor-core CRT" step)
//
// Chunk over B so the working set of one prime pass stays in L2: at
// N = 65536, W = 1747, k = 57, one pair is 3 * 57 * 65536 * 8 B ~= 90 MB.
//
// Nothing in this file is device code, which is worth keeping true for as long
// as possible. Probe the device once in `fherma_init`; if anything is missing —
// no device, a failed allocation, a failed upload — leave a flag clear and let
// `fherma_run` take the host path above. A submission that declines its fast
// path is still exact and merely slower; one that throws is a run with no
// result at all, and this is being edited from a machine with no GPU.
//
// `fherma_h2d` may only copy a and b byte for byte, `fherma_d2h` only copy c,
// and fherma_run must synchronise before returning. See the challenge rules.
