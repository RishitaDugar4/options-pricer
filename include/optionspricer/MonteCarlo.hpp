#pragma once

#include <cstdint>

#include "optionspricer/Option.hpp"

namespace optionspricer::monte_carlo {

struct Result {
    double price = 0.0;
    double standardError = 0.0;
};

// Monte Carlo price under geometric Brownian motion, using antithetic
// variates for variance reduction. European style only (terminal payoff
// simulation cannot value early exercise).
// Path simulation is SIMD-vectorized where available: 8 single-precision
// paths per batch via AVX2 on x86_64 (-mavx2), or via two NEON float32x4_t
// half-batches on arm64. Falls back to a scalar, double-precision
// implementation otherwise.
// Throws std::invalid_argument if params.style is American.
Result price(const OptionParams& params, std::uint64_t numPaths = 100000,
             std::uint64_t seed = 42);

// True if price() runs the vectorized (AVX2/NEON) path on this build, false
// if it's using the scalar fallback.
bool hasSimdMonteCarlo();

// Implementation variants exposed for benchmarking SIMD vs. scalar
// head-to-head. Prefer price() for normal use: these skip the style and
// argument validation price() performs.
namespace detail {
Result priceScalar(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed);
Result priceSimd(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed);
}  // namespace detail

}  // namespace optionspricer::monte_carlo
