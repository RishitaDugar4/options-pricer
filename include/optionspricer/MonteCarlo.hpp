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
// On x86_64 builds compiled with AVX2 (-mavx2), path simulation runs 8
// single-precision paths per vector op; otherwise falls back to a scalar,
// double-precision implementation.
// Throws std::invalid_argument if params.style is American.
Result price(const OptionParams& params, std::uint64_t numPaths = 100000,
             std::uint64_t seed = 42);

}  // namespace optionspricer::monte_carlo
