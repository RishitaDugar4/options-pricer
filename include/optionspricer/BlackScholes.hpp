#pragma once

#include "optionspricer/Option.hpp"

namespace optionspricer::black_scholes {

// Closed-form Black-Scholes-Merton price. European style only.
// Throws std::invalid_argument if params.style is American.
double price(const OptionParams& params);

// Analytic Greeks for the same model.
Greeks greeks(const OptionParams& params);

}  // namespace optionspricer::black_scholes
