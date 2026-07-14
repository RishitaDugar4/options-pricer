#include "optionspricer/MonteCarlo.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace optionspricer::monte_carlo {

Result price(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
    params.validate();
    if (params.style != ExerciseStyle::European) {
        throw std::invalid_argument(
            "Monte Carlo terminal-payoff simulation only prices European-style options");
    }
    if (numPaths == 0) {
        throw std::invalid_argument("numPaths must be positive");
    }

    const double drift =
        (params.rate - params.dividendYield - 0.5 * params.volatility * params.volatility) *
        params.maturity;
    const double diffusion = params.volatility * std::sqrt(params.maturity);
    const double discount = std::exp(-params.rate * params.maturity);

    auto payoff = [&](double spotAtMaturity) {
        return params.type == OptionType::Call ? std::max(spotAtMaturity - params.strike, 0.0)
                                                 : std::max(params.strike - spotAtMaturity, 0.0);
    };

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Antithetic variates: pair each draw z with -z to reduce variance.
    double sum = 0.0;
    double sumSquares = 0.0;
    std::uint64_t sampleCount = 0;
    const std::uint64_t pairs = (numPaths + 1) / 2;
    for (std::uint64_t i = 0; i < pairs; ++i) {
        const double z = normal(rng);
        const double spotUp = params.spot * std::exp(drift + diffusion * z);
        const double spotDown = params.spot * std::exp(drift - diffusion * z);

        const double discountedUp = discount * payoff(spotUp);
        sum += discountedUp;
        sumSquares += discountedUp * discountedUp;
        ++sampleCount;

        if (sampleCount < numPaths) {
            const double discountedDown = discount * payoff(spotDown);
            sum += discountedDown;
            sumSquares += discountedDown * discountedDown;
            ++sampleCount;
        }
    }

    const double mean = sum / static_cast<double>(sampleCount);
    const double variance =
        sumSquares / static_cast<double>(sampleCount) - mean * mean;
    const double standardError =
        std::sqrt(std::max(variance, 0.0) / static_cast<double>(sampleCount));

    return Result{mean, standardError};
}

}  // namespace optionspricer::monte_carlo
