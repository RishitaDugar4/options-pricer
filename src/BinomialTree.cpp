#include "optionspricer/BinomialTree.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace optionspricer::binomial_tree {

double price(const OptionParams& params, int steps) {
    params.validate();
    if (steps <= 0) {
        throw std::invalid_argument("steps must be positive");
    }

    const double dt = params.maturity / steps;
    const double u = std::exp(params.volatility * std::sqrt(dt));
    const double d = 1.0 / u;
    const double growth = std::exp((params.rate - params.dividendYield) * dt);
    const double p = (growth - d) / (u - d);
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument(
            "risk-neutral probability out of [0,1]; reduce steps or check inputs");
    }
    const double discount = std::exp(-params.rate * dt);

    auto payoff = [&](double spot) {
        return params.type == OptionType::Call ? std::max(spot - params.strike, 0.0)
                                                 : std::max(params.strike - spot, 0.0);
    };

    // Terminal spot prices and payoffs at maturity.
    std::vector<double> values(steps + 1);
    for (int i = 0; i <= steps; ++i) {
        const double spotAtNode = params.spot * std::pow(u, steps - i) * std::pow(d, i);
        values[i] = payoff(spotAtNode);
    }

    // Backward induction through the tree.
    for (int step = steps - 1; step >= 0; --step) {
        for (int i = 0; i <= step; ++i) {
            const double continuation = discount * (p * values[i] + (1.0 - p) * values[i + 1]);
            if (params.style == ExerciseStyle::American) {
                const double spotAtNode = params.spot * std::pow(u, step - i) * std::pow(d, i);
                values[i] = std::max(continuation, payoff(spotAtNode));
            } else {
                values[i] = continuation;
            }
        }
    }

    return values[0];
}

}  // namespace optionspricer::binomial_tree
