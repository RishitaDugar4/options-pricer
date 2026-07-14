#include "optionspricer/BlackScholes.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace optionspricer::black_scholes {

namespace {

double normalCdf(double x) { return 0.5 * std::erfc(-x / std::numbers::sqrt2); }

double normalPdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
}

struct D1D2 {
    double d1;
    double d2;
};

D1D2 computeD1D2(const OptionParams& p) {
    const double sqrtT = std::sqrt(p.maturity);
    const double d1 = (std::log(p.spot / p.strike) +
                        (p.rate - p.dividendYield + 0.5 * p.volatility * p.volatility) *
                            p.maturity) /
                       (p.volatility * sqrtT);
    const double d2 = d1 - p.volatility * sqrtT;
    return {d1, d2};
}

void requireEuropean(const OptionParams& p) {
    if (p.style != ExerciseStyle::European) {
        throw std::invalid_argument(
            "Black-Scholes only prices European-style options; use binomial_tree for "
            "American exercise");
    }
}

}  // namespace

double price(const OptionParams& params) {
    params.validate();
    requireEuropean(params);

    const auto [d1, d2] = computeD1D2(params);
    const double discountedSpot = params.spot * std::exp(-params.dividendYield * params.maturity);
    const double discountedStrike = params.strike * std::exp(-params.rate * params.maturity);

    if (params.type == OptionType::Call) {
        return discountedSpot * normalCdf(d1) - discountedStrike * normalCdf(d2);
    }
    return discountedStrike * normalCdf(-d2) - discountedSpot * normalCdf(-d1);
}

Greeks greeks(const OptionParams& params) {
    params.validate();
    requireEuropean(params);

    const auto [d1, d2] = computeD1D2(params);
    const double sqrtT = std::sqrt(params.maturity);
    const double discountedSpot = params.spot * std::exp(-params.dividendYield * params.maturity);
    const double discountedStrike = params.strike * std::exp(-params.rate * params.maturity);
    const double pdf1 = normalPdf(d1);

    Greeks g;
    g.gamma = std::exp(-params.dividendYield * params.maturity) * pdf1 /
              (params.spot * params.volatility * sqrtT);
    g.vega = discountedSpot * pdf1 * sqrtT;

    if (params.type == OptionType::Call) {
        g.delta = std::exp(-params.dividendYield * params.maturity) * normalCdf(d1);
        g.theta = -discountedSpot * pdf1 * params.volatility / (2.0 * sqrtT) -
                  params.rate * discountedStrike * normalCdf(d2) +
                  params.dividendYield * discountedSpot * normalCdf(d1);
        g.rho = params.strike * params.maturity * std::exp(-params.rate * params.maturity) *
                normalCdf(d2);
    } else {
        g.delta = -std::exp(-params.dividendYield * params.maturity) * normalCdf(-d1);
        g.theta = -discountedSpot * pdf1 * params.volatility / (2.0 * sqrtT) +
                  params.rate * discountedStrike * normalCdf(-d2) -
                  params.dividendYield * discountedSpot * normalCdf(-d1);
        g.rho = -params.strike * params.maturity * std::exp(-params.rate * params.maturity) *
                normalCdf(-d2);
    }
    return g;
}

}  // namespace optionspricer::black_scholes
