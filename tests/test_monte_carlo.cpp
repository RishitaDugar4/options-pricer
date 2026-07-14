#include "optionspricer/MonteCarlo.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "optionspricer/BlackScholes.hpp"

using namespace optionspricer;

namespace {

OptionParams baseParams(OptionType type) {
    OptionParams p;
    p.type = type;
    p.style = ExerciseStyle::European;
    p.spot = 100.0;
    p.strike = 100.0;
    p.rate = 0.05;
    p.dividendYield = 0.0;
    p.volatility = 0.2;
    p.maturity = 1.0;
    return p;
}

}  // namespace

TEST(MonteCarlo, ConvergesToBlackScholesWithinSeveralStandardErrors) {
    OptionParams call = baseParams(OptionType::Call);
    const double bsCall = black_scholes::price(call);

    const auto result = monte_carlo::price(call, 200000, /*seed=*/7);
    EXPECT_NEAR(result.price, bsCall, 6.0 * result.standardError);
}

TEST(MonteCarlo, IsDeterministicForFixedSeed) {
    OptionParams put = baseParams(OptionType::Put);
    const auto first = monte_carlo::price(put, 10000, /*seed=*/123);
    const auto second = monte_carlo::price(put, 10000, /*seed=*/123);
    EXPECT_DOUBLE_EQ(first.price, second.price);
}

TEST(MonteCarlo, AmericanStyleThrows) {
    OptionParams p = baseParams(OptionType::Call);
    p.style = ExerciseStyle::American;
    EXPECT_THROW(monte_carlo::price(p), std::invalid_argument);
}

TEST(MonteCarlo, ZeroPathsThrows) {
    OptionParams p = baseParams(OptionType::Call);
    EXPECT_THROW(monte_carlo::price(p, 0), std::invalid_argument);
}

TEST(MonteCarlo, StandardErrorShrinksWithMorePaths) {
    OptionParams p = baseParams(OptionType::Call);
    const auto few = monte_carlo::price(p, 1000, /*seed=*/1);
    const auto many = monte_carlo::price(p, 100000, /*seed=*/1);
    EXPECT_LT(many.standardError, few.standardError);
}
