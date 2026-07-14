#include "optionspricer/BlackScholes.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace optionspricer;

namespace {

OptionParams atmParams(OptionType type) {
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

TEST(BlackScholes, AtmCallMatchesKnownValue) {
    // Textbook reference value for S=K=100, r=5%, sigma=20%, T=1.
    EXPECT_NEAR(black_scholes::price(atmParams(OptionType::Call)), 10.4506, 0.01);
}

TEST(BlackScholes, AtmPutMatchesKnownValue) {
    EXPECT_NEAR(black_scholes::price(atmParams(OptionType::Put)), 5.5735, 0.01);
}

TEST(BlackScholes, PutCallParityHolds) {
    OptionParams call = atmParams(OptionType::Call);
    OptionParams put = atmParams(OptionType::Put);
    const double c = black_scholes::price(call);
    const double p = black_scholes::price(put);
    const double lhs = c - p;
    const double rhs = call.spot * std::exp(-call.dividendYield * call.maturity) -
                        call.strike * std::exp(-call.rate * call.maturity);
    EXPECT_NEAR(lhs, rhs, 1e-9);
}

TEST(BlackScholes, DeepInTheMoneyCallApproachesIntrinsic) {
    OptionParams p = atmParams(OptionType::Call);
    p.spot = 1000.0;
    const double intrinsic = p.spot - p.strike * std::exp(-p.rate * p.maturity);
    EXPECT_NEAR(black_scholes::price(p), intrinsic, 1.0);
}

TEST(BlackScholes, DeepOutOfTheMoneyCallIsNearZero) {
    OptionParams p = atmParams(OptionType::Call);
    p.spot = 1.0;
    EXPECT_NEAR(black_scholes::price(p), 0.0, 1e-6);
}

TEST(BlackScholes, AmericanStyleThrows) {
    OptionParams p = atmParams(OptionType::Call);
    p.style = ExerciseStyle::American;
    EXPECT_THROW(black_scholes::price(p), std::invalid_argument);
}

TEST(BlackScholes, GreeksAreInExpectedRanges) {
    const Greeks callGreeks = black_scholes::greeks(atmParams(OptionType::Call));
    EXPECT_GT(callGreeks.delta, 0.0);
    EXPECT_LT(callGreeks.delta, 1.0);
    EXPECT_GT(callGreeks.gamma, 0.0);
    EXPECT_GT(callGreeks.vega, 0.0);

    const Greeks putGreeks = black_scholes::greeks(atmParams(OptionType::Put));
    EXPECT_LT(putGreeks.delta, 0.0);
    EXPECT_GT(putGreeks.delta, -1.0);
}

TEST(BlackScholes, InvalidParamsThrow) {
    OptionParams p = atmParams(OptionType::Call);
    p.spot = -1.0;
    EXPECT_THROW(black_scholes::price(p), std::invalid_argument);
}
