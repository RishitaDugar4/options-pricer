#include "optionspricer/BinomialTree.hpp"

#include <gtest/gtest.h>

#include "optionspricer/BlackScholes.hpp"

using namespace optionspricer;

namespace {

OptionParams baseParams(OptionType type, ExerciseStyle style) {
    OptionParams p;
    p.type = type;
    p.style = style;
    p.spot = 100.0;
    p.strike = 100.0;
    p.rate = 0.05;
    p.dividendYield = 0.0;
    p.volatility = 0.2;
    p.maturity = 1.0;
    return p;
}

}  // namespace

TEST(BinomialTree, EuropeanConvergesToBlackScholes) {
    OptionParams call = baseParams(OptionType::Call, ExerciseStyle::European);
    OptionParams put = baseParams(OptionType::Put, ExerciseStyle::European);

    const double bsCall = black_scholes::price(call);
    const double bsPut = black_scholes::price(put);

    EXPECT_NEAR(binomial_tree::price(call, 1000), bsCall, 0.05);
    EXPECT_NEAR(binomial_tree::price(put, 1000), bsPut, 0.05);
}

TEST(BinomialTree, AmericanCallWithNoDividendEqualsEuropean) {
    // Early exercise of a call is never optimal without dividends.
    OptionParams american = baseParams(OptionType::Call, ExerciseStyle::American);
    OptionParams european = baseParams(OptionType::Call, ExerciseStyle::European);
    EXPECT_NEAR(binomial_tree::price(american, 500), binomial_tree::price(european, 500), 1e-6);
}

TEST(BinomialTree, AmericanPutIsAtLeastAsValuableAsEuropean) {
    OptionParams american = baseParams(OptionType::Put, ExerciseStyle::American);
    OptionParams european = baseParams(OptionType::Put, ExerciseStyle::European);
    EXPECT_GE(binomial_tree::price(american, 500), binomial_tree::price(european, 500) - 1e-9);
}

TEST(BinomialTree, DeepInTheMoneyAmericanPutExceedsIntrinsicOrEqualsIt) {
    OptionParams p = baseParams(OptionType::Put, ExerciseStyle::American);
    p.spot = 50.0;
    const double intrinsic = p.strike - p.spot;
    EXPECT_GE(binomial_tree::price(p, 500), intrinsic - 1e-6);
}

TEST(BinomialTree, InvalidStepsThrows) {
    OptionParams p = baseParams(OptionType::Call, ExerciseStyle::European);
    EXPECT_THROW(binomial_tree::price(p, 0), std::invalid_argument);
}
