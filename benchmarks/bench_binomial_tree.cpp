#include <benchmark/benchmark.h>

#include <cmath>

#include "optionspricer/BinomialTree.hpp"
#include "optionspricer/BlackScholes.hpp"

using namespace optionspricer;

namespace {

OptionParams params(ExerciseStyle style, OptionType type) {
    OptionParams p;
    p.type = type;
    p.style = style;
    p.spot = 100.0;
    p.strike = 100.0;
    p.rate = 0.05;
    p.volatility = 0.2;
    p.maturity = 1.0;
    return p;
}

}  // namespace

// Tracks time and absolute error vs. the closed-form Black-Scholes price as
// the tree deepens, showing the classic O(1/steps) convergence / runtime
// tradeoff for a European option (where a reference price exists).
void BM_BinomialTreeEuropeanConvergence(benchmark::State& state) {
    const int steps = static_cast<int>(state.range(0));
    const OptionParams european = params(ExerciseStyle::European, OptionType::Put);
    const double reference = black_scholes::price(european);

    double price = 0.0;
    for (auto _ : state) {
        price = binomial_tree::price(european, steps);
        benchmark::DoNotOptimize(price);
    }
    state.counters["Steps"] = static_cast<double>(steps);
    state.counters["AbsError"] = std::abs(price - reference);
}
BENCHMARK(BM_BinomialTreeEuropeanConvergence)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000);

// American exercise has no closed form to diff against; this just tracks
// how tree cost scales with steps for the early-exercise code path.
void BM_BinomialTreeAmerican(benchmark::State& state) {
    const int steps = static_cast<int>(state.range(0));
    const OptionParams american = params(ExerciseStyle::American, OptionType::Put);

    for (auto _ : state) {
        benchmark::DoNotOptimize(binomial_tree::price(american, steps));
    }
    state.counters["Steps"] = static_cast<double>(steps);
}
BENCHMARK(BM_BinomialTreeAmerican)->Arg(10)->Arg(50)->Arg(100)->Arg(500)->Arg(1000)->Arg(5000);
