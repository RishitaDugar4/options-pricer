#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>

#include "optionspricer/BlackScholes.hpp"
#include "optionspricer/MonteCarlo.hpp"

using namespace optionspricer;

namespace {

OptionParams callParams() {
    OptionParams p;
    p.type = OptionType::Call;
    p.style = ExerciseStyle::European;
    p.spot = 100.0;
    p.strike = 100.0;
    p.rate = 0.05;
    p.volatility = 0.2;
    p.maturity = 1.0;
    return p;
}

}  // namespace

// Tracks time, statistical standard error, and absolute error vs. the
// closed-form Black-Scholes price as path count grows, showing the
// accuracy/speed tradeoff (error shrinks like 1/sqrt(numPaths), cost grows
// like numPaths).
void BM_MonteCarloConvergence(benchmark::State& state) {
    const auto numPaths = static_cast<std::uint64_t>(state.range(0));
    const OptionParams params = callParams();
    const double reference = black_scholes::price(params);

    monte_carlo::Result result;
    for (auto _ : state) {
        result = monte_carlo::price(params, numPaths, /*seed=*/42);
        benchmark::DoNotOptimize(result);
    }
    state.counters["Paths"] = static_cast<double>(numPaths);
    state.counters["AbsError"] = std::abs(result.price - reference);
    state.counters["StdError"] = result.standardError;
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(numPaths));
}
BENCHMARK(BM_MonteCarloConvergence)
    ->RangeMultiplier(10)
    ->Range(1000, 10000000)
    ->Unit(benchmark::kMillisecond);
