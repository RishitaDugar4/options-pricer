#include <benchmark/benchmark.h>

#include <cstdint>

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

// Head-to-head: detail::priceScalar always runs the double-precision scalar
// loop; detail::priceSimd runs whatever this build compiled (AVX2 on
// x86_64, NEON on arm64, or the scalar loop again if neither is available
// -- check the "monte_carlo_simd" custom context line to tell which).
void BM_MonteCarloScalar(benchmark::State& state) {
    const auto numPaths = static_cast<std::uint64_t>(state.range(0));
    const OptionParams params = callParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(monte_carlo::detail::priceScalar(params, numPaths, /*seed=*/42));
    }
    state.counters["Paths"] = static_cast<double>(numPaths);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(numPaths));
}
BENCHMARK(BM_MonteCarloScalar)->RangeMultiplier(10)->Range(1000, 1000000)->Unit(benchmark::kMillisecond);

void BM_MonteCarloSimd(benchmark::State& state) {
    const auto numPaths = static_cast<std::uint64_t>(state.range(0));
    const OptionParams params = callParams();
    for (auto _ : state) {
        benchmark::DoNotOptimize(monte_carlo::detail::priceSimd(params, numPaths, /*seed=*/42));
    }
    state.counters["Paths"] = static_cast<double>(numPaths);
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(numPaths));
}
BENCHMARK(BM_MonteCarloSimd)->RangeMultiplier(10)->Range(1000, 1000000)->Unit(benchmark::kMillisecond);
