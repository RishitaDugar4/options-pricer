#include <benchmark/benchmark.h>

#include "optionspricer/BlackScholes.hpp"

using namespace optionspricer;

namespace {

OptionParams atmParams(OptionType type) {
    OptionParams p;
    p.type = type;
    p.style = ExerciseStyle::European;
    p.spot = 100.0;
    p.strike = 100.0;
    p.rate = 0.05;
    p.volatility = 0.2;
    p.maturity = 1.0;
    return p;
}

}  // namespace

void BM_BlackScholesPriceCall(benchmark::State& state) {
    const OptionParams params = atmParams(OptionType::Call);
    for (auto _ : state) {
        benchmark::DoNotOptimize(black_scholes::price(params));
    }
}
BENCHMARK(BM_BlackScholesPriceCall);

void BM_BlackScholesPricePut(benchmark::State& state) {
    const OptionParams params = atmParams(OptionType::Put);
    for (auto _ : state) {
        benchmark::DoNotOptimize(black_scholes::price(params));
    }
}
BENCHMARK(BM_BlackScholesPricePut);

void BM_BlackScholesGreeks(benchmark::State& state) {
    const OptionParams params = atmParams(OptionType::Call);
    for (auto _ : state) {
        benchmark::DoNotOptimize(black_scholes::greeks(params));
    }
}
BENCHMARK(BM_BlackScholesGreeks);
