#include <benchmark/benchmark.h>

#include "optionspricer/MonteCarlo.hpp"

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::AddCustomContext("monte_carlo_simd", optionspricer::monte_carlo::hasSimdMonteCarlo()
                                                          ? "vectorized (AVX2/NEON)"
                                                          : "scalar fallback");
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
