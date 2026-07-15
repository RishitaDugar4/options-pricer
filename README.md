# options-pricer

A C++ options pricing library and CLI supporting three models:

- **Black-Scholes** — closed-form price and Greeks (delta, gamma, vega, theta, rho) for European options.
- **Binomial tree** (Cox-Ross-Rubinstein) — European and American exercise.
- **Monte Carlo** — GBM path simulation with antithetic variance reduction, European exercise.

## Build

Requires CMake 3.16+ and a C++20 compiler. GoogleTest and Google Benchmark are fetched automatically for the test and benchmark suites.

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/apps/options_pricer --model bs --type call --spot 100 --strike 100 --rate 0.05 --vol 0.2 --maturity 1 --greeks
./build/apps/options_pricer --model binomial --style american --type put --steps 1000
./build/apps/options_pricer --model mc --paths 200000
```

Run `--help` for the full list of options.

## Test

```sh
ctest --test-dir build --output-on-failure
```

## Benchmark

```sh
./build/benchmarks/optionspricer_benchmarks
```

Covers:
- `BM_BlackScholesPrice*` / `BM_BlackScholesGreeks` — raw closed-form pricing speed (nanoseconds).
- `BM_BinomialTreeEuropeanConvergence` — time and absolute error vs. Black-Scholes as step count grows.
- `BM_BinomialTreeAmerican` — tree cost scaling for early-exercise pricing (no closed form to diff against).
- `BM_MonteCarloConvergence` — time, standard error, and absolute error vs. Black-Scholes as path count grows.
- `BM_MonteCarloScalar` vs. `BM_MonteCarloSimd` — head-to-head scalar vs. vectorized (AVX2/NEON) path simulation at matched path counts. The `monte_carlo_simd` line in the output banner reports which vectorized backend (if any) this build compiled.

Pass `--benchmark_filter=Regex` to run a subset, or `--benchmark_format=json` for machine-readable output. See `--help` for the full flag list.

## Layout

```
include/optionspricer/   public headers (Option, BlackScholes, BinomialTree, MonteCarlo)
src/                      library implementation
apps/                     CLI entry point
tests/                    GoogleTest unit tests
benchmarks/               Google Benchmark suite
```
