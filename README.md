# options-pricer

A C++ options pricing library and CLI supporting three models:

- **Black-Scholes** — closed-form price and Greeks (delta, gamma, vega, theta, rho) for European options.
- **Binomial tree** (Cox-Ross-Rubinstein) — European and American exercise.
- **Monte Carlo** — GBM path simulation with antithetic variance reduction, European exercise.

## Build

Requires CMake 3.16+ and a C++20 compiler. GoogleTest is fetched automatically for the test suite.

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

## Layout

```
include/optionspricer/   public headers (Option, BlackScholes, BinomialTree, MonteCarlo)
src/                      library implementation
apps/                     CLI entry point
tests/                    GoogleTest unit tests
```
