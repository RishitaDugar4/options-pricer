include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(optionspricer_benchmarks
  main.cpp
  bench_black_scholes.cpp
  bench_binomial_tree.cpp
  bench_monte_carlo.cpp
  bench_monte_carlo_simd.cpp
)

target_link_libraries(optionspricer_benchmarks PRIVATE optionspricer benchmark::benchmark)
target_compile_options(optionspricer_benchmarks PRIVATE -Wall -Wextra)
