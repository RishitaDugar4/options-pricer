#include "optionspricer/MonteCarlo.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace optionspricer::monte_carlo {

namespace {

#if defined(__AVX2__)

constexpr int kLanes = 8;

// Single-precision exp() for 8 lanes at once. AVX2 has no native exp
// instruction, so this uses the standard Cephes range-reduction polynomial:
// exp(x) = 2^n * exp(g) with g kept in [-log(2)/2, log(2)/2] for the
// polynomial to stay accurate, then 2^n is built directly via IEEE-754
// exponent bit manipulation.
__m256 exp256Ps(__m256 x) {
    const __m256 expHi = _mm256_set1_ps(88.3762626647950f);
    const __m256 expLo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 expC1 = _mm256_set1_ps(0.693359375f);
    const __m256 expC2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);
    const __m256 one = _mm256_set1_ps(1.0f);

    x = _mm256_min_ps(x, expHi);
    x = _mm256_max_ps(x, expLo);

    __m256 fx = _mm256_add_ps(_mm256_mul_ps(x, log2ef), half);
    const __m256 flooredFx = _mm256_floor_ps(fx);
    const __m256 overshoot = _mm256_and_ps(_mm256_cmp_ps(flooredFx, fx, _CMP_GT_OS), one);
    fx = _mm256_sub_ps(flooredFx, overshoot);

    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, expC1));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, expC2));
    const __m256 z = _mm256_mul_ps(x, x);

    __m256 y = p0;
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p1);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p2);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p3);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p4);
    y = _mm256_add_ps(_mm256_mul_ps(y, x), p5);
    y = _mm256_add_ps(_mm256_mul_ps(y, z), x);
    y = _mm256_add_ps(y, one);

    // Build 2^n by writing n directly into the float exponent bits.
    __m256i exponent = _mm256_cvttps_epi32(fx);
    exponent = _mm256_add_epi32(exponent, _mm256_set1_epi32(0x7f));
    exponent = _mm256_slli_epi32(exponent, 23);
    const __m256 pow2n = _mm256_castsi256_ps(exponent);

    return _mm256_mul_ps(y, pow2n);
}

Result priceAvx2(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
    const float drift = static_cast<float>(
        (params.rate - params.dividendYield - 0.5 * params.volatility * params.volatility) *
        params.maturity);
    const float diffusion = static_cast<float>(params.volatility * std::sqrt(params.maturity));
    const float discount = static_cast<float>(std::exp(-params.rate * params.maturity));
    const float spot = static_cast<float>(params.spot);
    const float strike = static_cast<float>(params.strike);
    const float sign = params.type == OptionType::Call ? 1.0f : -1.0f;

    const __m256 driftVec = _mm256_set1_ps(drift);
    const __m256 diffusionVec = _mm256_set1_ps(diffusion);
    const __m256 spotVec = _mm256_set1_ps(spot);
    const __m256 strikeVec = _mm256_set1_ps(strike);
    const __m256 discountVec = _mm256_set1_ps(discount);
    const __m256 signVec = _mm256_set1_ps(sign);
    const __m256 zeroVec = _mm256_setzero_ps();

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Antithetic variates: pair each draw z with -z to reduce variance.
    // Each outer iteration draws 8 independent z's and simulates 8 "up"
    // paths (+z) and 8 "down" paths (-z) as two lane-wise AVX2 vectors.
    double sum = 0.0;
    double sumSquares = 0.0;
    std::uint64_t sampleCount = 0;
    const std::uint64_t totalPairs = (numPaths + 1) / 2;
    std::uint64_t pairsProcessed = 0;

    alignas(32) float zBuf[kLanes];
    alignas(32) float upBuf[kLanes];
    alignas(32) float downBuf[kLanes];

    while (pairsProcessed < totalPairs) {
        const std::uint64_t batchPairs = std::min<std::uint64_t>(kLanes, totalPairs - pairsProcessed);
        for (std::uint64_t lane = 0; lane < kLanes; ++lane) {
            zBuf[lane] = lane < batchPairs ? normal(rng) : 0.0f;
        }
        const __m256 z = _mm256_load_ps(zBuf);

        const __m256 expArgUp = _mm256_add_ps(driftVec, _mm256_mul_ps(diffusionVec, z));
        const __m256 expArgDown = _mm256_sub_ps(driftVec, _mm256_mul_ps(diffusionVec, z));
        const __m256 spotUp = _mm256_mul_ps(spotVec, exp256Ps(expArgUp));
        const __m256 spotDown = _mm256_mul_ps(spotVec, exp256Ps(expArgDown));

        const __m256 payoffUp =
            _mm256_max_ps(_mm256_mul_ps(signVec, _mm256_sub_ps(spotUp, strikeVec)), zeroVec);
        const __m256 payoffDown =
            _mm256_max_ps(_mm256_mul_ps(signVec, _mm256_sub_ps(spotDown, strikeVec)), zeroVec);

        _mm256_store_ps(upBuf, _mm256_mul_ps(discountVec, payoffUp));
        _mm256_store_ps(downBuf, _mm256_mul_ps(discountVec, payoffDown));

        for (std::uint64_t lane = 0; lane < batchPairs; ++lane) {
            const double discountedUp = upBuf[lane];
            sum += discountedUp;
            sumSquares += discountedUp * discountedUp;
            ++sampleCount;

            if (sampleCount < numPaths) {
                const double discountedDown = downBuf[lane];
                sum += discountedDown;
                sumSquares += discountedDown * discountedDown;
                ++sampleCount;
            }
        }
        pairsProcessed += batchPairs;
    }

    const double mean = sum / static_cast<double>(sampleCount);
    const double variance = sumSquares / static_cast<double>(sampleCount) - mean * mean;
    const double standardError = std::sqrt(std::max(variance, 0.0) / static_cast<double>(sampleCount));
    return Result{mean, standardError};
}

#else  // !defined(__AVX2__)

Result priceScalar(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
    const double drift =
        (params.rate - params.dividendYield - 0.5 * params.volatility * params.volatility) *
        params.maturity;
    const double diffusion = params.volatility * std::sqrt(params.maturity);
    const double discount = std::exp(-params.rate * params.maturity);

    auto payoff = [&](double spotAtMaturity) {
        return params.type == OptionType::Call ? std::max(spotAtMaturity - params.strike, 0.0)
                                                 : std::max(params.strike - spotAtMaturity, 0.0);
    };

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Antithetic variates: pair each draw z with -z to reduce variance.
    double sum = 0.0;
    double sumSquares = 0.0;
    std::uint64_t sampleCount = 0;
    const std::uint64_t pairs = (numPaths + 1) / 2;
    for (std::uint64_t i = 0; i < pairs; ++i) {
        const double z = normal(rng);
        const double spotUp = params.spot * std::exp(drift + diffusion * z);
        const double spotDown = params.spot * std::exp(drift - diffusion * z);

        const double discountedUp = discount * payoff(spotUp);
        sum += discountedUp;
        sumSquares += discountedUp * discountedUp;
        ++sampleCount;

        if (sampleCount < numPaths) {
            const double discountedDown = discount * payoff(spotDown);
            sum += discountedDown;
            sumSquares += discountedDown * discountedDown;
            ++sampleCount;
        }
    }

    const double mean = sum / static_cast<double>(sampleCount);
    const double variance = sumSquares / static_cast<double>(sampleCount) - mean * mean;
    const double standardError = std::sqrt(std::max(variance, 0.0) / static_cast<double>(sampleCount));
    return Result{mean, standardError};
}

#endif  // defined(__AVX2__)

}  // namespace

Result price(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
    params.validate();
    if (params.style != ExerciseStyle::European) {
        throw std::invalid_argument(
            "Monte Carlo terminal-payoff simulation only prices European-style options");
    }
    if (numPaths == 0) {
        throw std::invalid_argument("numPaths must be positive");
    }

#if defined(__AVX2__)
    return priceAvx2(params, numPaths, seed);
#else
    return priceScalar(params, numPaths, seed);
#endif
}

}  // namespace optionspricer::monte_carlo
