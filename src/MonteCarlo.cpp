#include "optionspricer/MonteCarlo.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#define OPTIONSPRICER_MC_AVX2 1
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define OPTIONSPRICER_MC_NEON 1
#endif

namespace optionspricer::monte_carlo {

namespace {

#if defined(OPTIONSPRICER_MC_AVX2)

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

#elif defined(OPTIONSPRICER_MC_NEON)

constexpr int kLanes = 8;

// Single-precision exp() for 4 lanes at once (NEON has no native exp either).
// Same Cephes range-reduction polynomial as the AVX2 path above, just built
// from NEON intrinsics: exp(x) = 2^n * exp(g) with g in
// [-log(2)/2, log(2)/2], and 2^n written directly into the float exponent
// bits.
float32x4_t exp4Ps(float32x4_t x) {
    const float32x4_t expHi = vdupq_n_f32(88.3762626647950f);
    const float32x4_t expLo = vdupq_n_f32(-88.3762626647949f);
    const float32x4_t log2ef = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t expC1 = vdupq_n_f32(0.693359375f);
    const float32x4_t expC2 = vdupq_n_f32(-2.12194440e-4f);
    const float32x4_t p0 = vdupq_n_f32(1.9875691500e-4f);
    const float32x4_t p1 = vdupq_n_f32(1.3981999507e-3f);
    const float32x4_t p2 = vdupq_n_f32(8.3334519073e-3f);
    const float32x4_t p3 = vdupq_n_f32(4.1665795894e-2f);
    const float32x4_t p4 = vdupq_n_f32(1.6666665459e-1f);
    const float32x4_t p5 = vdupq_n_f32(5.0000001201e-1f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);

    x = vminq_f32(x, expHi);
    x = vmaxq_f32(x, expLo);

    float32x4_t fx = vaddq_f32(vmulq_f32(x, log2ef), half);
    const float32x4_t flooredFx = vrndmq_f32(fx);
    const uint32x4_t gtMask = vcgtq_f32(flooredFx, fx);
    const float32x4_t overshoot = vbslq_f32(gtMask, one, zero);
    fx = vsubq_f32(flooredFx, overshoot);

    x = vsubq_f32(x, vmulq_f32(fx, expC1));
    x = vsubq_f32(x, vmulq_f32(fx, expC2));
    const float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = p0;
    y = vaddq_f32(vmulq_f32(y, x), p1);
    y = vaddq_f32(vmulq_f32(y, x), p2);
    y = vaddq_f32(vmulq_f32(y, x), p3);
    y = vaddq_f32(vmulq_f32(y, x), p4);
    y = vaddq_f32(vmulq_f32(y, x), p5);
    y = vaddq_f32(vmulq_f32(y, z), x);
    y = vaddq_f32(y, one);

    // Build 2^n by writing n directly into the float exponent bits.
    int32x4_t exponent = vcvtq_s32_f32(fx);
    exponent = vaddq_s32(exponent, vdupq_n_s32(0x7f));
    exponent = vshlq_n_s32(exponent, 23);
    const float32x4_t pow2n = vreinterpretq_f32_s32(exponent);

    return vmulq_f32(y, pow2n);
}

// Simulates one 4-lane half-batch: spot * exp(drift + diffusionSigned*z),
// then payoff, then discount. Passing diffusionVec negated computes the
// antithetic ("down") leg with the same code path as the "up" leg.
float32x4_t simulateHalfBatch(float32x4_t z, float32x4_t driftVec, float32x4_t diffusionSignedVec,
                               float32x4_t spotVec, float32x4_t strikeVec, float32x4_t signVec,
                               float32x4_t discountVec, float32x4_t zeroVec) {
    const float32x4_t expArg = vaddq_f32(driftVec, vmulq_f32(diffusionSignedVec, z));
    const float32x4_t spotAtMaturity = vmulq_f32(spotVec, exp4Ps(expArg));
    const float32x4_t payoff =
        vmaxq_f32(vmulq_f32(signVec, vsubq_f32(spotAtMaturity, strikeVec)), zeroVec);
    return vmulq_f32(discountVec, payoff);
}

#endif  // OPTIONSPRICER_MC_AVX2 / OPTIONSPRICER_MC_NEON

}  // namespace

namespace detail {

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

#if defined(OPTIONSPRICER_MC_AVX2)

Result priceSimd(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
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

#elif defined(OPTIONSPRICER_MC_NEON)

Result priceSimd(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
    const float drift = static_cast<float>(
        (params.rate - params.dividendYield - 0.5 * params.volatility * params.volatility) *
        params.maturity);
    const float diffusion = static_cast<float>(params.volatility * std::sqrt(params.maturity));
    const float discount = static_cast<float>(std::exp(-params.rate * params.maturity));
    const float spot = static_cast<float>(params.spot);
    const float strike = static_cast<float>(params.strike);
    const float sign = params.type == OptionType::Call ? 1.0f : -1.0f;

    const float32x4_t driftVec = vdupq_n_f32(drift);
    const float32x4_t diffusionVec = vdupq_n_f32(diffusion);
    const float32x4_t negDiffusionVec = vdupq_n_f32(-diffusion);
    const float32x4_t spotVec = vdupq_n_f32(spot);
    const float32x4_t strikeVec = vdupq_n_f32(strike);
    const float32x4_t discountVec = vdupq_n_f32(discount);
    const float32x4_t signVec = vdupq_n_f32(sign);
    const float32x4_t zeroVec = vdupq_n_f32(0.0f);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Antithetic variates: pair each draw z with -z to reduce variance.
    // Each outer iteration draws 8 independent z's and simulates 8 "up"
    // paths (+z) and 8 "down" paths (-z) as two NEON 4-lane half-batches.
    double sum = 0.0;
    double sumSquares = 0.0;
    std::uint64_t sampleCount = 0;
    const std::uint64_t totalPairs = (numPaths + 1) / 2;
    std::uint64_t pairsProcessed = 0;

    alignas(16) float zBuf[kLanes];
    alignas(16) float upBuf[kLanes];
    alignas(16) float downBuf[kLanes];

    while (pairsProcessed < totalPairs) {
        const std::uint64_t batchPairs = std::min<std::uint64_t>(kLanes, totalPairs - pairsProcessed);
        for (std::uint64_t lane = 0; lane < kLanes; ++lane) {
            zBuf[lane] = lane < batchPairs ? normal(rng) : 0.0f;
        }
        const float32x4_t zLo = vld1q_f32(zBuf);
        const float32x4_t zHi = vld1q_f32(zBuf + 4);

        const float32x4_t upLo = simulateHalfBatch(zLo, driftVec, diffusionVec, spotVec, strikeVec,
                                                    signVec, discountVec, zeroVec);
        const float32x4_t upHi = simulateHalfBatch(zHi, driftVec, diffusionVec, spotVec, strikeVec,
                                                    signVec, discountVec, zeroVec);
        const float32x4_t downLo = simulateHalfBatch(zLo, driftVec, negDiffusionVec, spotVec,
                                                      strikeVec, signVec, discountVec, zeroVec);
        const float32x4_t downHi = simulateHalfBatch(zHi, driftVec, negDiffusionVec, spotVec,
                                                      strikeVec, signVec, discountVec, zeroVec);

        vst1q_f32(upBuf, upLo);
        vst1q_f32(upBuf + 4, upHi);
        vst1q_f32(downBuf, downLo);
        vst1q_f32(downBuf + 4, downHi);

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

#else  // no SIMD available on this target

Result priceSimd(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
    return priceScalar(params, numPaths, seed);
}

#endif  // OPTIONSPRICER_MC_AVX2 / OPTIONSPRICER_MC_NEON

}  // namespace detail

bool hasSimdMonteCarlo() {
#if defined(OPTIONSPRICER_MC_AVX2) || defined(OPTIONSPRICER_MC_NEON)
    return true;
#else
    return false;
#endif
}

Result price(const OptionParams& params, std::uint64_t numPaths, std::uint64_t seed) {
    params.validate();
    if (params.style != ExerciseStyle::European) {
        throw std::invalid_argument(
            "Monte Carlo terminal-payoff simulation only prices European-style options");
    }
    if (numPaths == 0) {
        throw std::invalid_argument("numPaths must be positive");
    }

    return detail::priceSimd(params, numPaths, seed);
}

}  // namespace optionspricer::monte_carlo
