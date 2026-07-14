#pragma once

#include <stdexcept>

namespace optionspricer {

enum class OptionType { Call, Put };
enum class ExerciseStyle { European, American };

// Parameters shared by all pricing models.
struct OptionParams {
    OptionType type = OptionType::Call;
    ExerciseStyle style = ExerciseStyle::European;
    double spot = 100.0;         // S: current underlying price
    double strike = 100.0;       // K: strike price
    double rate = 0.05;          // r: risk-free rate (annualized, continuously compounded)
    double dividendYield = 0.0;  // q: continuous dividend yield (annualized)
    double volatility = 0.2;     // sigma: annualized volatility
    double maturity = 1.0;       // T: time to expiry in years

    void validate() const {
        if (spot <= 0.0) throw std::invalid_argument("spot must be positive");
        if (strike <= 0.0) throw std::invalid_argument("strike must be positive");
        if (volatility <= 0.0) throw std::invalid_argument("volatility must be positive");
        if (maturity <= 0.0) throw std::invalid_argument("maturity must be positive");
    }
};

struct Greeks {
    double delta = 0.0;
    double gamma = 0.0;
    double vega = 0.0;   // sensitivity to a 1.00 (100%) change in volatility
    double theta = 0.0;  // sensitivity to a 1 year decrease in maturity
    double rho = 0.0;    // sensitivity to a 1.00 (100%) change in rate
};

}  // namespace optionspricer
