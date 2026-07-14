#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>

#include "optionspricer/BinomialTree.hpp"
#include "optionspricer/BlackScholes.hpp"
#include "optionspricer/MonteCarlo.hpp"
#include "optionspricer/Option.hpp"

namespace {

using optionspricer::ExerciseStyle;
using optionspricer::OptionParams;
using optionspricer::OptionType;

void printUsage(std::string_view argv0) {
    std::cout <<
        "Usage: " << argv0 << " [options]\n\n"
        "Options:\n"
        "  --model MODEL       bs | binomial | mc            (default: bs)\n"
        "  --type TYPE         call | put                    (default: call)\n"
        "  --style STYLE       european | american            (default: european)\n"
        "  --spot S            underlying spot price          (default: 100)\n"
        "  --strike K          strike price                   (default: 100)\n"
        "  --rate R            risk-free rate                 (default: 0.05)\n"
        "  --dividend Q        continuous dividend yield       (default: 0)\n"
        "  --vol SIGMA         volatility                      (default: 0.2)\n"
        "  --maturity T        time to expiry in years         (default: 1.0)\n"
        "  --steps N           binomial tree steps             (default: 500)\n"
        "  --paths N           monte carlo paths               (default: 100000)\n"
        "  --seed N            monte carlo RNG seed             (default: 42)\n"
        "  --greeks            also print Black-Scholes greeks (bs model only)\n"
        "  --help               show this message\n";
}

// Parses "--flag value" pairs into a lookup table.
std::unordered_map<std::string, std::string> parseArgs(int argc, char** argv,
                                                         bool& showGreeks, bool& showHelp) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            showHelp = true;
            continue;
        }
        if (flag == "--greeks") {
            showGreeks = true;
            continue;
        }
        if (flag.rfind("--", 0) != 0 || i + 1 >= argc) {
            throw std::invalid_argument("unrecognized argument: " + flag);
        }
        args[flag.substr(2)] = argv[++i];
    }
    return args;
}

double getDouble(const std::unordered_map<std::string, std::string>& args, const std::string& key,
                  double defaultValue) {
    auto it = args.find(key);
    if (it == args.end()) return defaultValue;
    return std::stod(it->second);
}

int getInt(const std::unordered_map<std::string, std::string>& args, const std::string& key,
           int defaultValue) {
    auto it = args.find(key);
    if (it == args.end()) return defaultValue;
    return std::stoi(it->second);
}

std::uint64_t getUInt64(const std::unordered_map<std::string, std::string>& args,
                         const std::string& key, std::uint64_t defaultValue) {
    auto it = args.find(key);
    if (it == args.end()) return defaultValue;
    return std::stoull(it->second);
}

std::string getString(const std::unordered_map<std::string, std::string>& args,
                       const std::string& key, std::string defaultValue) {
    auto it = args.find(key);
    if (it == args.end()) return defaultValue;
    return it->second;
}

OptionType parseType(const std::string& s) {
    if (s == "call") return OptionType::Call;
    if (s == "put") return OptionType::Put;
    throw std::invalid_argument("--type must be 'call' or 'put'");
}

ExerciseStyle parseStyle(const std::string& s) {
    if (s == "european") return ExerciseStyle::European;
    if (s == "american") return ExerciseStyle::American;
    throw std::invalid_argument("--style must be 'european' or 'american'");
}

}  // namespace

int main(int argc, char** argv) {
    bool showGreeks = false;
    bool showHelp = false;
    std::unordered_map<std::string, std::string> args;
    try {
        args = parseArgs(argc, argv, showGreeks, showHelp);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (showHelp) {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }

    try {
        OptionParams params;
        params.type = parseType(getString(args, "type", "call"));
        params.style = parseStyle(getString(args, "style", "european"));
        params.spot = getDouble(args, "spot", 100.0);
        params.strike = getDouble(args, "strike", 100.0);
        params.rate = getDouble(args, "rate", 0.05);
        params.dividendYield = getDouble(args, "dividend", 0.0);
        params.volatility = getDouble(args, "vol", 0.2);
        params.maturity = getDouble(args, "maturity", 1.0);
        params.validate();

        const std::string model = getString(args, "model", "bs");

        std::cout << std::fixed << std::setprecision(6);

        if (model == "bs") {
            const double p = optionspricer::black_scholes::price(params);
            std::cout << "Black-Scholes price: " << p << "\n";
            if (showGreeks) {
                const auto g = optionspricer::black_scholes::greeks(params);
                std::cout << "  delta: " << g.delta << "\n"
                          << "  gamma: " << g.gamma << "\n"
                          << "  vega:  " << g.vega << "\n"
                          << "  theta: " << g.theta << "\n"
                          << "  rho:   " << g.rho << "\n";
            }
        } else if (model == "binomial") {
            const int steps = getInt(args, "steps", 500);
            const double p = optionspricer::binomial_tree::price(params, steps);
            std::cout << "Binomial tree price (" << steps << " steps): " << p << "\n";
        } else if (model == "mc") {
            const std::uint64_t paths = getUInt64(args, "paths", 100000);
            const std::uint64_t seed = getUInt64(args, "seed", 42);
            const auto result = optionspricer::monte_carlo::price(params, paths, seed);
            std::cout << "Monte Carlo price (" << paths << " paths): " << result.price
                       << " +/- " << result.standardError << "\n";
        } else {
            std::cerr << "Error: unknown --model '" << model << "' (expected bs, binomial, mc)\n";
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
