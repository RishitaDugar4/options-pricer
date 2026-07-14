#pragma once

#include "optionspricer/Option.hpp"

namespace optionspricer::binomial_tree {

// Cox-Ross-Rubinstein binomial tree price. Supports both European and
// American exercise styles. `steps` is the number of time steps in the tree.
double price(const OptionParams& params, int steps = 500);

}  // namespace optionspricer::binomial_tree
