#include "experiment.hpp"
#include "functions.hpp"

void batch_exp(std::string exp_name) {
    const size_t numIterations = 500; // how many times we repeat computation to eliminate noise/random deviations
    const size_t numSamples = 3000;   // number of items in estimator
    const size_t maxNumSpecs = 10;
    const size_t numTargets = 1;
    const size_t numAttackers = 1;

    using in_type = long; // specific to distribution

    vector<in_type> N_vals{4, 8, 16};
    // distribution-specific parameters, will vary in each iteration
    for (const auto &N : N_vals) {
        const in_type range_from = 0;
        const in_type range_to = N - 1;
        std::uniform_int_distribution<in_type> dist(range_from, range_to);

        if (exp_name == "max") {

            using out_type = long; // specific to the FUNCTION

            discrete_exp<in_type, out_type, std::uniform_int_distribution>(
                exp_name,
                dist,
                static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_max<in_type>),
                range_from,
                range_to,
                numSamples,
                numIterations,
                numTargets,
                numAttackers,
                maxNumSpecs);
        } else if (exp_name == "min") {

            using out_type = long; // specific to the FUNCTION

            discrete_exp<in_type, out_type, std::uniform_int_distribution>(
                exp_name,
                dist,
                static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_min<in_type>),
                range_from,
                range_to,
                numSamples,
                numIterations,
                numTargets,
                numAttackers,
                maxNumSpecs);
        } else if (exp_name == "var") {
            using out_type = double; // specific to the FUNCTION
            discrete_exp<in_type, out_type, std::uniform_int_distribution>(
                exp_name,
                dist,
                static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_var<in_type>),
                // compute_median,
                range_from,
                range_to,
                numSamples,
                numIterations,
                numTargets,
                numAttackers,
                maxNumSpecs);
        }
        else if (exp_name == "median") {
            using out_type = double; // specific to the FUNCTION
            discrete_exp<in_type, out_type, std::uniform_int_distribution>(
                exp_name,
                dist,
                static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_median<in_type>),
                // compute_median,
                range_from,
                range_to,
                numSamples,
                numIterations,
                numTargets,
                numAttackers,
                maxNumSpecs);
        }
    }
}
