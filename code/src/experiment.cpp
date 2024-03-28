#include "experiment.hpp"
#include "functions.hpp"

void batch_exp_discrete_uniform(std::string exp_name) {
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
                range_from,
                range_to,
                numSamples,
                numIterations,
                numTargets,
                numAttackers,
                maxNumSpecs);
        } else if (exp_name == "var_nd") {
            using out_type = long; // specific to the FUNCTION
            discrete_exp<in_type, out_type, std::uniform_int_distribution>(
                exp_name,
                dist,
                static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_var_nd<in_type>),
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

void batch_exp_poisson(std::string exp_name) {
    const size_t numIterations = 5; // how many times we repeat computation to eliminate noise/random deviations
    const size_t numSamples = 1000;   // number of items in estimator
    const size_t maxNumSpecs = 10;
    const size_t numTargets = 1;
    const size_t numAttackers = 1;

    using in_type = long; // specific to distribution (will always be an integral type for discrete)

    // vector<in_type> lam_vals{4, 8, 16};
    vector<in_type> lam_vals{4, 8, 16};
    // distribution-specific parameters, will vary in each iteration
    for (const auto &lam_v : lam_vals) {
        const in_type range_from = 0;
        const in_type range_to = lam_v*10; //this may not be enough?

        std::poisson_distribution<in_type> dist(lam_v);

        if (exp_name == "max") {
            using out_type = long; // specific to the FUNCTION
            discrete_exp<in_type, out_type, std::poisson_distribution>(
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
            discrete_exp<in_type, out_type, std::poisson_distribution>(
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
            discrete_exp<in_type, out_type, std::poisson_distribution>(
                exp_name,
                dist,
                static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_var<in_type>),
                range_from,
                range_to,
                numSamples,
                numIterations,
                numTargets,
                numAttackers,
                maxNumSpecs);
        } else if (exp_name == "var_nd") {
            using out_type = long; // specific to the FUNCTION
            discrete_exp<in_type, out_type, std::poisson_distribution>(
                exp_name,
                dist,
                static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_var_nd<in_type>),
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
            discrete_exp<in_type, out_type, std::poisson_distribution>(
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
