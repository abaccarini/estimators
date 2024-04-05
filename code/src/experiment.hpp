#ifndef _EXPERIMENT_HPP_
#define _EXPERIMENT_HPP_

#include "data_io.hpp"
#include "knn_est.hpp"
#include "plug-in_est.hpp"
#include <functional>
#include <string>

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void discrete_exp(std::string exp_name, DIST_T<IN_T> dist, std::function<OUT_T(std::map<IN_T, size_t> &, size_t)> func, const IN_T input_range_from, const IN_T input_range_to, const size_t numSamples, const size_t numIterations, const size_t numTargets, const size_t numAttackers, const size_t maxNumSpecs) {
    const std::uint64_t seed = 12345;

    plug_in_est<IN_T, OUT_T, DIST_T> estimator(seed, dist);
    std::map<size_t, std::map<IN_T, entType>> awae_data;

    for (size_t numSpecs = 1; numSpecs <= maxNumSpecs; numSpecs++) {
        std::map<IN_T, entType> awae_vals;
        for (IN_T j = input_range_from; j <= input_range_to; j++) {
            entType result = 0.0;
            // repeating computation to eliminate random perturbations
            for (size_t k = 0; k < numIterations; k++) {
                result += estimator.estimate_leakage(func, numSamples, numTargets, numAttackers, numSpecs, {j}, input_range_from, input_range_to);
            }
            awae_vals.insert({j, result / static_cast<entType>(numIterations)});
        }
        awae_data.insert({numSpecs, awae_vals});
    }
    discrete_data<IN_T, OUT_T, DIST_T> dd = {exp_name,
                                             dist,
                                             numSamples,
                                             numTargets,
                                             numAttackers,
                                             numIterations,
                                             estimator.target_init_entropy,
                                             awae_data};

    writeJSON_discrete<IN_T, OUT_T, DIST_T>(dd);
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void continuous_exp(std::string exp_name, DIST_T<IN_T> dist, std::function<OUT_T(std::vector<IN_T> &, size_t)> func, const IN_T input_range_from, const IN_T input_range_to, const size_t numOutputSamples, const size_t numInputSamples, const size_t numIterations, const size_t numTargets, const size_t numAttackers, const size_t maxNumSpecs, const IN_T step_size) {
    const std::uint64_t seed = 12345;

    std::map<size_t, std::map<IN_T, entType>> awae_data;
    uint k = 1; // 1st NN
    knn_est<IN_T, OUT_T, DIST_T> estimator(seed, dist, func, numOutputSamples, numInputSamples, 0, numTargets, k);
    for (size_t numSpecs = 1; numSpecs <= maxNumSpecs; numSpecs++) {
        estimator.set_numSpectators(numSpecs);

        std::map<IN_T, entType> awae_vals;
        // going in step_size increments to try and obtain a continuous function of x_A
        for (IN_T j = input_range_from; j <= input_range_to; j += (step_size)) {
            entType result = 0.0;
            estimator.x_A = {j};

            // repeating computation to eliminate random perturbations
            for (size_t k = 0; k < numIterations; k++) {
                result += estimator.estimate_leakage();
            }
            awae_vals.insert({j, result / static_cast<entType>(numIterations)});
        }
        awae_data.insert({numSpecs, awae_vals});
    }
    continuous_data<IN_T, OUT_T, DIST_T> dd = {exp_name,
                                               dist,
                                               numOutputSamples,
                                               numInputSamples,
                                               numTargets,
                                               numAttackers,
                                               numIterations,
                                               estimator.target_init_entropy,
                                               awae_data};

    writeJSON_continuous<IN_T, OUT_T, DIST_T>(dd);
}

void batch_exp_discrete_uniform(std::string, const size_t numSamples = 3000);
void batch_exp_poisson(std::string);

void batch_exp_normal(std::string exp_name);
void batch_exp_lognormal(std::string exp_name);
#endif // _EXPERIMENT_HPP_
