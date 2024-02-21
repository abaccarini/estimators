#pragma once

#include "XoshiroCpp.hpp"
#include "utilities.hpp"
// #include <cmath>
#include <ctgmath>

// #include <functional>
#include <algorithm>
#include <iostream>
#include <stdexcept>
// #include <iomanip>
// #include <iostream>
// #include <map>
// #include <random>

using std::vector;

inline float abs_value(float x) {
    return fabs(x);
}

inline int abs_value(int x) {
    return abs(x);
}

// IN_T    : type of raw input into the function
// OUT_T   : type of output produced by the function
// DIST_T  : distribution used to generate sample data --> the template parameter for the distribution should match the type of IN_T --> can this be inferred automatically?
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
class knn_est {
public:
    knn_est(const std::uint64_t seed, DIST_T<IN_T> _dist);
    ~knn_est(){};

    IN_T sample_input_data();
    vector<IN_T> generateSamples(const auto qty);
    vector<OUT_T> kNN(vector<OUT_T> &samples, const int k);

private:
    XoshiroCpp::Xoshiro256PlusPlus rng_xos;
    DIST_T<IN_T> dist; // any distribution provided in <random>, as to avoid inverse transform sampling
};

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
knn_est<IN_T, OUT_T, DIST_T>::knn_est(const std::uint64_t seed, DIST_T<IN_T> _dist) : rng_xos(seed), dist(_dist) {}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
IN_T knn_est<IN_T, OUT_T, DIST_T>::sample_input_data() {
    return dist(rng_xos);
}

// populating vector of samples
// cant use mapping here because samples can be continuous/not discertized
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
vector<IN_T> knn_est<IN_T, OUT_T, DIST_T>::generateSamples(const auto qty) {
    vector<IN_T> result(qty);
    for (auto i = 0; i < qty; i++) {
        result[i] = sample_input_data();
    }
    return result;
}

// wait, we only need to sort samples once (when its first generated and all spectator inputs)
// then we need to insert X_A and X_T,  (worst case O(n))
// so lets assume <samples> is sorted
// this version only applies to d = 1
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
vector<OUT_T> knn_est<IN_T, OUT_T, DIST_T>::kNN(vector<OUT_T> &samples, const int k) {
    try {

        auto qty = samples.size();
        if (k >= qty)
            throw std::out_of_range("k cannot be larger than the number of samples");

        // sort(samples.begin(), samples.end()); // standard sort, O(n log n)

        vector<IN_T> result(qty);                       // pre-allocating result
        result[0] = abs_value(samples[0] - samples[k]); // knn can be read directly

        vector<IN_T> distances(2 * k); // allocating once
        size_t lower_offset = 0, upper_offset = 0;
        size_t lower_bound = 0, upper_bound = 0;
        size_t ctr;
        for (auto i = 1; i < qty - 1; i++) {
            ctr = 0;
            // here set the offsets and the upper limit for the two for loops
            // calculate how many items are available to look at
            lower_bound = i < k ? i : k;
            lower_offset = i < k ? k - i : 0;

            upper_bound = (i > (qty - 1 - k)) ? qty - 1 - i : k;
            upper_offset = (i > (qty - 1 - k)) ? k - (qty - 1 - i) : 0;

            for (auto j = 0; j < upper_bound; j++) { // looking right
                distances[ctr++] = abs_value(samples[i] - samples[i + j + 1]);
            }

            for (auto j = 0; j < lower_bound; j++) { // looking left
                distances[ctr++] = abs_value(samples[i] - samples[i - (j + 1)]);
            }

            std::nth_element(distances.begin(), distances.begin() + k - 1, distances.end() - lower_offset - upper_offset); // getting kth NN, check this with the offsets

            result[i] = distances[k - 1];
        }

        result[qty - 1] = abs_value(samples[qty - 1] - samples[qty - 1 - k]); // knn can be read directly
        return result;
    } catch (const std::exception &ex) {
        std::cerr << "(kNN) " << ex.what() << std::endl;
        exit(1);
    }
}
