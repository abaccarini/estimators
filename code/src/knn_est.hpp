#pragma once

#include "XoshiroCpp.hpp"
#include "pdfs.hpp"
#include "utilities.hpp"
#include <algorithm>
#include <boost/math/special_functions/digamma.hpp>
#include <ctgmath>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>

using std::vector;
using entType = long double;

/// @brief returns the absolute value of an float
inline float abs_value(float x) {
    return fabsf(x);
}

/// @brief returns the absolute value of an double
inline float abs_value(double x) {
    return fabs(x);
}

/// @brief returns the absolute value of an double
inline float abs_value(long double x) {
    return fabsl(x);
}

/// @brief returns the absolute value of an integer
inline int abs_value(int x) {
    return abs(x);
}
/// @brief returns the absolute value of an integer
inline int abs_value(long int x) {
    return labs(x);
}
/// @brief returns the absolute value of an integer
inline int abs_value(long long int x) {
    return llabs(x);
}

// IN_T    : type of raw input into the function
// OUT_T   : type of output produced by the function
// DIST_T  : distribution used to generate sample data --> the template parameter for the distribution should match the type of IN_T --> can this be inferred automatically?
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
class knn_est {
public:
    knn_est(const std::uint64_t seed, DIST_T<IN_T> _dist, std::function<OUT_T(vector<IN_T> &, size_t)> func, size_t _num_samples, size_t numS, size_t numT, uint _k);
    ~knn_est(){};

    IN_T sample_input_data();

    vector<IN_T> generateSamples(const auto qty);
    vector<IN_T> generateSamples(const auto qty, IN_T x_T);
    vector<IN_T> generateSamples(const auto qty, IN_T x_T, vector<IN_T> &x_A);
    void generateSamples(vector<IN_T> &results, const auto qty, IN_T x_T, vector<IN_T> &x_A);

    vector<OUT_T> kNN(vector<OUT_T> &samples, const int k);

    void set_x_A(vector<IN_T> &x);
    vector<IN_T> get_x_A();
    IN_T evaluate_pdf(IN_T x); // only used for integration

    void set_numSpectators(size_t _numSpec);


    entType estimate_h(IN_T x_T);
    entType estimate_h(); // corretcness testing only
    entType estimate_leakage();

    entType calculateDifferentialEntropy(DIST_T<IN_T> dist);

    std::string get_func_type();

    entType target_init_entropy;

private:
    XoshiroCpp::Xoshiro256PlusPlus rng_xos;
    DIST_T<IN_T> dist; // any distribution provided in <random>, as to avoid inverse transform sampling
    // does <random> allow access to the pdf of dist?

    // the function we are evaluating, stored in the estimator object
    std::function<OUT_T(vector<IN_T> &, size_t)> func;

    const size_t num_samples; // the number of OUTPUT samples we produce
    size_t numS;
    const size_t numT;
    vector<IN_T> x_A;

    uint k; // kth nearest neighbor
};

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
knn_est<IN_T, OUT_T, DIST_T>::knn_est(const std::uint64_t seed, DIST_T<IN_T> _dist,
                                      std::function<OUT_T(vector<IN_T> &, size_t)> _func,
                                      size_t _num_samples,
                                      size_t _numS,
                                      size_t _numT,
                                      uint _k) : rng_xos(seed),
                                                 dist(_dist),
                                                 func(_func),
                                                 num_samples(_num_samples),
                                                 numS(_numS),
                                                 numT(_numT),
                                                 k(_k) {

    target_init_entropy = calculateDifferentialEntropy(dist); // only needs to be computed once (in constructor)
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
entType knn_est<IN_T, OUT_T, DIST_T>::calculateDifferentialEntropy(DIST_T<IN_T> dist) {
    try {
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::normal_distribution<IN_T>>)
            return 0.5 * log2(2 * M_PI * M_E * dist.stddev() * dist.stddev());
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::uniform_real_distribution<IN_T>>)
            return log2(dist.b() - dist.a());
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::lognormal_distribution<IN_T>>)
            // computed in nats, then converted to bits
            return (dist.m() + 0.5 * log(2 * M_PI * M_E * dist.s())) / log(2.0);
        // return dist.m() + 0.5 * log2(2 * M_PI * M_E * dist.s() * dist.s());
        throw std::runtime_error("unsupported distribution encountered, please add the pdf of the distribution you want in pdfs.hpp, as well as additional \"if\" block here.");
    } catch (const std::exception &ex) {
        std::cerr << "(calculateDifferentialEntropy): " << ex.what() << std::endl;
        exit(1);
    }
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
IN_T knn_est<IN_T, OUT_T, DIST_T>::sample_input_data() {
    return dist(rng_xos);
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void knn_est<IN_T, OUT_T, DIST_T>::set_x_A(vector<IN_T> &x) {
    x_A = x;
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void knn_est<IN_T, OUT_T, DIST_T>::set_numSpectators(size_t _numSpec) {
    numS = _numSpec;
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
vector<IN_T> knn_est<IN_T, OUT_T, DIST_T>::get_x_A() {
    return x_A;
}

// if this is inside an integral evaluation, only produce one set
// populating vector of samplesp
// cant use mapping here because samples can be continuous/not discertized
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
vector<IN_T> knn_est<IN_T, OUT_T, DIST_T>::generateSamples(const auto qty) {
    vector<IN_T> result(qty);
    for (auto i = 0; i < qty; i++) {
        result[i] = sample_input_data();
    }
    return result;
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
vector<IN_T> knn_est<IN_T, OUT_T, DIST_T>::generateSamples(const auto qty, IN_T x_T) {
    vector<IN_T> result(qty + 1);
    for (auto i = 0; i < qty; i++) {
        result[i] = sample_input_data();
    }
    result[qty] = x_T;
    return result;
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
vector<IN_T> knn_est<IN_T, OUT_T, DIST_T>::generateSamples(const auto qty, IN_T x_T, vector<IN_T> &x_A) {
    vector<IN_T> result(qty + 1 + x_A.size());
    for (auto i = 0; i < qty; i++)
        result[i] = sample_input_data();
    result[qty] = x_T;
    result[qty + 1] = x_A[0]; // this is only for one attacker input
    return result;
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void knn_est<IN_T, OUT_T, DIST_T>::generateSamples(vector<IN_T> &result, const auto qty, IN_T x_T, vector<IN_T> &x_A) {
    for (auto i = 0; i < qty; i++)
        result[i] = sample_input_data();
    result[qty] = x_T;
    result[qty + 1] = x_A[0]; // this is only for one attacker input
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

// we are forced (for every integration iteration) to re-sample the data, since the sample vector will almost certainly change with every x_T that is evaluated
// x_T will be determined by the integration algorithm
// the estimator uses the natural logarithm inside
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
entType knn_est<IN_T, OUT_T, DIST_T>::estimate_h(IN_T x_T) {
    // generate samples inside here
    // sort (for kNN calculation)
    // vector<OUT_T> samples = {generateSamples(num_samples), x_T, x_A}; // this doesnt work
    vector<OUT_T> distances(num_samples);
    auto numInputs = numS + numT + x_A.size();
    // vector<IN_T> samples(numInputs);
    vector<IN_T> samples;

    for (size_t i = 0; i < num_samples; i++) {
        // generateSamples(samples, numS, x_T, x_A);
        samples = generateSamples(numS, x_T, x_A); // this takes advantage of return value optimization
        // std::cout << "samples[" << i << "] = " << samples << " : func() = "<<func(samples, numInputs)<<std::endl;

        distances[i] = func(samples, numInputs);
    }
    sort(distances.begin(), distances.end()); // standard sort, O(n log n)
    // std::cout << "distances = " << distances << std::endl;
    distances = kNN(distances, k); // computing the

    entType result = 0.0;
    static const entType c_d_p = 2.0; // d = 1, p is always 2 for 1D euclidean distance (equiv to manhattan )
    for (size_t i = 0; i < num_samples; i++) {
        // result += log(static_cast<entType>(num_samples) * c_d_p * distances[i] / static_cast<entType>(k));
        result += log(static_cast<entType>(num_samples)) + log(c_d_p) + log(distances[i]) - log(static_cast<entType>(k));
    }

    return ((1.0 / static_cast<entType>(num_samples)) * result + log(static_cast<entType>(k)) - boost::math::digamma(static_cast<entType>(k))) * log2(M_E); // need to convert from nats to bits
}

// general estimator used just for testing estimator correctness
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
entType knn_est<IN_T, OUT_T, DIST_T>::estimate_h() {
    // generate samples inside here
    // sort (for kNN calculation)
    vector<OUT_T> distances = generateSamples(num_samples);
    sort(distances.begin(), distances.end()); // standard sort, O(n log n)
    // std::cout << "distances = " << distances << std::endl;
    distances = kNN(distances, k); // computing the

    entType result = 0.0;
    static const entType c_d_p = 2.0; // d = 1, p is always 2 for 1D euclidean distance (equiv to manhattan )
    for (size_t i = 0; i < num_samples; i++) {
        result += log((static_cast<entType>(num_samples) * c_d_p * distances[i]) / static_cast<entType>(k)); // no difference between these two versions (exact same)
        // result += log(static_cast<entType>(num_samples)) + log(c_d_p) + log(distances[i] )-  log(static_cast<entType>(k)); // no difference between these two versions (exact same)
    }

    return ((1.0 / static_cast<entType>(num_samples)) * result + log(static_cast<entType>(k)) - boost::math::digamma(static_cast<entType>(k))) * log2(M_E); // need to convert from nats to bits
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
entType knn_est<IN_T, OUT_T, DIST_T>::estimate_leakage() {

    // H_T can be computed exactly
    entType H_O_XT_xA = 0.0, H_O_xA = 0.0;

    // if (range_from > range_to)
    //     throw std::runtime_error("range_from cannot be larger than range_to");

    // // this needs to be replaced with an input generator for num_T > 1
    // for (IN_T i = range_from; i <= range_to; i++) {
    //     // H_O_XT_xA += (1.0 / n) * estimate_H_cond(func, num_samples, num_S, {i}, x_A); //   THE LEADING 1/N IS SPECIFIC TO THE UNIFORM DISTRIBUTION!!!!!
    //     H_O_XT_xA += (evaluate_pmf(i)) * estimate_H_cond(func, num_samples, num_S, {i}, x_A);
    // }

    // both x_T and x_s must be randomly sampled, x_A is fixed
    // H_O_xA = estimate_H_cond(func, num_samples, num_S + num_T, {}, x_A);
    return (target_init_entropy + H_O_XT_xA - H_O_xA);
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
IN_T knn_est<IN_T, OUT_T, DIST_T>::evaluate_pdf(IN_T x_T) {
    try {
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::normal_distribution<IN_T>>)
            return normal_pdf(x_T, dist.mean(), dist.stddev());
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::uniform_real_distribution<IN_T>>)
            return uniform_real_pdf(x_T, dist.a(), dist.b());
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::lognormal_distribution<IN_T>>)
            return lognormal_pdf(x_T, dist.m(), dist.s());
        throw std::runtime_error("unsupported distribution encountered, please add the pdf of the distribution you want in pdfs.hpp, as well as additional \"if\" block here.");
    } catch (const std::exception &ex) {
        std::cerr << "(evaluate_pdf): " << ex.what() << std::endl;
    }
}

template <typename T, typename... U>
size_t getAddress(std::function<T(U...)> f) {
    typedef T(fnType)(U...);
    fnType **fnPointer = f.template target<fnType *>();
    return (size_t)*fnPointer;
}

// template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
// std::string knn_est<IN_T, OUT_T, DIST_T>::get_func_type() {
//     std::function<IN_T(std::vector<IN_T>&, const size_t &)> test = &compute_sum;
//     if(getAddress(func) == getAddress(test)){
//         printf("func is sum\n");
//     }else {
//         printf("func is not sum\n");

//     }
// //     if(getAddress(func) == getAddress(std::function<IN_T>( compute_max))){
// //         printf("func is max\n");
// //     }else {
// //         printf("func is not max\n");
// //     }
// }
