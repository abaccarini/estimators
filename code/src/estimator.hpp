#ifndef _ESTIMATOR_HPP_
#define _ESTIMATOR_HPP_
#include "XoshiroCpp.hpp"
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>

#define EPSILON 0.0000000001

template <typename T>
class plug_in_est {
public:
    plug_in_est(const std::uint64_t seed, const int range_from, const int range_to);
    ~plug_in_est(){};

    T sample();
    std::map<T, T> generateSamples(auto qty);

    long double estimate_H(std::map<T, T> samples, const size_t num_samples);
    long double estimate_H_cond(std::function<T(std::map<T, T> &, size_t)> func, const size_t &num_output_samples, const size_t &num_input_samples, std::vector<T> x_T, std::vector<T> x_A);

    long double estimate_leakage(std::function<T(std::map<T, T> &, size_t)> func, const size_t &num_samples, const size_t &num_T, const size_t &num_A, const size_t &num_S, std::vector<T> x_A);

private:
    XoshiroCpp::Xoshiro256PlusPlus rng_xos;
    const int range_from;
    const int range_to;
    std::uniform_int_distribution<int> dist;
};

// range_from and range_to are INCLUSIVE, so values are sampled over the domain [range_from, range_to]
template <typename T>
plug_in_est<T>::plug_in_est(const std::uint64_t seed, const int _range_from, const int _range_to) : rng_xos(seed), range_from(_range_from), range_to(_range_to), dist(range_from, range_to) {
}

template <typename T>
T plug_in_est<T>::sample() {
    return dist(rng_xos);
}

template <typename T>
std::map<T, T> plug_in_est<T>::generateSamples(auto qty) {
    std::map<T, T> ptr;
    for (size_t i = 0; i < qty; i++) {
        // est.sample();
        ++ptr[sample()];
    }
    return ptr;
}

template <typename T>
long double plug_in_est<T>::estimate_H(std::map<T, T> samples, const size_t num_samples) {
    // samples: frequency table mapping an input -> no. occurences
    // num_samples: sum of all values in map (the amount of sample data)
    long double p_n = 0.0;
    long double result = 0.0;
    for (const auto &[num, count] : samples) {
        // if (count > 0){ // dont think we need to do this, because we will always get  values that are nonzero
        // otherwise they wouldn't have been inserted into the mapping
        p_n = (1.0 / (static_cast<long double>(num_samples))) * static_cast<long double>(count);
        // std::cout<<static_cast<long double>(num_samples)<<","<<static_cast<long double>(count)<<","<<p_n<<","<<std::endl;
        if (p_n > EPSILON) // safe to compute, otherwise it's just zero and not added
            result += p_n * log2(p_n);
    }
    return (-1.0) * result;
}

// x_A and x_T will always be singular values encapsulated in a vector
// the vector component allows us to know (in constant time) if the argument was provided
// num_output_samples corresponds to the actual amount of data we want to feed the estimator (should be "large")
template <typename T>
long double plug_in_est<T>::estimate_H_cond(std::function<T(std::map<T, T> &, size_t)> func, const size_t &num_output_samples, const size_t &num_input_samples, std::vector<T> x_T, std::vector<T> x_A) {
    std::map<T, T> input_data;
    std::map<T, T> output_data;

    for (size_t i = 0; i < num_output_samples; i++) {
        input_data = generateSamples(num_input_samples); // this is properly generated
        if (x_T.size() > 0)                              // if we have a known x_T to add to input_data
            ++input_data[x_T[0]];
        if (x_A.size() > 0) // if we have a known x_A to add to input_data
            ++input_data[x_A[0]];
        // for (const auto &[num, count] : input_data)
        // std::cout << num << " generated " << std::setw(4) << count << " times\n";
        // std::cout << "result " << std::setw(4) << func(input_data, num_input_samples) << "\n";
        ++output_data[func(input_data, num_input_samples + x_T.size() + x_A.size())];
    }
    // for (const auto &[num, count] : output_data)
    // std::cout << num << " generated " << std::setw(4) << count << " times\n";

    return estimate_H(output_data, num_output_samples);
}

// computes the awae given an x_A value
// can you reuse SPECTATOR samples generated in H_O_XT_xA when computing H_O_xA?
template <typename T>
long double plug_in_est<T>::estimate_leakage(std::function<T(std::map<T, T> &, size_t)> func, const size_t &num_samples, const size_t &num_T, const size_t &num_A, const size_t &num_S, std::vector<T> x_A) {

    // H_T can be computed exactly
    long double H_O_XT_xA = 0.0, H_O_xA = 0.0;
    long double n = (long double)(range_to - range_from + 1);
    long double H_T = log2(n); // entropy of uniform RV

    // this needs to be replaced with an input generator for num_T > 1
    for (T i = range_from; i <= range_to; i++) {
        H_O_XT_xA += (1.0 / n) * estimate_H_cond(func, num_samples, num_S, {i}, x_A);
    }
    // both x_T and x_s must be randomly sampled, x_A is fixed
    H_O_xA = estimate_H_cond(func, num_samples, num_S + num_T, {}, x_A);
    // std::cout << "H_T       : " << H_T << std::endl;
    // std::cout << "H_O_XT_xA : " << H_O_XT_xA << std::endl;
    // std::cout << "H_O_XT_xA : " << H_O_XT_xA << std::endl;

    return (H_T + H_O_XT_xA - H_O_xA);
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
class plug_in_est_new {
public:
    plug_in_est_new(const std::uint64_t seed, DIST_T<IN_T> _dist);
    ~plug_in_est_new(){};

    IN_T sample();
    std::map<IN_T, int> generateSamples(auto qty);

    long double estimate_H(std::map<IN_T, int> samples, const size_t num_samples);


    long double estimate_H_cond(std::function<OUT_T(std::map<IN_T, int> &, size_t)> func, const size_t &num_output_samples, const size_t &num_input_samples, std::vector<IN_T> x_T, std::vector<IN_T> x_A);

    long double estimate_leakage(std::function<OUT_T(std::map<IN_T, int> &, size_t)> func, const size_t &num_samples, const size_t &num_T, const size_t &num_A, const size_t &num_S, std::vector<IN_T> x_A, const int range_from, const int range_to);

private:
    XoshiroCpp::Xoshiro256PlusPlus rng_xos;
    DIST_T<IN_T> dist;
};

// range_from and range_to are INCLUSIVE, so values are sampled over the domain [range_from, range_to]
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
plug_in_est_new<IN_T, OUT_T, DIST_T>::plug_in_est_new(const std::uint64_t seed, DIST_T<IN_T> _dist) : rng_xos(seed), dist(_dist) {
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
IN_T plug_in_est_new<IN_T, OUT_T, DIST_T>::sample() {
    return dist(rng_xos);
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
std::map<IN_T, int> plug_in_est_new<IN_T, OUT_T, DIST_T>::generateSamples(const auto qty) {
    std::map<IN_T, int> ptr;
    for (size_t i = 0; i < qty; i++) {
        ++ptr[sample()];
    }
    return ptr;
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
long double plug_in_est_new<IN_T, OUT_T, DIST_T>::estimate_H(std::map<IN_T, int> samples, const size_t num_samples) {
    // samples: frequency table mapping an input -> no. occurences
    // num_samples: sum of all values in map (the amount of sample data)
    long double p_n = 0.0;
    long double result = 0.0;
    for (const auto &[num, count] : samples) {
        // if (count > 0){ // dont think we need to do this, because we will always get  values that are nonzero
        // otherwise they wouldn't have been inserted into the mapping
        p_n = (1.0 / (static_cast<long double>(num_samples))) * static_cast<long double>(count);
        // std::cout<<static_cast<long double>(num_samples)<<","<<static_cast<long double>(count)<<","<<p_n<<","<<std::endl;
        if (p_n > EPSILON) // safe to compute, otherwise it's just zero and not added
            result += p_n * log2(p_n);
    }
    return (-1.0) * result;
}

// x_A and x_T will always be singular values encapsulated in a vector
// the vector component allows us to know (in constant time) if the argument was provided
// num_output_samples corresponds to the actual amount of data we want to feed the estimator (should be "large")
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
long double plug_in_est_new<IN_T, OUT_T, DIST_T>::estimate_H_cond(std::function<OUT_T(std::map<IN_T, int> &, size_t)> func, const size_t &num_output_samples, const size_t &num_input_samples, std::vector<IN_T> x_T, std::vector<IN_T> x_A) {
    std::map<IN_T, int> input_data;
    std::map<OUT_T, int> output_data;

    for (size_t i = 0; i < num_output_samples; i++) {
        input_data = generateSamples(num_input_samples); // this is properly generated
        if (x_T.size() > 0)                              // if we have a known x_T to add to input_data
            ++input_data[x_T[0]];
        if (x_A.size() > 0) // if we have a known x_A to add to input_data
            ++input_data[x_A[0]];
        // for (const auto &[num, count] : input_data)
        // std::cout << num << " generated " << std::setw(4) << count << " times\n";
        // std::cout << "result " << std::setw(4) << func(input_data, num_input_samples) << "\n";
        ++output_data[func(input_data, num_input_samples + x_T.size() + x_A.size())];
    }
    // for (const auto &[num, count] : output_data)
    // std::cout << num << " generated " << std::setw(4) << count << " times\n";

    return estimate_H(output_data, num_output_samples);
}

// computes the awae given an x_A value
// can you reuse SPECTATOR samples generated in H_O_XT_xA when computing H_O_xA?
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
long double plug_in_est_new<IN_T, OUT_T, DIST_T>::estimate_leakage(std::function<OUT_T(std::map<IN_T, int> &, size_t)> func, const size_t &num_samples, const size_t &num_T, const size_t &num_A, const size_t &num_S, std::vector<IN_T> x_A, const int range_from, const int range_to) {

    // H_T can be computed exactly
    long double H_O_XT_xA = 0.0, H_O_xA = 0.0;
    // auto range_from = dist.min(); // could be useful for finite discrete dists, but not infinite (e.g. poisson)
    // auto range_to = dist.max(); // could be useful for finite discrete dists

    long double n = (long double)(range_to - range_from + 1);
    long double H_T = log2(n); // entropy of uniform RV

    // this needs to be replaced with an input generator for num_T > 1
    for (IN_T i = range_from; i <= range_to; i++) {
        H_O_XT_xA += (1.0 / n) * estimate_H_cond(func, num_samples, num_S, {i}, x_A);
    }
    // both x_T and x_s must be randomly sampled, x_A is fixed
    H_O_xA = estimate_H_cond(func, num_samples, num_S + num_T, {}, x_A);
    // std::cout << "H_T       : " << H_T << std::endl;
    // std::cout << "H_O_XT_xA : " << H_O_XT_xA << std::endl;
    // std::cout << "H_O_XT_xA : " << H_O_XT_xA << std::endl;

    return (H_T + H_O_XT_xA - H_O_xA);
}

#endif // _ESTIMATOR_HPP_