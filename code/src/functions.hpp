#ifndef _FUNCTIONS_HPP_
#define _FUNCTIONS_HPP_

#include <cstddef>
#include <map>
#include <numeric>
#include <vector>

// computes the median, but not "true" median if the num_samples is even
template <typename T>
T compute_median_min(std::map<T, size_t> &map, const size_t &num_samples) {
    auto n = (num_samples) / 2;
    auto total = 0;
    // this is better than trying to create a vector of all the values in the map,
    // which will require traversal regardless to determine how many of each sample we need for initlaizaion
    // this is worst case O(N)
    // for (const auto &[num, count] : map)
    // std::cout << num << " generated " <<count << " times\n";

    for (const auto &[num, count] : map) {
        total += count;
        if (total > n) {
            return num;
        }
    }
    return 0; // should never get here regardless
}

template <typename T, typename U>
U compute_median(std::map<T, size_t> &map, const size_t &num_samples) {
    auto n = (num_samples) / 2;
    auto total = 0;
    if (!(num_samples & 1)) // if there is an even number of samples, do the following
        for (const auto &[num, count] : map) {
            total += count;
            if (total > n) { // this needs to be reworked for even samples
                return num;
            }
        }
    else {
        for (const auto &[num, count] : map) {
            total += count;
            if (total > n) {
                return num;
            }
        }
    }
    return 0; // should never get here regardless
}

template <typename T, typename U>
U compute_mean(std::map<T, size_t> &map, const size_t &num_samples) {
    U total = 0;
    for (const auto &[num, count] : map)
        total += count * num;
    return total / static_cast<U>(num_samples);
}

// computes the unbiased sample variance (Bessel's correction)
// summation is divided by (num_samples - 1)
template <typename T, typename U>
U compute_var(std::map<T, size_t> &map, const size_t &num_samples) {
    U mu = compute_mean<T, U>(map, num_samples);
    U total = 0;
    for (const auto &[num, count] : map)
        total += static_cast<U>(count) * (static_cast<U>(num) - mu) * (static_cast<U>(num) - mu); // (x_i - mu)^2, count is the number of occurences of a pice of data
    return total / static_cast<U>(num_samples - 1);
}

template <typename T, typename U>
std::pair<U,U> compute_var_mean(std::map<T, size_t> &map, const size_t &num_samples) {
    U mu = compute_mean<T, U>(map, num_samples);
    U total = 0;
    for (const auto &[num, count] : map)
        total += static_cast<U>(count) * (static_cast<U>(num) - mu) * (static_cast<U>(num) - mu); // (x_i - mu)^2, count is the number of occurences of a pice of data
    
    return { total / static_cast<U>(num_samples - 1), mu };
}


// added num_samples argument just so that every function has the same format
// good for now, may need to modify estimate_H_cond later for more complex functions
template <typename T>
T compute_max(std::map<T, size_t> &map, const size_t &num_samples) {
    return (--map.end())->first;
}

template <typename T>
T compute_min(std::map<T, size_t> &map, const size_t &num_samples) {
    return (map.begin())->first;
}

// since these functions are overloaded, we need to cast them in order for the compiler to know which one to look at:
// static_cast<T (*)(std::map<T, T> &, const size_t &)>(compute_max<T>)
template <typename T>
T compute_max(std::vector<T> &vec, const size_t &num_samples) {
    return max(vec.begin(), vec.end());
}

template <typename T>
T compute_min(std::vector<T> &vec, const size_t &num_samples) {
    return min(vec.begin(), vec.end());
}

template <typename T>
T compute_sum(std::vector<T> &vec, const size_t &num_samples) {
    return std::accumulate(vec.begin(), vec.end(), static_cast<T>(0));
}

#endif // _FUNCTIONS_HPP_