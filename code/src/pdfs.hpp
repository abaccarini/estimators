#ifndef _PDFS_HPP_
#define _PDFS_HPP_
#include <cmath>

// NOT SIGMA SQUARED
template <typename T>
T normal_pdf(T x, T mu, T sigma) {
    static const T inv_sqrt_2pi = 0.3989422804014327;
    T a = (x - mu) / sigma;
    return inv_sqrt_2pi / sigma * std::exp(-T(0.5) * a * a);
}

// NOT SIGMA SQUARED
template <typename T>
T lognormal_pdf(T x, T mu, T sigma) {
    static const T inv_sqrt_2pi = 0.3989422804014327; // 1/sqrt(2*pi)
    T a = (log(x) - mu) / sigma;
    return inv_sqrt_2pi / (x * sigma) * std::exp(-T(0.5) * a * a);
}

// if a <= x <= b, then return 1/(b-a), 0 otherwise
template <typename T>
T uniform_real_pdf(T x, T a, T b) {
    return ((x <= b) || (x >= a)) ? T(1) / (b - a) : T(0);
}

#endif // _PDFS_HPP_