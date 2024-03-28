#ifndef _PMFS_HPP_
#define _PMFS_HPP_
#include <cmath>

#define EPSILON 0.0000000001

template <typename T, typename U>
T poisson_pmf(U x, double lam) {
    return pow(M_E, x * log(lam) - lam - lgamma(x + 1.0));
}

template <typename T>
T gFunc(T n) {
    // chosen by convention, since lim_{n->0} n log n = 0
    if (n < EPSILON) {
        return 0.0;
    } else {
        return n * log2(n);
    }
}

#endif // _PMFS_HPP_