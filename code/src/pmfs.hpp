#ifndef _PMFS_HPP_
#define _PMFS_HPP_
#include <cmath>

template <typename T>
T poisson_pmf(int x, double lam) {
    return pow(M_E, x * log(lam) - lam -lgamma(x + 1.0));
}
#endif // _PMFS_HPP_