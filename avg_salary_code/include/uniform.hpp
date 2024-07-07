#ifndef UNIFORM_H_
#define UNIFORM_H_

#include "constants.hpp"
#include "plot.hpp"
#include "utilities.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <gmp.h>
#include <gmpxx.h>
#include <iomanip>
#include <iostream>
#include <mpfr.h>
#include <mpir.h>
#include <numeric>
#include <string>
#include <types.hpp>
#include <vector>
#define PREC 220

long double uniform_pmf(int y, int n, int k);
long double uniform_pmf_test(int y, int n, int k);
void simple_driver_uniform(int N);
void simple_driver_uniform_test(int N);

long double uniform_pmf_mpfr(int y, int n, int k);

#endif