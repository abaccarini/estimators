#ifndef GAUSSIAN_MIXED_H_
#define GAUSSIAN_MIXED_H_

#include "constants.hpp"
#include "yield.hpp"
#include "normal.hpp"
#include "normal_v2.hpp"
#include <cmath>
#include <gmp.h>
#include <gmpxx.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <iostream>
#include <map>
#include <mpfr.h>
#include <mpir.h>
#include <string>
#include <vector>
#include <numeric>

using namespace std;
double gaussian_pdf(double x, void *p);
double log_gaussian_pdf(double x, void *p);
void single_mixed_gaussian_case_2();

struct comb_coeff {
    vector<uint64_t>v;
    uint64_t coeff; 
    long double coeff_fl; 
};

struct gaussian_params {
    std::vector<double> sigmas;
    std::vector<double> mus;
    uint64_t n; // total number of participants, used to generate all permutations of group size
};

#endif
