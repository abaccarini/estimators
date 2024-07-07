#ifndef GAMMA_H_
#define GAMMA_H_

#include "constants.hpp"
#include "plot.hpp"
#include "utilities.hpp"
#include <algorithm>
#include <boost/math/special_functions/digamma.hpp>
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

void simple_driver_gamma();

typedef struct gamma_params_struct {
    double shape_k;
    double scale_theta;
} gamma_params;

// this (currently) requires the scales to be the same for all RVs
typedef struct sum_gamma_RVs_struct {

    double shape_k = 0;
    double scale_theta = 0;
    uint L = 0; // number of RVs we're summing over
    sum_gamma_RVs_struct(double _shape_k, double _scale_theta, uint _L) : L(_L) {
        L = _L;
        shape_k = L * _shape_k;
        scale_theta = _scale_theta;
    }

} sum_gamma_RVs;

typedef struct output_str_gamma_struct {
    gamma_params target_params;
    gamma_params spec_params;
    uint num_targets;
    uint num_spec;
    long double h_T = 0.0;
    long double h_S = 0.0;
    long double h_T_S = 0.0;
    long double awae_differential = 0.0;
} output_str_gamma;

long double single_differential_entropy_gamma(sum_gamma_RVs X);

void writeData_gamma_est(vector<output_str_gamma> out_str, string experiment_name, string params, string distribution);

#endif