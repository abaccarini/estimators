#ifndef LOGNORMAL_V2_H_
#define LOGNORMAL_V2_H_

#include "constants.hpp"
#include "ent_est.hpp"
#include "normal_v2.hpp"
#include "plot.hpp"
#include "poisson.hpp"
#include "sln.hpp"
#include "utilities.hpp"
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

using namespace std;

typedef struct output_diff_est_struct {
    normal_params target_params;
    normal_params spec_params;
    uint num_targets;
    uint num_spec;
    uint numSamples;
    uint numIterations;
    uint k;
    long double h_T = 0.0;
    long double h_S = 0.0;
    long double h_T_S = 0.0;
    long double awae_differential = 0.0;
} output_diff_est;

typedef struct lognormal_params_struct {
    double mu;
    double sigma;
    double eps;
} lognormal_params;

// sigma is already squared
typedef struct lognormal_RV_struct {
    double mu = 0.0;
    double sigma = 0.0;
} lognormal_RV;

// sigma is already squared
typedef struct sum_lognormal_RVs_struct {

    double mu = 0;
    double sigma = 0;
    uint L = 0; // number of RVs we're summing over
    sum_lognormal_RVs_struct(double _mu, double _sigma, uint _L) : L(_L) {
        if (L == 1) {
            sigma = _sigma;
            mu = _mu;
        } else {

            sigma = (log((exp(_sigma) - 1) / L + 1));
            mu = log(L * exp(_mu)) + 0.5 * (_sigma - sigma);
        }
    }

} sum_lognormal_RVs;

double evaluate_lognormal_pmf(sum_lognormal_RVs &X, uint i, double delta, double left_bound, uint integ_type);
long double get_right_root_lognormal(lognormal_RV &X, double eps);
long double get_right_root_lognormal(sum_lognormal_RVs &X, double eps);
long double single_differential_entropy_lognormal(sum_lognormal_RVs X);
long double single_differential_entropy_lognormal(lognormal_RV X);
output_str_shannon_vs_diff single_calculation_lognormal(normal_params &target_params, normal_params &spec_params, uint num_spec, uint num_targets, uint N, double eps, uint integ_type);

output_diff_est single_calculation_lognormal_est(normal_params &target_params, normal_params &spec_params, uint num_spec, uint num_targets,uint numSamples, uint numIterations, uint k);

long double normal_pdf(lognormal_RV &X, double x);
long double ln_pdf(sum_lognormal_RVs &X, double x);
double ln_pdf_v2(double x, void *p);

void single_exp_lognormal_main();
void writeData_lognormal_est(vector<output_diff_est> out_str, string experiment_name, string params, string distribution) ;

#endif
