#ifndef NORMAL_V2_H_
#define NORMAL_V2_H_

#include "constants.hpp"
#include "joint.hpp"
#include "joint_gaussian_v2.hpp"
#include "normal.hpp"
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
#include <mpfr.h>
#include <mpir.h>
#include <string>
#include <vector>
#include <map>

using namespace std;

typedef struct output_str_shannon_vs_diff_struct {
    normal_params target_params;
    normal_params spec_params;
    uint num_targets;
    uint num_spec;
    long double H_T = 0.0;
    long double H_S = 0.0;
    long double H_T_S = 0.0;
    long double delta_T = 0.0;
    long double delta_S = 0.0;
    long double delta_T_S = 0.0;
    long double awae_shannon = 0.0;
    long double h_T = 0.0;
    long double h_S = 0.0;
    long double h_T_S = 0.0;
    long double awae_differential = 0.0;
} output_str_shannon_vs_diff;



typedef struct output_mixed {
    normal_params target_params;
    vector<normal_params> spec_params;
    uint num_targets;
    vector<uint> num_spec;
    long double h_T = 0.0;
    long double h_S = 0.0;
    long double h_T_S = 0.0;
    long double awae_differential = 0.0;
} output_mixed;



long double get_right_root_normal(gaussian_RV &X, double eps);
long double get_right_root_normal(sum_gaussian_RVs_v2 &X, double eps);
double evaluate_normal_pmf(sum_gaussian_RVs_v2 &X, double delta, double left_bound, uint integ_type);
double evaluate_normal_pmf(gaussian_RV &X, double delta, double left_bound, uint integ_type);
double normal_pdf(gaussian_RV &X, double x);
double normal_pdf(sum_gaussian_RVs_v2 &X, double x);

output_str_shannon_vs_diff single_calculation_gaussian(normal_params &target_params, normal_params &spec_params, uint num_spec, uint num_targets, uint N, double eps, uint integ_type) ;

void single_exp_gaussian_main();

void writeData_normal_v2(vector<output_str_shannon_vs_diff_struct> out_str, string experiment_name, string params, string distribution);
void single_mixed_gaussian_main();

output_mixed single_calculation_gaussian_mixed(normal_params &target_params, normal_params &spec_params, uint num_spec, uint num_targets, uint N, double eps, uint integ_type) ;
output_mixed single_calculation_gaussian_mixed(normal_params &target_params, vector<normal_params> &spec_params, vector<uint> num_spec, uint num_targets);
void writeData_normal_mixed(vector<output_mixed> out_str,  string params, string distribution);


#endif
