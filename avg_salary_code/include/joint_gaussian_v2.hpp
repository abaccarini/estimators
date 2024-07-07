#ifndef JOINT_GAUSS_V2_H_
#define JOINT_GAUSS_V2_H_

// #define _CRTDBG_MAP_ALLOC
// #include <stdlib.h>
// #include <crtdbg.h>

#include "constants.hpp"
// #include "matplotlibcpp.h"
#include "joint.hpp"
#include "normal.hpp"
#include "plot.hpp"
#include "poisson.hpp"
#include "sln.hpp"
#include "utilities.hpp"
// #include <bits/stdc++.h>
#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gmp.h>
#include <gmpxx.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <iomanip>
#include <iostream>
#include <mpfr.h>
#include <mpir.h>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>

typedef struct output_str_3exp_struct {
    normal_params target;
    normal_params shared_params;
    normal_params ex1_params;
    normal_params ex2_params;
    normal_params ex3_params;
    uint num_shared;
    uint num_spec_1;
    uint num_spec_2;
    uint num_spec_3;
    long double target_init = 0.0;
    long double one_exp = 0.0;
    long double two_exp = 0.0;
    long double three_exp = 0.0;
} output_str_3exp;




// the new version adopt the same form of the poisson RVs
typedef struct sum_gaussian_RVs_struct_v2 {
    std::vector<gaussian_RV> const &shared_RVs;
    std::vector<gaussian_RV> const &unique_RVs;

    double mu = 0;
    double sigma = 0;

    double mu_shared = 0;
    double sigma_shared = 0;

    double mu_unique = 0;
    double sigma_unique = 0;

    sum_gaussian_RVs_struct_v2(std::vector<gaussian_RV> const &_shared_RVs,
                               std::vector<gaussian_RV> const &_unique_RVs) : shared_RVs(_shared_RVs), unique_RVs(_unique_RVs) {
        for (auto &spec : shared_RVs) {
            mu_shared += spec.mu;
            sigma_shared += spec.sigma;
        }
        for (auto &spec : unique_RVs) {
            mu_unique += spec.mu;
            sigma_unique += spec.sigma;
        }
        mu += mu_shared + mu_unique;
        sigma += sigma_shared + sigma_unique;
        // cout<<"mu = "<<mu<<", sigma = "<<sigma<<endl;
        // cout<<"mu_shared = "<<mu_shared<<", sigma_shared = "<<sigma_shared<<endl;
    }

} sum_gaussian_RVs_v2;

typedef struct joint_exps_gaussian_RVs_struct_v2 {
    // gaussian_RV NULL_TARGET;
    // std::vector<gaussian_RV> const &target;
    sum_gaussian_RVs_v2 const &X_1;
    sum_gaussian_RVs_v2 const &X_2;

    Eigen::VectorXd mean_vector; // column vector
    Eigen::MatrixXd cov_matrix;  // 2x2 matrix

    joint_exps_gaussian_RVs_struct_v2(
        sum_gaussian_RVs_v2 const &_X_1,
        sum_gaussian_RVs_v2 const &_X_2)
        : // target(*(new vector<gaussian_RV>)),
          X_1(_X_1), X_2(_X_2) {

        // shared_RVs must be the same across the two experiments
        assert(&X_1.shared_RVs == &X_2.shared_RVs);
        // shared spectators must be the same across experimetns
        assert(X_1.mu_shared == X_2.mu_shared);
        assert(X_1.sigma_shared == X_2.sigma_shared);

        mean_vector.resize(2);
        cov_matrix.resize(2, 2);

        mean_vector << X_1.mu, X_2.mu;
        cov_matrix << X_1.sigma, X_1.sigma_shared,
            X_2.sigma_shared, X_2.sigma;
        // delete &target;
    }
    // joint_exps_gaussian_RVs_struct_v2(
    //     std::vector<gaussian_RV> const &_target,
    //     sum_gaussian_RVs_v2 const &_X_1,
    //     sum_gaussian_RVs_v2 const &_X_2)
    //     : target(_target), X_1(_X_1), X_2(_X_2) {

    //     // shared_RVs must be the same across the two experiments
    //     assert(&X_1.shared_RVs == &X_2.shared_RVs);
    //     // shared spectators must be the same across experimetns
    //     assert(X_1.mu_shared == X_2.mu_shared);
    //     assert(X_1.sigma_shared == X_2.sigma_shared);
    //     assert(target.size() == 1);

    //     mean_vector.resize(3);
    //     cov_matrix.resize(3, 3);

    //     mean_vector << target.at(0).mu, X_1.mu, X_2.mu;
    //     cov_matrix << X_1.sigma, X_1.sigma_shared,
    //         X_2.sigma_shared, X_2.sigma;
    //     cov_matrix << target.at(0).sigma, target.at(0).sigma, target.at(0).sigma,
    //         target.at(0).sigma, X_1.sigma, target.at(0).sigma + X_1.sigma_shared,
    //         target.at(0).sigma, target.at(0).sigma + X_2.sigma_shared, X_2.sigma;
    // }

} joint_exps_gaussian_RVs_v2;

typedef struct multivariate_joint_gaussian_RV_struct {
    std::vector<sum_gaussian_RVs_v2> const &X_vec;

    Eigen::VectorXd mean_vector; // X_vec.size() vector
    Eigen::MatrixXd cov_matrix;  // X_vec.size() * X_vec.size() matrix

    multivariate_joint_gaussian_RV_struct(std::vector<sum_gaussian_RVs_v2> const &_X_vec) : X_vec(_X_vec) {

        // shared_mu and shared_sigma should be the same across all sum_RVs
        assert(std::all_of(X_vec.begin(), X_vec.end(), [&](sum_gaussian_RVs_v2 x) { return (x.mu_shared == X_vec.begin()->mu_shared && x.sigma_shared == X_vec.begin()->sigma_shared); }));

        mean_vector.resize(X_vec.size());
        cov_matrix.resize(X_vec.size(), X_vec.size());
        for (size_t i = 0; i < X_vec.size(); i++) {
            mean_vector(i) = X_vec.at(i).mu;
            for (size_t j = 0; j < X_vec.size(); j++) {
                if (i == j) {
                    cov_matrix(i, j) = X_vec.at(j).sigma;
                } else {
                    cov_matrix(i, j) = X_vec.at(j).sigma_shared;
                }
            }
        }

        // // shared_RVs must be the same across the two experiments
        // assert(&X_1.shared_RVs == &X_2.shared_RVs);
        // // shared spectators must be the same across experimetns
        // assert(X_1.mu_shared == X_2.mu_shared);
        // assert(X_1.sigma_shared == X_2.sigma_shared);
        // mean_vector << X_1.mu, X_2.mu;
        // cov_matrix << X_1.sigma, X_1.sigma_shared,
        //     X_2.sigma_shared, X_2.sigma;
    }
} multivariate_joint_gaussian_RV;
    
void joint_main_multiple_exps();
void joint_main_two_targets();
void test_single_v2();
void joint_main_3exp();

output_str single_calculation_two_targets(normal_params &target_params_1, normal_params &target_params_2, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2);

long double joint_differential_entropy_v2(joint_exps_gaussian_RVs_v2 X_vec);
long double single_differential_entropy_v2(sum_gaussian_RVs_v2 X);
long double single_differential_entropy_v2(gaussian_RV X);
long double single_differential_entropy_v2(vector<gaussian_RV> &X);

long double multivariate_differential_entropy(multivariate_joint_gaussian_RV X);

long double term_1( vector<gaussian_RV> &target_1, multivariate_joint_gaussian_RV &X);

long double term_1( vector<gaussian_RV> &target_1, multivariate_joint_gaussian_RV X) ;

// target only participates in exp_1
long double trivariate_joint_gassian_entropy_one_target(
    vector<gaussian_RV> &target_1,
    vector<gaussian_RV> &shared_s,
    vector<gaussian_RV> &unique_1,
    vector<gaussian_RV> &unique_2);

long double gaussian_cond_entropy(
    vector<gaussian_RV> &target_1,
    vector<gaussian_RV> &target_2,
    vector<gaussian_RV> &shared_s,
    vector<gaussian_RV> &unique_1,
    vector<gaussian_RV> &unique_2);

long double trivariate_joint_gassian_entropy(
    vector<gaussian_RV> &target_1,
    vector<gaussian_RV> &target_2,
    vector<gaussian_RV> &shared_s,
    vector<gaussian_RV> &unique_1,
    vector<gaussian_RV> &unique_2);

output_str single_calculation_one_target_one_exp_second_not_first(normal_params &target_params_1, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2) ;
void writeData_joint_norm_three_exps(vector<output_str_3exp> out_str, string experiment_name);

#endif