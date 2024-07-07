#ifndef JOINT_H_
#define JOINT_H_

#include "constants.hpp"
#include "normal.hpp"
#include "plot.hpp"
#include "poisson.hpp"
#include "sln.hpp"
#include "utilities.hpp"
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

typedef struct gaussian_RV_struct {
    double mu = 0.0;
    double sigma = 0.0;
} gaussian_RV;

typedef struct output_struct {
    normal_params target;
    normal_params shared_params;
    normal_params ex1_params;
    normal_params ex2_params;
    uint num_shared;
    uint num_spec_1;
    uint num_spec_2;
    long double diff_ent_joint_T_exps = 0.0;
    long double diff_ent_joint_exps = 0.0;
    long double diff_ent_cond = 0.0;
    long double target_init = 0.0;
    long double awae = 0.0;
} output_str;




typedef struct sum_gaussian_RVs_struct {
    gaussian_RV const &target;
    std::vector<gaussian_RV> const &shared_spectators;
    std::vector<gaussian_RV> const &unique_spectators;

    double mu = 0;
    double sigma = 0;

    double mu_shared = 0;
    double sigma_shared = 0;

    double mu_unique = 0;
    double sigma_unique = 0;

    sum_gaussian_RVs_struct(gaussian_RV const &_target,
                            std::vector<gaussian_RV> const &_shared_spectators,
                            std::vector<gaussian_RV> const &_unique_spectators) : target(_target),
                                                                                  shared_spectators(_shared_spectators),
                                                                                  unique_spectators(_unique_spectators) {
        mu = target.mu;
        sigma = target.sigma;
        for (auto &spec : shared_spectators) {
            // this will never change if therer are no shared spectators
            mu_shared += spec.mu;
            // this will never change if therer are no shared spectators
            sigma_shared += spec.sigma;
        }
        for (auto &spec : unique_spectators) {

            // this will never change if therer are no shared spectators
            mu_unique += spec.mu;
            // this will never change if therer are no shared spectators
            sigma_unique += spec.sigma;
        }
        mu += mu_shared + mu_unique;
        sigma += sigma_shared + sigma_unique;

        // sigma = sqrt(sigma);
    }

} sum_gaussian_RVs;

typedef struct joint_exps_gaussian_RVs_struct {
    // gaussian_RV NULL_TARGET;
    gaussian_RV const &target;
    sum_gaussian_RVs const &exp_1;
    sum_gaussian_RVs const &exp_2;

    Eigen::VectorXd mean_vector; // column vector
    Eigen::MatrixXd cov_matrix;  // 2x2 matrix

    joint_exps_gaussian_RVs_struct(sum_gaussian_RVs const &_exp_1, sum_gaussian_RVs const &_exp_2)
        : target(*(new gaussian_RV)), exp_1(_exp_1), exp_2(_exp_2) {

        // targets must be the same across the two experiments
        assert(&exp_1.target == &exp_2.target);
        // shared spectators must be the same across experimetns
        assert(&exp_1.shared_spectators == &exp_2.shared_spectators);

        mean_vector.resize(2);
        cov_matrix.resize(2, 2);

        mean_vector << exp_1.mu, exp_2.mu;
        cov_matrix << exp_1.sigma, exp_1.target.sigma + exp_1.sigma_shared,
            exp_2.target.sigma + exp_2.sigma_shared, exp_2.sigma;

        delete &target;
    }

    joint_exps_gaussian_RVs_struct(gaussian_RV const &_target, sum_gaussian_RVs const &_exp_1, sum_gaussian_RVs const &_exp_2) : target(_target), exp_1(_exp_1), exp_2(_exp_2) {

        // targets must be the same across the two experiments
        assert(&exp_1.target == &exp_2.target);
        // input target should be same as exp target
        assert(&target == &exp_1.target);
        // shared spectators must be the same across experimetns
        assert(&exp_1.shared_spectators == &exp_2.shared_spectators);

        mean_vector.resize(3);
        cov_matrix.resize(3, 3);

        mean_vector << target.mu, exp_1.mu, exp_2.mu;
        cov_matrix << target.sigma, target.sigma, target.sigma,
            target.sigma, exp_1.sigma, exp_1.target.sigma + exp_1.sigma_shared,
            target.sigma, exp_2.target.sigma + exp_2.sigma_shared, exp_2.sigma;
    }

} joint_exps_gaussian_RVs;


void joint_main();
void test_single_v1();
double single_differential_entropy(gaussian_RV X);
double single_differential_entropy_spec(sum_gaussian_RVs X);
double single_differential_entropy_spec_target(sum_gaussian_RVs X);

double joint_differential_entropy(joint_exps_gaussian_RVs X_vec);
double cond_entropy(joint_exps_gaussian_RVs X_T_exps, joint_exps_gaussian_RVs X_exps);

output_str single_calculation(normal_params &target_params, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2);


void writeData_joint_norm(vector<output_struct> out_str, string experiment_name);
void writeData_joint_norm_two_targets(vector<output_struct> out_str, string experiment_name) ;

#endif