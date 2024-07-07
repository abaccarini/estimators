#ifndef JOINT_POISSON_H_
#define JOINT_POISSON_H_

#include "constants.hpp"
#include "plot.hpp"
#include "poisson.hpp"
#include "sln_mpfr.hpp"
#include "types.hpp"
#include "utilities.hpp"
#include <Eigen/Dense>
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

#include <vector>

extern "C" {
    #include "mpfr_funcs.h" 
}

// #define POISSON_PREC 64
// #define MPFR_FLAG 1

void init_mpfr();
void exit_mpfr();

typedef struct poisson_params_v2_struct {
    uint lambda = 0;
    uint N = 0;
} poisson_params_v2;

typedef struct poisson_RV_v2_struct {
    uint lambda = 0;
    uint N = 0;
} poisson_RV_v2;

typedef struct sum_poisson_RVs_v2_struct {
    // poisson_RV_v2 const &target;
    std::vector<poisson_RV_v2> const &shared_RVs;
    std::vector<poisson_RV_v2> const &unique_RVs;

    double lambda_unique = 0.0;
    double lambda_shared = 0.0;
    double lambda_total = 0.0;

    int N = 0;

    sum_poisson_RVs_v2_struct(std::vector<poisson_RV_v2> const &_shared_RVs, std::vector<poisson_RV_v2> const &_unique_RVs) : shared_RVs(_shared_RVs), unique_RVs(_unique_RVs) {
        for (auto &spec : shared_RVs) {
            lambda_shared += spec.lambda;
            N += spec.N - 1;
        }
        for (auto &spec : unique_RVs) {
            lambda_unique += spec.lambda;
            N += spec.N - 1;
        }
        lambda_total = lambda_shared + lambda_unique;
        // cout<<"shared_RVs.size() "<<shared_RVs.size()<<endl;
    }

} sum_poisson_RVs_v2;

typedef struct joint_poisson_RV_v2_struct {
    sum_poisson_RVs_v2 const &X_1;
    sum_poisson_RVs_v2 const &X_2;

    uint lambda_X_1 = 0;
    uint lambda_X_2 = 0;
    uint lambda_X_12 = 0;
    uint lambda_bar = 0;

    uint num_shared = 0;

    joint_poisson_RV_v2_struct(
        sum_poisson_RVs_v2 const &_X_1,
        sum_poisson_RVs_v2 const &_X_2)
        : X_1(_X_1), X_2(_X_2) {

        // shared RVs must be the same across sums
        assert(&X_1.shared_RVs == &X_2.shared_RVs);
        // assert(X_1.shared_RVs.size() == X_2.shared_RVs.size());
        // this check may be redundant, may delete later
        assert(X_1.lambda_shared == X_2.lambda_shared);

        num_shared = X_1.shared_RVs.size();
        // cout<<"num_shared "<<num_shared<<endl;

        lambda_X_1 = X_1.lambda_unique;
        lambda_X_2 = X_2.lambda_unique;
        lambda_X_12 = X_2.lambda_shared;
        lambda_bar = lambda_X_1 + lambda_X_2 + lambda_X_12;
    }

} joint_poisson_RV_v2;

long double shannon_entropy(poisson_RV_v2 X);
long double shannon_entropy(sum_poisson_RVs_v2 X);
long double shannon_entropy(vector<poisson_RV_v2> &X);

long double poisson_pmf(int x, poisson_RV_v2 X);
long double poisson_pmf(int x, vector<poisson_RV_v2> &X);
long double poisson_pmf(int x, sum_poisson_RVs_v2 X);

long double poisson_pmf_mpfr(int x, sum_poisson_RVs_v2 X);
long double poisson_pmf_mpfr(int x, poisson_RV_v2 X);
long double poisson_pmf_mpfr(int x, vector<poisson_RV_v2> &X);

long double joint_poisson_pmf_v2(uint x_1, uint x_2, joint_poisson_RV_v2 &X);

long double poisson_bivariate_H(joint_poisson_RV_v2 &X);
long double poisson_bivariate_H_std(joint_poisson_RV_v2 &X);
long double poisson_bivariate_H_mpfr(joint_poisson_RV_v2 &X);

long double poisson_bivariate_H_std(joint_poisson_RV_v2 &X);
long double poisson_bivariate_H_mpfr(joint_poisson_RV_v2 &X);

long double poisson_rho_std(uint x_i, uint x_j, joint_poisson_RV_v2 &X);

long double poisson_trivariate_H(vector<poisson_RV_v2> target, vector<poisson_RV_v2> shared_s, vector<poisson_RV_v2> unique_1, vector<poisson_RV_v2> unique_2);

void joint_poisson_main_v2();

typedef struct poisson_output_v2_struct {
    poisson_params_v2 target;
    poisson_params_v2 shared_params;
    poisson_params_v2 ex1_params;
    poisson_params_v2 ex2_params;
    uint num_shared;
    uint num_spec_1;
    uint num_spec_2;
    long double target_init = 0.0;
    long double awae = 0.0;
    long double trivariate_shannon_ent = 0.0;
    long double bivariate_shannon_ent = 0.0;
    long double shannon_ent_cond = 0.0;
} poisson_output_v2;
poisson_output_v2 single_calculation_v2(poisson_params_v2 &target_params, poisson_params_v2 &shared_s_params, poisson_params_v2 &unique_1_s_params, poisson_params_v2 &unique_2_s_params, uint num_shared, uint num_spec_1, uint num_spec_2);

#endif
