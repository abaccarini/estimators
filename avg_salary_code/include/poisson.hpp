#ifndef POISSON_H_
#define POISSON_H_

#include "constants.hpp"
#include "plot.hpp"
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

#define POISSON_PREC 100
#define MPFR_FLAG 0

// using namespace std;

typedef struct poisson_params_struct {
    double lambda = 0.0;
    int N = 0;
} poisson_params;


typedef struct poisson_RV_struct {
    double lambda = 0.0;
    int N = 0;
} poisson_RV;

// input is two vectors
// shared vector will either be a single-element vector (just the target)
// or sum of RVs (e.g. shared spectators)
// we group the input RVs into shared/unique
// eg for(X_S + X_S_1,  X_T + X_S + X_S_1), group x_T and X_S_1 together
typedef struct sum_poisson_RVs_struct {
    // poisson_RV const &target;
    std::vector<poisson_RV> const &shared_RVs;
    std::vector<poisson_RV> const &unique_RVs;

    double lambda_unique = 0.0;
    double lambda_shared = 0.0;
    double lambda_total = 0.0;

    int N = 0;

    sum_poisson_RVs_struct( std::vector<poisson_RV> const &_shared_RVs, std::vector<poisson_RV> const &_unique_RVs) : shared_RVs(_shared_RVs), unique_RVs(_unique_RVs) {
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

} sum_poisson_RVs;

typedef struct joint_poisson_RV_struct {
    sum_poisson_RVs const &X_1;
    sum_poisson_RVs const &X_2;

    Eigen::Vector3d lambda_vector; // column vector

    joint_poisson_RV_struct(
        sum_poisson_RVs const &_X_1,
        sum_poisson_RVs const &_X_2)
        : X_1(_X_1), X_2(_X_2) {

        // shared RVs must be the same across sums
        assert(&X_1.shared_RVs == &X_2.shared_RVs);
        // this check may be redundant, may delete later
        assert(X_1.lambda_shared == X_2.lambda_shared);

        lambda_vector << X_1.lambda_unique, X_2.lambda_unique, X_2.lambda_shared;
        // lambda_1, lambda_2, lambda
    }

} joint_poisson_RV;

long double poisson_pmf(int x, double lambda, double numRVs);
long double joint_poisson_pmf_mpfr(int x_1, int x_2, joint_poisson_RV X) ;

long double joint_poisson_pmf(int x_1, int x_2, joint_poisson_RV X);


long double poisson_pmf_mpfr(int x, sum_poisson_RVs X) ;
long double poisson_pmf_mpfr(int x, poisson_RV X) ;
long double poisson_pmf_mpfr(int x, vector<poisson_RV> &X);

long double poisson_pmf(int x, sum_poisson_RVs X) ;
long double poisson_pmf(int x, poisson_RV X) ;
long double poisson_pmf(int x, vector<poisson_RV> &X) ;


long double shannon_entropy(poisson_RV X) ;
long double shannon_entropy(vector<poisson_RV> &X) ;
long double shannon_entropy(sum_poisson_RVs X) ;

long double trivariate_joint_shannon_entropy(
    vector<poisson_RV> &target,
    vector<poisson_RV> &shared_s,
    vector<poisson_RV> &unique_1,
    vector<poisson_RV> &unique_2);

void joint_poisson_main();
void simple_driver_poisson(int N);



typedef struct poisson_output_struct {
    poisson_params target;
    poisson_params shared_params;
    poisson_params ex1_params;
    poisson_params ex2_params;
    uint num_shared;
    uint num_spec_1;
    uint num_spec_2;
    long double target_init = 0.0;
    long double awae = 0.0;
    long double trivariate_shannon_ent = 0.0;
    long double bivariate_shannon_ent = 0.0;
    long double shannon_ent_cond = 0.0;
} poisson_output;

poisson_output single_calculation(poisson_params &target_params, poisson_params &shared_s_params, poisson_params &unique_1_s_params, poisson_params &unique_2_s_params, uint num_shared, uint num_spec_1, uint num_spec_2);
void writeData_joint_poisson(vector<poisson_output> out_str, string experiment_name);


#endif