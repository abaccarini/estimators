#ifndef TEST_MULT_GAUSS_H_
#define TEST_MULT_GAUSS_H_

// #include "joint_gaussian_v2.hpp"
#include <Eigen/Dense>
#include <algorithm>
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
#include <map>
#include <mpfr.h>
#include <mpir.h>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <vector>

#define GET_BIT(X, N) ((uint(X) >> uint(N)) & uint(1))

using namespace std;
// this can be used for both (O_1, O_2, O_3) AND (S_1 +.., S_2 + .., S_3 + ...), just set T = 0
//

typedef struct output_str_3exp_full_struct {
    double sigma;
    uint S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, t_1, t_2, t_3;
    long double H_X_T = 0.0, H_X_T_O_1 = 0.0, H_X_T_O_12 = 0.0, H_X_T_O_123 = 0.0;
} output_str_3exp_full;

typedef struct output_str_3exp_brute_force_struct {
    double sigma;
    uint S_1, S_2, S_3, S_12, S_13, S_23, S_123, T;
    long double H_X_T = 0.0, H_X_T_O_1 = 0.0, H_X_T_O_12_t11 = 0.0, H_X_T_O_12_t10 = 0.0, H_X_T_O_123_t111 = 0.0, H_X_T_O_123_t110 = 0.0, H_X_T_O_123_t101 = 0.0, H_X_T_O_123_t011 = 0.0, H_X_T_O_123_t100 = 0.0, H_X_T_O_123_t010 = 0.0, H_X_T_O_123_t001 = 0.0;
} output_str_3exp_brute_force;

typedef struct N_exp_data_struct {
    vector<int> spec_values;
    double min_cond_entropy;
} N_exp_data;

bool compareBy_H_T_O_123(const output_str_3exp_full &a, const output_str_3exp_full &b);
bool compareByCondEnt(const N_exp_data &a, const N_exp_data &b);

typedef struct mv_gaussian_struct {
    double sigma;                                   // sigma^2 is the same for all rvs
    uint S_1, S_2, S_3, S_12, S_13, S_23, S_123, T; //
    Eigen::MatrixXd cov_matrix;                     // 3x3 matrix
    vector<uint> T_flags;                           // determines which experiments the target participates in

    mv_gaussian_struct(
        double _sigma, uint _S_1, uint _S_2, uint _S_3, uint _S_12, uint _S_13, uint _S_23, uint _S_123, uint _T, vector<uint> _T_flags) : sigma(_sigma), S_1(_S_1), S_2(_S_2), S_3(_S_3), S_12(_S_12), S_13(_S_13), S_23(_S_23), S_123(_S_123), T(_T), cov_matrix(3, 3), T_flags(_T_flags) {

        // cov_matrix.resize(3, 3)

        cov_matrix(0, 0) = (S_1 + S_12 + S_13 + S_123 + (T * T_flags.at(0))) * sigma; // Var(O_1)
        cov_matrix(1, 1) = (S_2 + S_12 + S_23 + S_123 + (T * T_flags.at(1))) * sigma; // Var(O_2)
        cov_matrix(2, 2) = (S_3 + S_13 + S_23 + S_123 + (T * T_flags.at(2))) * sigma; // Var(O_3)

        cov_matrix(0, 1) = (S_12 + S_123 + (T * T_flags.at(0) * T_flags.at(1))) * sigma; // Cov(O_1, O_2)
        cov_matrix(1, 0) = cov_matrix(0, 1);

        cov_matrix(0, 2) = (S_13 + S_123 + (T * T_flags.at(0) * T_flags.at(2))) * sigma; // Cov(O_1, O_3)
        cov_matrix(2, 0) = cov_matrix(0, 2);

        cov_matrix(1, 2) = (S_23 + S_123 + (T * T_flags.at(1) * T_flags.at(2))) * sigma; // Cov(O_2, O_3)
        cov_matrix(2, 1) = cov_matrix(1, 2);
    }
    mv_gaussian_struct(
        double _sigma, uint _S_1, uint _S_2, uint _S_12, uint _S_13, uint _S_23, uint _S_123, uint _T, vector<uint> _T_flags) : sigma(_sigma), S_1(_S_1), S_2(_S_2), S_3(0), S_12(_S_12), S_13(_S_13), S_23(_S_23), S_123(_S_123), T(_T), cov_matrix(2, 2), T_flags(_T_flags) {

        cov_matrix(0, 0) = (S_1 + S_12 + S_13 + S_123 + (T * T_flags.at(0))) * sigma; // O_1
        cov_matrix(1, 1) = (S_2 + S_12 + S_23 + S_123 + (T * T_flags.at(1))) * sigma; // O_2

        cov_matrix(0, 1) = (S_12 + S_123 + (T * T_flags.at(0) * T_flags.at(1))) * sigma; // Cov(O_1, O_2)
        cov_matrix(1, 0) = cov_matrix(0, 1);
    }

} mv_gaussian;

typedef struct N_gaussian_struct {
    // double sigma; // sigma^2 is the same for all rvs
    // uint T;
    // vector<uint> T_flags;
    // vector<pair<string, int>> spec_map; // maps a spectator var string to its size (e.g. '123' -> |s_123|)
    Eigen::MatrixXd cov_matrix; // 3x3 matrix
                                // determines which experiments the target participates in

    N_gaussian_struct(double sigma, uint N, uint Delta, uint T, vector<uint> T_flags, vector<pair<string, int>> spec_map) : // sigma(_sigma), T(_T), T_flags(_T_flags), spec_map(_spec_map),
                                                                                                                            cov_matrix(N, N) {
        for (size_t i = 0; i < N; i++) {
            for (size_t j = i; j < N; j++) { // only looping over upper diagonal (including diag)
                cov_matrix(i, j) = 0;
                if (i == j) {
                    cov_matrix(i, j) = (Delta + T * T_flags.at(i)) * sigma;

                } else {
                    for (size_t k = 0; k < spec_map.size(); k++) {
                        // if (spec_map.at(k).first.contains(to_string(i + 1)) and spec_map.at(k).first.contains(to_string(j + 1))) {
                        if ((spec_map.at(k).first.find(to_string(i + 1)) != std::string::npos) and (spec_map.at(k).first.find(to_string(j + 1)) != std::string::npos)) {
                            cov_matrix(i, j) += spec_map.at(k).second;
                        }
                    }
                    cov_matrix(i, j) = (cov_matrix(i, j) + (T * T_flags.at(i) * T_flags.at(j))) * sigma;
                    cov_matrix(j, i) = cov_matrix(i, j);
                }
            }
        }
    }

} N_gaussian;

long double mv_differential_entropy(mv_gaussian X);
long double single_differential_entropy(uint n, double sigma);

output_str_3exp_full single_calculation(double sigma, uint S_1, uint S_2, uint S_3, uint S_12, uint S_13, uint S_23, uint S_123, uint T, vector<uint> T_flags);
output_str_3exp_brute_force single_calculation_brute_force(double sigma, uint S_1, uint S_2, uint S_3, uint S_12, uint S_13, uint S_23, uint S_123, uint T);
void brute_force(uint total_num_spec, uint num_targets, double sigma, uint topK);

void writeData_joint_normal_3xp(vector<output_str_3exp_full> out_str, string experiment_name);
void writeData_joint_normal_3xp(vector<output_str_3exp_brute_force> out_str, string experiment_name);

void test_mult_gaussian_main();

vector<string> generate_spec_combo(uint Delta);

void test_N_exp_main();
void generate_power_set(vector<int> set, int n, vector<string> &output);

// void doRecursion(int index, int Delta, int minus_term, int N, double sigma, vector<pair<string, int>> &spec_map, ofstream &stream);
// void doRecursion(int index, int Delta, int minus_term, int N, double sigma, vector<pair<string, int>> &spec_map, uint top_N, vector<N_exp_data> &out_data), ;
void doRecursion(int index, int Delta, int minus_term, int N, double sigma, vector<pair<string, int>> &spec_map, uint top_N, vector<N_exp_data> &out_data);

void single_calculation(double sigma, uint N, uint Delta, vector<pair<string, int>> &spec_map, uint top_N, vector<N_exp_data> &out_data);

bool compareByCondEnt(const N_exp_data &a, const N_exp_data &b);

void wirteData_N_exp(vector<N_exp_data> out_data, vector<pair<string, int>> &spec_map, double sigma, uint N, uint Delta);

void evaluate_4_exp(int Delta, int N, double sigma, vector<pair<string, int>> &svar, uint top_N,vector<N_exp_data> &out_data);

#endif
