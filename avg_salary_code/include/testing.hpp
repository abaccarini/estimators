#ifndef TESTING_H_
#define TESTING_H_

// libraries
#include <iostream>
#include <stdio.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <numeric>
#include <ostream>
#include <string>
#include <unistd.h>

// headers
#include "entropy.hpp"
#include "plot.hpp"

using namespace std;

output_max median_exp(string experimentName, function<double(vector<int>, vector<int>, vector<int>)> funct, int N, vector<int> a_range, vector<double> probabilities, vector<uint> distFlags, vector<int> participants);

output_max max_exp(string experimentName, function<int(vector<int>, vector<int>, vector<int>)> funct, int N, vector<int> a_range, vector<double> probabilities, vector<uint> distFlags, vector<int> participants, vector<int> output_domain={});
void writeData_max(vector<output_max> out_str, string experiment_name, string distribution, string param_string) ;

void test_misc();
void test_misc_2();
void test_max() ;
void test_median() ;

void test_main();
void test_dot_prod();
void test_comparison();
void test_sum_poisson();
void test_sum();
void singleExperiment(string, function<int(vector<int>, vector<int>, vector<int>)>, int, vector<int>, vector<double>, vector<uint>, vector<int>);
void singleExperiment(string, function<int(vector<int>, vector<int>, vector<int>)>, int, vector<int>, vector<double>, vector<uint>, vector<int>, vector<int>);
void test_advanced();

int comparison(vector<int>, vector<int>, vector<int>);
int summation(vector<int>, vector<int>, vector<int>);
int advanced(vector<int>, vector<int>, vector<int>);
int dot_prod(vector<int>, vector<int>, vector<int>);

int decision_tree(vector<int>, vector<int>, vector<int>);

#endif