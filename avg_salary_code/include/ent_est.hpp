#ifndef ENT_EST_H
#define ENT_EST_H

// #include "lognormal_v2.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sf.h>
#include <iomanip>
#include <iostream>
#include <sys/time.h> 
#include <list>
#include <numeric>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

vector<double> generate_lognormal_samples(double mu, double sigma, uint num_samples, uint num_RVs);
vector<double> k_NN(vector<double> samples, uint k);
void insert_and_sort(vector<double> &k_NN_distances, double x);
void test_estimator();
double calculate_estimator(double mu, double sigma, uint k, uint numSamples, uint numIterations, uint numRVs) ;

#endif