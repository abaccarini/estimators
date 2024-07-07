#ifndef NORMAL_H_
#define NORMAL_H_

#include "constants.hpp"
// #include "matplotlibcpp.h"
#include "plot.hpp"
#include "poisson.hpp"
#include "sln.hpp"
#include "utilities.hpp"
// #include <bits/stdc++.h>
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

using namespace std;

double normal_pdf(double x, void *p);
double normal_pdf_eps(double x, void *p);
double log_normal_pdf(double x, void *p);

void debug_normal();
void test_normal(int N, double mu, double sigma);
double evaluate_normal_pmf(gsl_integration_workspace *w, uint w_size, gsl_function *F, uint i, double delta, double left_bound);

typedef struct normal_params_struct {
    double mu;
    double sigma;
    double eps;
} normal_params;

double differential_entropy_normal(gsl_integration_workspace *w, uint w_size, normal_params params);
double differential_entropy_normal_new(normal_params params);

#endif