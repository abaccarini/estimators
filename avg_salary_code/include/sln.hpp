// SLN = sum (of) lognormals
#ifndef SLN_H_
#define SLN_H_

#include "constants.hpp"
#include "plot.hpp"
#include "poisson.hpp"
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

double ln_pdf(double x, void *p);
double ln_pdf(double z, double mu, double sigma);
double ln_pdf_eps(double x, void *p);
double ln_pdf_deriv(double x, void *p);
void ln_pdf_deriv_fdf(double x, void *p, double *f, double *df);
double log_ln_pdf(double x, void *p);

void test_sln(int N, double mu, double sigma);

double uniform_pdf(double x, void *p);

void test_sln_pdf();
void test_gsl();
void test_root_solver();
void update_fw_params(double &mu_out, double &sigma_out, double mu, double sigma, uint16_t L);

double evaluate_ln_pmf(gsl_integration_workspace *w, uint w_size, gsl_function *F, uint i, double delta);
double ln_root_solver(double mu, double sigma, double epsilon, double rel_error, double upper_bound, int max_iter);
double ln_root_solver_derivative(double mu, double sigma, double epsilon, double rel_error, double upper_bound, int max_iter);

// double differential_entropy(gsl_integration_workspace *w, uint w_size, gsl_function *F);
double differential_entropy_ln(gsl_integration_workspace *w, uint w_size, struct ln_params params);

struct ln_params {
    double mu;
    double sigma;
    double eps;
};

struct uniform_params {
    double a;
    double b;
};

#endif