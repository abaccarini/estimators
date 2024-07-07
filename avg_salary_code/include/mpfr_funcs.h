#ifndef MPFR_FUNCS_H
#define MPFR_FUNCS_H

#include <stdio.h>
#include <math.h>
#include <gmp.h>
// #include <gmpxx.h>
#include <mpfr.h>

typedef unsigned uint; // for ring size in [1,30]; 

#define POISSON_PREC 100

void c_poisson_bivariate_H_mpfr(double *ret_val, uint X_1_N, uint X_2_N, uint lambda_X_1, uint lambda_X_2, uint lambda_X_12, uint lambda_bar);

#endif
