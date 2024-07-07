// SLN = sum (of) lognormals
#ifndef SLN_MPFR_H_
#define SLN_MPFR_H_

#include "constants.hpp"
#include <chrono>
#include <cmath>
#include <gmp.h>
#include <gmpxx.h>
#include <gsl/gsl_integration.h>
#include <iostream>
#include <mpfr.h>
#include <mpir.h>
#include <string>
#include <vector>
#define PREC 220

using namespace std;

void test_sln_mpfr(int);

void test_sln_pdf_mpfr();
void gfunc_mpfr(mpfr_t output, mpfr_t input) ;

void ln_pdfn_mpfr(mpfr_t output, uint64_t z, mpfr_t mu, mpfr_t sigma);

void ln_pdfn_mpfr(mpfr_t output, mpfr_t z, mpfr_t mu, mpfr_t sigma);
void get_fw_paramsn_mpfr(mpfr_t mu_out, mpfr_t sigma_out, mpfr_t mu, mpfr_t sigma, uint16_t L);

#endif