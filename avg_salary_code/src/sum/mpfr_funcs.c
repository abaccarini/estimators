#include "mpfr_funcs.h"

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

void c_poisson_bivariate_H_mpfr(double *ret_val, uint X_1_N, uint X_2_N, uint lambda_X_1, uint lambda_X_2, uint lambda_X_12, uint lambda_bar) {
    
mpfr_t output, temp_inner, rho_sum, temp_rho_1, temp_rho_2, alpha, eps;

    mpfr_init2(output, POISSON_PREC);
    mpfr_init2(temp_inner, POISSON_PREC);
    mpfr_init2(rho_sum, POISSON_PREC);
    mpfr_init2(temp_rho_1, POISSON_PREC);
    mpfr_init2(temp_rho_2, POISSON_PREC);
    mpfr_init2(alpha, POISSON_PREC);
    mpfr_init2(eps, POISSON_PREC);
    mpfr_set_str(eps, "0.00000001", 10, MPFR_RNDD);
    
    // double ret_val;

    mpfr_set_zero(output, 1);
    mpfr_set_zero(rho_sum, 1);

    for (uint x_i = 0; x_i <= X_1_N; x_i++) {
        for (uint x_j = 0; x_j <= X_2_N; x_j++) {
            mpfr_set_zero(temp_rho_1, 1);
            mpfr_set_zero(temp_rho_2, 1);

            for (int i = 0; i <= min(x_i, x_j); i++) {

                mpfr_set_zero(temp_inner, 1);

                mpfr_set_d(alpha, (double)(lambda_X_1), MPFR_RNDD);
                mpfr_log(alpha, alpha, MPFR_RNDD);
                mpfr_mul_si(alpha, alpha, x_i - i, MPFR_RNDD);
                mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

                mpfr_set_d(alpha, (double)(lambda_X_2), MPFR_RNDD);
                mpfr_log(alpha, alpha, MPFR_RNDD);
                mpfr_mul_si(alpha, alpha, x_j - i, MPFR_RNDD);
                mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

                mpfr_set_d(alpha, (double)(lambda_X_12), MPFR_RNDD);
                mpfr_log(alpha, alpha, MPFR_RNDD);
                mpfr_mul_si(alpha, alpha, i, MPFR_RNDD);
                mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

                mpfr_set_si(alpha, x_i - i + 1, MPFR_RNDD);
                mpfr_lngamma(alpha, alpha, MPFR_RNDD);
                mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

                mpfr_set_si(alpha, x_j - i + 1, MPFR_RNDD);
                mpfr_lngamma(alpha, alpha, MPFR_RNDD);
                mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

                mpfr_set_si(alpha, i + 1, MPFR_RNDD);
                mpfr_lngamma(alpha, alpha, MPFR_RNDD);
                mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

                mpfr_exp(temp_inner, temp_inner, MPFR_RNDD);
                mpfr_add(temp_rho_1, temp_rho_1, temp_inner, MPFR_RNDD);
            }
            // gfunc_mpfr(temp_rho_2, temp_rho_1);

            if (mpfr_less_p(temp_rho_1, eps)) {
                mpfr_set_zero(temp_rho_2, 1);

            } else {

                mpfr_log2(temp_rho_2, temp_rho_1, MPFR_RNDD);
                mpfr_mul(temp_rho_2, temp_rho_2, temp_rho_1, MPFR_RNDD); // z*sqrt(pi*sigma^2)
            }

            mpfr_add(rho_sum, rho_sum, temp_rho_2, MPFR_RNDD);
        }
    }
    mpfr_set_d(alpha, (-1.0) * (int)(lambda_bar), MPFR_RNDD);
    mpfr_exp(alpha, alpha, MPFR_RNDD);

    mpfr_mul(rho_sum, rho_sum, alpha, MPFR_RNDD);

    mpfr_set_d(output, log2(M_E) * (int)(lambda_bar), MPFR_RNDD);
    mpfr_sub(output, output, rho_sum, MPFR_RNDD);

    *ret_val = mpfr_get_ld(output, MPFR_RNDD);




    mpfr_clear(output);
    mpfr_clear(temp_inner);
    mpfr_clear(alpha);
    mpfr_clear(rho_sum);
    mpfr_clear(temp_rho_1);
    mpfr_clear(temp_rho_2);
    mpfr_clear(eps);

    mpfr_free_cache();
    // return ret_val;
}