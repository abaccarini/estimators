#include "sln_mpfr.hpp"

// sigma IS squared
void ln_pdf_mpfr(mpfr_t output, uint64_t z, mpfr_t mu, mpfr_t sigma) {

    if (z == 0) {
        mpfr_set_zero(output, 1);

    } else {

        mpfr_t divisor;
        mpfr_init2(divisor, 100);
        mpfr_const_pi(divisor, MPFR_RNDD);            // pi
        mpfr_mul(divisor, divisor, sigma, MPFR_RNDD); // pi*sigma^2
        mpfr_mul_ui(divisor, divisor, 2, MPFR_RNDD);  // 2*pi*sigma^2
        mpfr_sqrt(divisor, divisor, MPFR_RNDD);       // sqrt(pi*sigma^2)
        mpfr_mul_ui(divisor, divisor, z, MPFR_RNDD);  // z*sqrt(pi*sigma^2)

        mpfr_log_ui(output, z, MPFR_RNDD);       // ln(x)
        mpfr_sub(output, output, mu, MPFR_RNDD); // ln(x) - mu

        mpfr_pow_ui(output, output, 2, MPFR_RNDD);  // (ln(x) - mu)^2
        mpfr_div_ui(output, output, 2, MPFR_RNDD);  // (ln(x) - mu)^2/2
        mpfr_div(output, output, sigma, MPFR_RNDD); //  (ln(x) - mu)^2/(2*sigma)

        mpfr_mul_si(output, output, -1, MPFR_RNDD); // -1 * (ln(x) - mu)^2/(2*sigma)

        mpfr_exp(output, output, MPFR_RNDD); // exp(-1 * (ln(x) - mu)^2/(2*sigma))

        mpfr_div(output, output, divisor, MPFR_RNDD); // exp(-1 * (ln(x) - mu)^2/(2*sigma)) / z*sqrt(pi*sigma^2)

        mpfr_clear(divisor);
        mpfr_free_cache();
    }
}

void ln_pdf_mpfr(mpfr_t output, mpfr_t z, mpfr_t mu, mpfr_t sigma) {

    mpfr_t eps;
    mpfr_init2(eps, PREC);
    // mpfr_set_d(eps, 0.00000001, MPFR_RNDD);
    mpfr_set_str(eps, "0.00000001", 10, MPFR_RNDD);
    if (mpfr_less_p(z, eps)) {
        mpfr_set_zero(output, 1);

    } else {

        mpfr_t divisor;
        mpfr_init2(divisor, 100);
        mpfr_const_pi(divisor, MPFR_RNDD);            // pi
        mpfr_mul(divisor, divisor, sigma, MPFR_RNDD); // pi*sigma^2
        mpfr_mul_ui(divisor, divisor, 2, MPFR_RNDD);  // 2*pi*sigma^2
        mpfr_sqrt(divisor, divisor, MPFR_RNDD);       // sqrt(pi*sigma^2)
        mpfr_mul(divisor, divisor, z, MPFR_RNDD);  // z*sqrt(pi*sigma^2)

        mpfr_log(output, z, MPFR_RNDD);       // ln(x)
        mpfr_sub(output, output, mu, MPFR_RNDD); // ln(x) - mu

        mpfr_pow_ui(output, output, 2, MPFR_RNDD);  // (ln(x) - mu)^2
        mpfr_div_ui(output, output, 2, MPFR_RNDD);  // (ln(x) - mu)^2/2
        mpfr_div(output, output, sigma, MPFR_RNDD); //  (ln(x) - mu)^2/(2*sigma)

        mpfr_mul_si(output, output, -1, MPFR_RNDD); // -1 * (ln(x) - mu)^2/(2*sigma)

        mpfr_exp(output, output, MPFR_RNDD); // exp(-1 * (ln(x) - mu)^2/(2*sigma))

        mpfr_div(output, output, divisor, MPFR_RNDD); // exp(-1 * (ln(x) - mu)^2/(2*sigma)) / z*sqrt(pi*sigma^2)

        mpfr_clear(divisor);
        mpfr_free_cache();
    }
    mpfr_clear(eps);

}
void gfunc_mpfr(mpfr_t output, mpfr_t input) {
    mpfr_t eps;
    mpfr_init2(eps, PREC);
    mpfr_set_str(eps, "0.00000001", 10, MPFR_RNDD);

    if (mpfr_less_p(input, eps)) {
        mpfr_set_zero(output, 1);

    } else {

        mpfr_log2(output, input, MPFR_RNDD);
        mpfr_mul(output, output, input, MPFR_RNDD); // z*sqrt(pi*sigma^2)
    }

    mpfr_clear(eps);
}

void get_fw_params_mpfr(mpfr_t mu_out, mpfr_t sigma_out, mpfr_t mu, mpfr_t sigma, uint16_t L) {

    mpfr_t var, temp_sigma_out, temp_mu_out;
    mpfr_init2(var, 100);
    mpfr_init2(temp_sigma_out, 100);
    mpfr_init2(temp_mu_out, 100);
    // mpfr_set_d(e, 1.0, MPFR_RNDD);
    // mpfr_exp(e, e, MPFR_RNDD); // calculating e
    // mpfr_out_str(stdout, 10, 0, e, MPFR_RNDD);
    // putchar('\n');
    // mpfr_clear(e);
    // mpfr_free_cache();

    mpfr_expm1(temp_sigma_out, sigma, MPFR_RNDD);              // calculates e^op - 1
    mpfr_div_ui(temp_sigma_out, temp_sigma_out, L, MPFR_RNDD); // divides by L
    mpfr_add_ui(temp_sigma_out, temp_sigma_out, 1, MPFR_RNDD); // adds 1
    mpfr_log(temp_sigma_out, temp_sigma_out, MPFR_RNDD);       // takes ln

    mpfr_sub(var, sigma, temp_sigma_out, MPFR_RNDD);
    mpfr_mul_d(var, var, 0.5, MPFR_RNDD);

    mpfr_exp(temp_mu_out, mu, MPFR_RNDD); // calculates e^op
    mpfr_mul_ui(temp_mu_out, temp_mu_out, L, MPFR_RNDD);
    mpfr_log(temp_mu_out, temp_mu_out, MPFR_RNDD); // takes ln
    mpfr_add(temp_mu_out, temp_mu_out, var, MPFR_RNDD);

    mpfr_swap(temp_mu_out, mu_out);
    mpfr_swap(temp_sigma_out, sigma_out);

    mpfr_clear(temp_mu_out);
    mpfr_clear(temp_sigma_out);
    mpfr_clear(var);
    mpfr_free_cache();
}

void test_sln_mpfr(int N) {
    mpfr_t mu, sigma, mu_round, sigma_round;

    // uint16_t L = 5;
    mpfr_init2(sigma, PREC);
    mpfr_init2(mu, PREC);
    mpfr_init2(sigma_round, PREC);
    mpfr_init2(mu_round, PREC);
    // mpfr_set_flt(sigma, 1.06, MPFR_RNDD);

    mpfr_set_d(mu, ((N - 1)) / 2, MPFR_RNDD); // setting to middle of range
    mpfr_set_str(sigma, "0.5", 10, MPFR_RNDD);

    uint16_t numTargets = 1;
    uint16_t numSpectators = 0;

    mpfr_t H_T, H_S, H_T_S, awae, temp;

    mpfr_init2(H_T, PREC);
    mpfr_init2(H_S, PREC);
    mpfr_init2(H_T_S, PREC);
    mpfr_init2(awae, PREC);
    mpfr_init2(temp, PREC);

    uint16_t cutoff = 3;
    for (uint16_t i = 1; i < cutoff; i++) {
        numSpectators = i;
        // cout << numSpectators << endl;
        // auto start = std::chrono::system_clock::now();

        mpfr_set_zero(H_T, 1);
        mpfr_set_zero(H_S, 1);
        mpfr_set_zero(H_T_S, 1);
        mpfr_set_zero(awae, 1);
        mpfr_set_zero(temp, 1);
        // printf("H_T = ");
        // mpfr_out_str(stdout, 10, 0, H_T, MPFR_RNDD);
        // putchar('\n');
        for (uint64_t i = 0; i <= (N - 1); i++) {

            ln_pdf_mpfr(temp, i, mu, sigma);
            // printf("temp = ");
            // mpfr_out_str(stdout, 10, 0, temp, MPFR_RNDD);
            // putchar('\n');

            gfunc_mpfr(temp, temp);

            // printf("g(temp) = ");
            // mpfr_out_str(stdout, 10, 0, temp, MPFR_RNDD);
            // putchar('\n');

            mpfr_add(H_T, H_T, temp, MPFR_RNDD);

            // printf("H_T = ");
            // mpfr_out_str(stdout, 10, 0, H_T, MPFR_RNDD);
            // putchar('\n');

            mpfr_set_zero(temp, 1);
        }
        mpfr_mul_si(H_T, H_T, -1, MPFR_RNDD); // -1 * (ln(x) - mu)^2/(2*sigma)

        // H_T = (-1.0) * H_T;
        // cout << "-------------" << endl;

        get_fw_params_mpfr(mu_round, sigma_round, mu, sigma, numSpectators);

        for (uint64_t i = 0; i <= ((N - 1) * numSpectators); i++) {
            ln_pdf_mpfr(temp, i, mu_round, sigma_round);
            gfunc_mpfr(temp, temp);
            mpfr_add(H_S, H_S, temp, MPFR_RNDD);
        }
        mpfr_mul_si(H_S, H_S, -1, MPFR_RNDD); // -1 * (ln(x) - mu)^2/(2*sigma)
        // H_S = (-1.0) * H_S;
        // cout << "-------------" << endl;

        get_fw_params_mpfr(mu_round, sigma_round, mu, sigma, numSpectators + numTargets);

        for (uint64_t i = 0; i <= (N - 1) * (numSpectators + numTargets); i++) {
            ln_pdf_mpfr(temp, i, mu_round, sigma_round);
            gfunc_mpfr(temp, temp);
            mpfr_add(H_T_S, H_T_S, temp, MPFR_RNDD);
        }
        mpfr_mul_si(H_T_S, H_T_S, -1, MPFR_RNDD); // -1 * (ln(x) - mu)^2/(2*sigma)
        // H_T_S = (-1.0) * H_T_S;
        // cout << "-------------" << endl;
        mpfr_out_str(stdout, 10, 0, H_T, MPFR_RNDD);
        putchar('\n');
        mpfr_out_str(stdout, 10, 0, H_S, MPFR_RNDD);
        putchar('\n');
        mpfr_out_str(stdout, 10, 0, H_T_S, MPFR_RNDD);
        putchar('\n');

        mpfr_add(awae, H_T, H_S, MPFR_RNDD);
        mpfr_sub(awae, awae, H_T_S, MPFR_RNDD);
        printf("awae = ");
        mpfr_out_str(stdout, 10, 0, awae, MPFR_RNDD);
        putchar('\n');

        cout << "-------------" << endl;

        // std::cout << i << ": "
        //   << "H_S = " << H_S << "\t "
        //   << "H_T_S = " << H_T_S << "\t ";
    }

    // uint16_tf("sigma_in = ");
    // mpfr_out_str(stdout, 10, 0, sigma, MPFR_RNDD);
    // putchar('\n');

    // mpfr_out_str(stdout, 10, 0, mu, MPFR_RNDD);
    // putchar('\n');

    // get_fw_params_mpfr(mu, sigma, mu, sigma, L);
    // printf("sigma_out = ");
    // mpfr_out_str(stdout, 10, 0, sigma, MPFR_RNDD);
    // putchar('\n');

    // printf("mu_out = ");
    // mpfr_out_str(stdout, 10, 0, mu, MPFR_RNDD);
    // putchar('\n');

    mpfr_clear(sigma);
    mpfr_clear(mu);
    mpfr_clear(sigma_round);
    mpfr_clear(mu_round);

    mpfr_clear(H_T);
    mpfr_clear(H_S);
    mpfr_clear(H_T_S);
    mpfr_clear(awae);
    mpfr_clear(temp);

    mpfr_free_cache();
}
void test_sln_pdf_mpfr() {
    mpfr_t mu, sigma, x, f_of_x;

    // uint16_t L = 5;
    mpfr_init2(sigma, PREC);
    mpfr_init2(mu, PREC);
    mpfr_init2(x, PREC);
    mpfr_init2(f_of_x, PREC);
    // mpfr_set_flt(sigma, 1.06, MPFR_RNDD);

    // mpfr_set_d(mu, ((N - 1)) / 2, MPFR_RNDD); // setting to middle of range
    mpfr_set_str(mu, "0.0", 10, MPFR_RNDD);
    mpfr_set_str(sigma, "0.25", 10, MPFR_RNDD);
    mpfr_set_str(x, "1.0", 10, MPFR_RNDD);
    mpfr_pow_ui(sigma, sigma, 2, MPFR_RNDD);

    ln_pdf_mpfr(f_of_x, x, mu, sigma);

    mpfr_printf("f_of_x = %.20RNf \n", f_of_x );

    // mpfr_t eps;
    // mpfr_init2(eps, PREC);
    // // mpfr_set_d(eps, 0.00000001, MPFR_RNDD);
    // mpfr_set_str(eps, "0.0000001", 10, MPFR_RNDD);

    // mpfr_printf( "eps = %.10RNf \n", eps );
    // // printf("eps = ");
    // // mpfr_out_str(stdout, 10, 0, eps, MPFR_RNDD);

    // mpfr_clear(eps);

    
    mpfr_clear(sigma);
    mpfr_clear(mu);
    mpfr_clear(x);
    mpfr_clear(f_of_x);
}
