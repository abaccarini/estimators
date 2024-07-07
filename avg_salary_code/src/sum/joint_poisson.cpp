#include "joint_poisson.hpp"
mpfr_t output, temp_inner, rho_sum, temp_rho_1, temp_rho_2, alpha, eps;

long double poisson_bivariate_H(joint_poisson_RV_v2 &X) {
    if (X.num_shared == 0) {
        long double ta = shannon_entropy(X.X_1) + shannon_entropy(X.X_2);

        return ta;
    } else {
        long double result = 0.0;
        switch (MPFR_FLAG) {
        case 0:
            result = poisson_bivariate_H_std(X);
            break;

        case 1:
            result = poisson_bivariate_H_mpfr(X);
            break;
        case 2:
            // result = poisson_bivariate_H_gmp(X);
            break;
        default:
            printf("INVALID MPFR_FLAG\n");
            break;
        }
        return result;
    }
}

long double shannon_entropy(poisson_RV_v2 X) {
    long double result = 0.0;

    if (MPFR_FLAG) {
        // printf("USING MPFR\n");

        for (size_t x = 0; x <= X.N; x++) {
            result += gFunc(poisson_pmf_mpfr(x, X));
        }
    } else {
        for (size_t x = 0; x <= X.N; x++) {
            result += gFunc(poisson_pmf(x, X));
        }
    }

    return (-1.0) * result;
}

long double shannon_entropy(sum_poisson_RVs_v2 X) {
    long double result = 0.0;
    if (MPFR_FLAG) {
        // printf("USING MPFR\n");

        for (int x = 0; x <= X.N; x++) {
            result += gFunc(poisson_pmf_mpfr(x, X));
        }
    } else {
        for (int x = 0; x <= X.N; x++) {
            result += gFunc(poisson_pmf(x, X));
        }
    }
    return (-1.0) * result;
}

long double shannon_entropy(vector<poisson_RV_v2> &X) {
    assert(X.size() == 1);
    long double result = 0.0;
    if (MPFR_FLAG) {
        // printf("USING MPFR\n");

        for (size_t x = 0; x <= X.at(0).N; x++) {
            result += gFunc(poisson_pmf_mpfr(x, X));
        }
    } else {
        for (size_t x = 0; x <= X.at(0).N; x++) {
            result += gFunc(poisson_pmf(x, X));
        }
    }

    return (-1.0) * result;
}

long double poisson_pmf(int x, sum_poisson_RVs_v2 X) {
    return pow(M_E, x * log(X.lambda_total) - (X.lambda_total) - lgamma(x + 1.0));
}

// single poisson RV
long double poisson_pmf(int x, poisson_RV_v2 X) {
    return pow(M_E, x * log(X.lambda) - (X.lambda) - lgamma(x + 1.0));
}

long double poisson_pmf(int x, vector<poisson_RV_v2> &X) {
    double lambda = 0;
    // summing up all lambdas in vector
    for (auto &n : X)
        lambda += n.lambda;
    return pow(M_E, x * log(lambda) - (lambda)-lgamma(x + 1.0));
}

// long double poisson_pmf_gmp(int x, sum_poisson_RVs_v2 X) {
//     double ret_val;
//     mpf_t out, a;
//     mpf_init2(a, POISSON_PREC);
//     mpf_init2(out, POISSON_PREC);
//     // mpf_set_zero(out, 1);

//     mpf_set_ui(a, X.lambda_total);
//     mpf_log(a, a, MPFR_RNDD);
//     mpf_mul_si(a, a, x, MPFR_RNDD);
//     mpf_add(out, out, a, MPFR_RNDD);

//     mpf_sub_si(out, out, X.lambda_total, MPFR_RNDD);

//     mpf_set_si(a, x + 1, MPFR_RNDD);
//     mpf_lngamma(a, a, MPFR_RNDD);
//     mpf_sub(out, out, a, MPFR_RNDD);

//     mpf_exp(out, out, MPFR_RNDD);
//     ret_val = mpf_get_ld(out, MPFR_RNDD);

//     mpf_clear(out);
//     mpf_clear(a);
//     mpf_free_cache();

//     return ret_val;
// }

long double poisson_pmf_mpfr(int x, sum_poisson_RVs_v2 X) {
    double ret_val;
    // mpfr_t output, alpha;
    // mpfr_init2(alpha, POISSON_PREC);
    // mpfr_init2(output, POISSON_PREC);
    mpfr_set_zero(output, 1);

    mpfr_set_ui(alpha, X.lambda_total, MPFR_RNDD);
    mpfr_log(alpha, alpha, MPFR_RNDD);
    mpfr_mul_si(alpha, alpha, x, MPFR_RNDD);
    mpfr_add(output, output, alpha, MPFR_RNDD);

    mpfr_sub_si(output, output, X.lambda_total, MPFR_RNDD);

    mpfr_set_si(alpha, x + 1, MPFR_RNDD);
    mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    mpfr_sub(output, output, alpha, MPFR_RNDD);

    mpfr_exp(output, output, MPFR_RNDD);
    ret_val = mpfr_get_ld(output, MPFR_RNDD);

    // mpfr_clear(output);
    // mpfr_clear(alpha);
    // mpfr_free_cache();

    return ret_val;
}

// single poisson RV
long double poisson_pmf_mpfr(int x, poisson_RV_v2 X) {
    double ret_val;
    // mpfr_t output, alpha;
    // mpfr_init2(alpha, POISSON_PREC);
    // mpfr_init2(output, POISSON_PREC);
    mpfr_set_zero(output, 1);

    mpfr_set_ui(alpha, X.lambda, MPFR_RNDD);
    mpfr_log(alpha, alpha, MPFR_RNDD);
    mpfr_mul_si(alpha, alpha, x, MPFR_RNDD);
    mpfr_add(output, output, alpha, MPFR_RNDD);

    mpfr_sub_si(output, output, X.lambda, MPFR_RNDD);

    mpfr_set_si(alpha, x + 1, MPFR_RNDD);
    mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    mpfr_sub(output, output, alpha, MPFR_RNDD);

    mpfr_exp(output, output, MPFR_RNDD);
    ret_val = mpfr_get_ld(output, MPFR_RNDD);

    // mpfr_clear(output);
    // mpfr_clear(alpha);
    // mpfr_free_cache();

    return ret_val;
}

long double poisson_pmf_mpfr(int x, vector<poisson_RV_v2> &X) {
    double ret_val;
    // mpfr_t output, alpha;
    // mpfr_init2(alpha, POISSON_PREC);
    // mpfr_init2(output, POISSON_PREC);
    mpfr_set_zero(output, 1);

    double lambda = 0;

    // summing up all lambdas in vector
    for (auto &n : X)
        lambda += n.lambda;

    mpfr_set_ui(alpha, lambda, MPFR_RNDD);
    mpfr_log(alpha, alpha, MPFR_RNDD);
    mpfr_mul_si(alpha, alpha, x, MPFR_RNDD);
    mpfr_add(output, output, alpha, MPFR_RNDD);

    mpfr_sub_si(output, output, lambda, MPFR_RNDD);

    mpfr_set_si(alpha, x + 1, MPFR_RNDD);
    mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    mpfr_sub(output, output, alpha, MPFR_RNDD);

    mpfr_exp(output, output, MPFR_RNDD);
    ret_val = mpfr_get_ld(output, MPFR_RNDD);

    // mpfr_clear(output);
    // mpfr_clear(alpha);
    // mpfr_free_cache();

    return ret_val;
}

long double poisson_bivariate_H_std(joint_poisson_RV_v2 &X) {
    long double result = 0.0;
    for (int x_1 = 0; x_1 <= X.X_1.N; x_1++) {
        for (int x_2 = 0; x_2 <= X.X_2.N; x_2++) {

            result += gFunc(poisson_rho_std(x_1, x_2, X));
            // result += (poisson_rho_std(x_1, x_2, X));
        }
    }
    return X.lambda_bar * log2(M_E) - pow(M_E, (-1.0) * X.lambda_bar) * result;
}

long double poisson_rho_std(uint x_i, uint x_j, joint_poisson_RV_v2 &X) {
    long double result = 0.0;

    for (uint k = 0; k <= std::min(x_i, x_j); k++) {
        result += pow(M_E, ((x_i - k) * log(X.lambda_X_1) + (x_j - k) * log(X.lambda_X_2) + (k)*log(X.lambda_X_12) - lgamma(x_i - k + 1) - lgamma(x_j - k + 1) - lgamma(k + 1)));
        // result += (pow(X.lambda_X_1, x_i - k)*pow(X.lambda_X_2, x_j - k) *pow(X.lambda_X_12, k)   )/ (tgamma(x_i - k + 1) * tgamma(x_j - k + 1)* tgamma(k + 1) );
    }

    return result;
}

void init_mpfr() {
    mpfr_init2(output, POISSON_PREC);
    mpfr_init2(temp_inner, POISSON_PREC);
    mpfr_init2(rho_sum, POISSON_PREC);
    mpfr_init2(temp_rho_1, POISSON_PREC);
    mpfr_init2(temp_rho_2, POISSON_PREC);
    mpfr_init2(alpha, POISSON_PREC);
    mpfr_init2(eps, PREC);
    mpfr_set_str(eps, "0.00000001", 10, MPFR_RNDD);
}

void exit_mpfr() {

    mpfr_clear(output);
    mpfr_clear(temp_inner);
    mpfr_clear(alpha);
    mpfr_clear(rho_sum);
    mpfr_clear(temp_rho_1);
    mpfr_clear(temp_rho_2);
    mpfr_clear(eps);

    mpfr_free_cache();
}

long double poisson_bivariate_H_mpfr(joint_poisson_RV_v2 &X) {

    double ret_val = 0;

    c_poisson_bivariate_H_mpfr(&ret_val, X.X_1.N, X.X_2.N, X.lambda_X_1, X.lambda_X_2, X.lambda_X_12, X.lambda_bar);

    // mpfr_set_zero(output, 1);
    // mpfr_set_zero(rho_sum, 1);

    // for (uint x_i = 0; x_i <= X.X_1.N; x_i++) {
    //     for (uint x_j = 0; x_j <= X.X_2.N; x_j++) {
    //         mpfr_set_zero(temp_rho_1, 1);
    //         mpfr_set_zero(temp_rho_2, 1);

    //         for (int i = 0; i <= std::min(x_i, x_j); i++) {

    //             mpfr_set_zero(temp_inner, 1);

    //             mpfr_set_d(alpha, double(X.lambda_X_1), MPFR_RNDD);
    //             mpfr_log(alpha, alpha, MPFR_RNDD);
    //             mpfr_mul_si(alpha, alpha, x_i - i, MPFR_RNDD);
    //             mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

    //             mpfr_set_d(alpha, double(X.lambda_X_2), MPFR_RNDD);
    //             mpfr_log(alpha, alpha, MPFR_RNDD);
    //             mpfr_mul_si(alpha, alpha, x_j - i, MPFR_RNDD);
    //             mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

    //             mpfr_set_d(alpha, double(X.lambda_X_12), MPFR_RNDD);
    //             mpfr_log(alpha, alpha, MPFR_RNDD);
    //             mpfr_mul_si(alpha, alpha, i, MPFR_RNDD);
    //             mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

    //             mpfr_set_si(alpha, x_i - i + 1, MPFR_RNDD);
    //             mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    //             mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

    //             mpfr_set_si(alpha, x_j - i + 1, MPFR_RNDD);
    //             mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    //             mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

    //             mpfr_set_si(alpha, i + 1, MPFR_RNDD);
    //             mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    //             mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

    //             mpfr_exp(temp_inner, temp_inner, MPFR_RNDD);
    //             mpfr_add(temp_rho_1, temp_rho_1, temp_inner, MPFR_RNDD);
    //         }
    //         // gfunc_mpfr(temp_rho_2, temp_rho_1);

    //         if (mpfr_less_p(temp_rho_1, eps)) {
    //             mpfr_set_zero(temp_rho_2, 1);

    //         } else {

    //             mpfr_log2(temp_rho_2, temp_rho_1, MPFR_RNDD);
    //             mpfr_mul(temp_rho_2, temp_rho_2, temp_rho_1, MPFR_RNDD); // z*sqrt(pi*sigma^2)
    //         }

    //         mpfr_add(rho_sum, rho_sum, temp_rho_2, MPFR_RNDD);
    //     }
    // }
    // mpfr_set_d(alpha, (-1.0) * int(X.lambda_bar), MPFR_RNDD);
    // mpfr_exp(alpha, alpha, MPFR_RNDD);

    // mpfr_mul(rho_sum, rho_sum, alpha, MPFR_RNDD);

    // mpfr_set_d(output, log2(M_E) * int(X.lambda_bar), MPFR_RNDD);
    // mpfr_sub(output, output, rho_sum, MPFR_RNDD);

    // ret_val = mpfr_get_ld(output, MPFR_RNDD);

    return ret_val;
}

long double poisson_trivariate_H(
    vector<poisson_RV_v2> target,
    vector<poisson_RV_v2> shared_s,
    vector<poisson_RV_v2> unique_1,
    vector<poisson_RV_v2> unique_2) {
    // printf("trivar\n");

    long double a, c;
    a = shannon_entropy(target);
    // cout<<"shared_s.size() "<< shared_s.size()<<endl;

    sum_poisson_RVs_v2 S_1 = {shared_s, unique_1};
    sum_poisson_RVs_v2 S_2 = {shared_s, unique_2};
    joint_poisson_RV_v2 joint_rv = {S_1, S_2};
    // joint_poisson_RV_v2 joint_rv = {{shared_s, unique_1}, {shared_s, unique_2}};

    // printf("trivar_bivar\n");
    // cout<<"("<<joint_rv.lambda_X_1<<", "<<joint_rv.lambda_X_2<<", "<<joint_rv.lambda_X_12<<")\n";
    // // cout<<"("<<joint_rv.X_1.shared_RVs.size()<<", "<<joint_rv.X_2.shared_RVs.size()<<")\n";
    // cout<<"("<<joint_rv.num_shared<<")\n";

    // cout<<"("<<joint_rv.X_1.N<<", "<<joint_rv.X_2.N<<")\n";

    c = poisson_bivariate_H(joint_rv);
    // printf("end trivar_bivar\n");

    return a + c;
}

poisson_output_v2 single_calculation_v2(poisson_params_v2 &target_params, poisson_params_v2 &shared_s_params, poisson_params_v2 &unique_1_s_params, poisson_params_v2 &unique_2_s_params, uint num_shared, uint num_spec_1, uint num_spec_2) {

    vector<poisson_RV_v2> target;
    target.push_back({target_params.lambda, target_params.N});
    vector<poisson_RV_v2> shared_s;
    vector<poisson_RV_v2> unique_1;
    vector<poisson_RV_v2> unique_2;

    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_s_params.lambda, shared_s_params.N});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({unique_1_s_params.lambda, unique_1_s_params.N});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({unique_2_s_params.lambda, unique_2_s_params.N});
    }

    vector<poisson_RV_v2> X_1(target);
    X_1.insert(X_1.end(), shared_s.begin(), shared_s.end());

    long double trivariate_ent = 0, bivariate_ent = 0, target_init = 0, awae = 0, cond = 0;
    auto start = std::chrono::system_clock::now();

    // printf("------------ trivariate_ent --------\n");

    // joint_poisson_RV_v2 joint_rv = {
    //     {X_1, unique_1}, // X_1 = (X_T + X_S) + X_S_1
    //     {X_1, unique_2}  // X_2= (X_T + X_S) + X_S_2
    // };

    trivariate_ent = poisson_trivariate_H(target,
                                          shared_s,
                                          unique_1,
                                          unique_2);

    // joint_poisson_RV_v2 joint_rv = {
    //     {X_1, unique_1}, // X_1 = (X_T + X_S) + X_S_1
    //     {X_1, unique_2}  // X_2= (X_T + X_S) + X_S_2
    // };

    sum_poisson_RVs_v2 S_1 = {X_1, unique_1};
    sum_poisson_RVs_v2 S_2 = {X_1, unique_2};
    joint_poisson_RV_v2 joint_rv = {S_1, S_2};

    // cout<<"("<<joint_rv.lambda_X_1<<", "<<joint_rv.lambda_X_2<<", "<<joint_rv.lambda_X_12<<")\n";
    // cout<<"("<<joint_rv.X_1.shared_RVs.size()<<", "<<joint_rv.X_1.shared_RVs.size()<<")\n";

    // printf("------------ bivariate --------\n");

    bivariate_ent = poisson_bivariate_H(joint_rv);
    cond = trivariate_ent - bivariate_ent;
    // X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());

    target_init = shannon_entropy(target);

    awae = target_init + shannon_entropy({shared_s, unique_1}) - shannon_entropy({X_1, unique_1});

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    if (num_spec_1 == 0 && num_spec_2 == 0) {
        cond = awae;
    }

    printf("%-9u, %-9u, %-9u, %-.9Lf, %-.9Lf, %-.9Lf, %-.9Lf, %-.9Lf, %-.9f,\n",
           num_shared,
           num_spec_1,
           num_spec_2,
           target_init,
           awae,
           trivariate_ent,
           bivariate_ent,
           cond,
           elapsed_seconds.count());

    poisson_output_v2 output = {
        target_params,
        shared_s_params,
        unique_1_s_params,
        unique_2_s_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        target_init,
        awae,
        trivariate_ent,
        bivariate_ent,
        cond};

    return output;
}

void joint_poisson_main_v2() {

    // vector<int> N_vals{8, 16, 32, 64, 128};

    vector<uint> N_vals{8};
    uint lambda;
    int counter = 0;

    init_mpfr();

    uint upper_bound = 50;
    for (auto &N : N_vals) {
        lambda = N / 2;

        poisson_params_v2 target_params = {lambda, N};
        poisson_params_v2 shared_s_params = {lambda, N};
        poisson_params_v2 unique_1_params = {lambda, N};
        poisson_params_v2 unique_2_params = {lambda, N};

        vector<poisson_output_v2> output;

        // int ctr = 0;

        printf("%-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s,\n",
               "shared    ",
               "spec_1    ",
               "spec_2  ",
               "H_T         ",
               "awae        ",
               "trivar     ",
               "bivar      ",
               "cond       ",
               "time(s)    ");
        for (size_t i = 1; i < upper_bound; i++) {
            output.push_back(single_calculation_v2(target_params, shared_s_params, unique_1_params, unique_2_params, 0, i, i));
        }
        // writeData_joint_poisson(output, "no_common_spec_" + to_string(counter));
        counter++;
    }
    for (auto &N : N_vals) {
        lambda = N / 2;

        poisson_params_v2 target_params = {lambda, N};
        poisson_params_v2 shared_s_params = {lambda, N};
        poisson_params_v2 unique_1_params = {lambda, N};
        poisson_params_v2 unique_2_params = {lambda, N};

        vector<poisson_output_v2> output;

        printf("%-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s,\n",
               "shared    ",
               "spec_1    ",
               "spec_2  ",
               "H_T         ",
               "awae        ",
               "trivar     ",
               "bivar      ",
               "cond       ",
               "time(s)    ");
        for (size_t i = 1; i < upper_bound; i++) {
            output.push_back(single_calculation_v2(target_params, shared_s_params, unique_1_params, unique_2_params, 1, i, i));
        }
        // writeData_joint_poisson(output, "no_common_spec_" + to_string(counter));
        counter++;
    }

    vector<uint> num_specs{10, 50};
    for (auto &total_num_spec : num_specs) {

        counter = 0;
        for (auto &N : N_vals) {
            lambda = N / 2;

            poisson_params_v2 target_params = {lambda, N};
            poisson_params_v2 shared_s_params = {lambda, N};
            poisson_params_v2 unique_1_params = {lambda, N};
            poisson_params_v2 unique_2_params = {lambda, N};

            vector<poisson_output_v2> output;
        printf("%-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s, %-9s,\n",
               "shared    ",
               "spec_1    ",
               "spec_2  ",
               "H_T         ",
               "awae        ",
               "trivar     ",
               "bivar      ",
               "cond       ",
               "time(s)    ");
            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                // num_shared = 2 * i;
                num_shared = i;
                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                // output.push_back(single_calculation(target_params, shared_s_params, unique_1_params, unique_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
                output.push_back(single_calculation_v2(target_params,
                                                       shared_s_params,
                                                       unique_1_params,
                                                       unique_2_params,
                                                       num_shared,
                                                       num_ex1,
                                                       num_ex2));
            }
            // writeData_joint_norm(output, to_string(total_num_spec) + string("_") + "vary_spec_percent_same_target_" + to_string(counter));
            counter++;
        }
        counter = 0;
    }

    exit_mpfr();
}
