#include "poisson.hpp"

void simple_driver_poisson(int N) {
    string experiment = "sum_poisson";

    int numTargets = 1;
    int numSpectators = 0;
    double lambda = N / 2;
    long double H_T;
    long double H_S;
    long double H_T_S;
    long double awae;
    vector<double> awae_results;
    vector<double> target_init_entropy;
    vector<int> spectators;

    int cutoff = 100;
    for (int i = 1; i < cutoff; i++) {

        numSpectators = i;
        // cout << numSpectators << endl;
        auto start = std::chrono::system_clock::now();

        H_T = 0;
        H_S = 0;
        H_T_S = 0;
        awae = 0;
        
        for (int i = 0; i <= 10*N; i++) {
            // temp = pow(M_E, i * log(lambda * numTargets) - (lambda * numTargets) - lgamma(i + 1.0));
            H_T += gFunc(poisson_pmf(i, lambda, numTargets));
            // H_T += gFunc(pow(M_E, i * log(lambda * numTargets) - (lambda * numTargets) - lgamma(i + 1.0)));
            // H_T += temp * log2(temp);
        }
        H_T = (-1.0) * H_T;
        // std::cout << "H_T = " << H_T << "\t ";

        for (int i = 0; i <= 10*((N - 1) * numSpectators); i++) {
            // temp = pow(M_E, i * log(lambda * numSpectators) - (lambda * numSpectators) - lgamma(i + 1.0));
            H_S += gFunc(poisson_pmf(i, lambda, numSpectators));
            // H_S += gFunc(pow(M_E, i * log(lambda * numSpectators) - (lambda * numSpectators) - lgamma(i + 1.0)));
            // H_S += temp * log2(temp);
        }
        H_S = (-1.0) * H_S;
        // std::cout << "H_S = " << H_S << "\t ";

        for (int i = 0; i <= 10*(N - 1) * (numSpectators + numTargets); i++) {
            // temp = pow(M_E, i * log(lambda * (numSpectators + numTargets)) - (lambda * (numSpectators + numTargets)) - lgamma(i + 1.0));
            H_T_S += gFunc(poisson_pmf(i, lambda, numSpectators + numTargets));
            // H_T_S += gFunc(pow(M_E, i * log(lambda * (numSpectators + numTargets)) - (lambda * (numSpectators + numTargets)) - lgamma(i + 1.0)));
            // H_T_S += temp * log2(temp);

            // cout << i << " - " << temp << ", " << H_T_S << endl;
        }
        H_T_S = (-1.0) * H_T_S;
        // std::cout << "H_T_S = " << H_T_S << std::endl;

        awae = H_T + H_S - H_T_S;
        // if (!isnan(awae)) {
        awae_results.push_back(awae);
        spectators.push_back(numSpectators);
        target_init_entropy.push_back(H_T);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        cout << numSpectators << ", " << awae << "\t " << elapsed_seconds.count() << "s" << endl;
        // cout << numSpectators << ", " << awae << endl;

        writeData_Poisson(experiment, awae_results, target_init_entropy, spectators, int(lambda), N);
        // }
    }
}

long double poisson_pmf(int x, double lambda, double numRVs) {
    return pow(M_E, x * log(lambda * (numRVs)) - (lambda * (numRVs)) - lgamma(x + 1.0));
}

void joint_poisson_main() {

    // vector<int> N_vals{8, 16, 32, 64, 128};
    vector<int> N_vals{8};
    double lambda;
    int counter = 0;

    uint upper_bound = 50;
    for (auto &N : N_vals) {
        lambda = N / 2;

        poisson_params target_params = {lambda, N};
        poisson_params shared_s_params = {lambda, N};
        poisson_params unique_1_params = {lambda, N};
        poisson_params unique_2_params = {lambda, N};

        vector<poisson_output> output;

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
            output.push_back(single_calculation(target_params,
                                                shared_s_params,
                                                unique_1_params,
                                                unique_2_params,
                                                0,
                                                i,
                                                i));
        }
        // writeData_joint_poisson(output, "no_common_spec_" + to_string(counter));
        counter++;
    }
    counter = 0;

    for (auto &N : N_vals) {
        lambda = N / 2;

        poisson_params target_params = {lambda, N};
        poisson_params shared_s_params = {lambda, N};
        poisson_params unique_1_params = {lambda, N};
        poisson_params unique_2_params = {lambda, N};

        vector<poisson_output> output;
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
            output.push_back(single_calculation(target_params,
                                                shared_s_params,
                                                unique_1_params,
                                                unique_2_params,
                                                1,
                                                i,
                                                i));
        }
        // writeData_joint_poisson(output, "one_common_spec_" + to_string(counter));
        counter++;
    }
    counter = 0;

    ;
}

// sum of poisson RVs
long double poisson_pmf_mpfr(int x, sum_poisson_RVs X) {
    double ret_val;
    mpfr_t output, alpha;
    mpfr_init2(alpha, POISSON_PREC);
    mpfr_init2(output, POISSON_PREC);
    mpfr_set_zero(output, 1);

    mpfr_set_d(alpha, X.lambda_total, MPFR_RNDD);
    mpfr_log(alpha, alpha, MPFR_RNDD);
    mpfr_mul_si(alpha, alpha, x, MPFR_RNDD);
    mpfr_add(output, output, alpha, MPFR_RNDD);

    mpfr_sub_si(output, output, X.lambda_total, MPFR_RNDD);

    mpfr_set_si(alpha, x + 1, MPFR_RNDD);
    mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    mpfr_sub(output, output, alpha, MPFR_RNDD);

    mpfr_exp(output, output, MPFR_RNDD);
    ret_val = mpfr_get_ld(output, MPFR_RNDD);

    mpfr_clear(output);
    mpfr_clear(alpha);
    mpfr_free_cache();

    return ret_val;
}

// single poisson RV
long double poisson_pmf_mpfr(int x, poisson_RV X) {
    double ret_val;
    mpfr_t output, alpha;
    mpfr_init2(alpha, POISSON_PREC);
    mpfr_init2(output, POISSON_PREC);
    mpfr_set_zero(output, 1);

    mpfr_set_d(alpha, X.lambda, MPFR_RNDD);
    mpfr_log(alpha, alpha, MPFR_RNDD);
    mpfr_mul_si(alpha, alpha, x, MPFR_RNDD);
    mpfr_add(output, output, alpha, MPFR_RNDD);

    mpfr_sub_si(output, output, X.lambda, MPFR_RNDD);

    mpfr_set_si(alpha, x + 1, MPFR_RNDD);
    mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    mpfr_sub(output, output, alpha, MPFR_RNDD);

    mpfr_exp(output, output, MPFR_RNDD);
    ret_val = mpfr_get_ld(output, MPFR_RNDD);

    mpfr_clear(output);
    mpfr_clear(alpha);
    mpfr_free_cache();

    return ret_val;
}

long double poisson_pmf_mpfr(int x, vector<poisson_RV> &X) {
    double ret_val;
    mpfr_t output, alpha;
    mpfr_init2(alpha, POISSON_PREC);
    mpfr_init2(output, POISSON_PREC);
    mpfr_set_zero(output, 1);

    double lambda = 0;

    // summing up all lambdas in vector
    for (auto &n : X)
        lambda += n.lambda;

    mpfr_set_d(alpha, lambda, MPFR_RNDD);
    mpfr_log(alpha, alpha, MPFR_RNDD);
    mpfr_mul_si(alpha, alpha, x, MPFR_RNDD);
    mpfr_add(output, output, alpha, MPFR_RNDD);

    mpfr_sub_si(output, output, lambda, MPFR_RNDD);

    mpfr_set_si(alpha, x + 1, MPFR_RNDD);
    mpfr_lngamma(alpha, alpha, MPFR_RNDD);
    mpfr_sub(output, output, alpha, MPFR_RNDD);

    mpfr_exp(output, output, MPFR_RNDD);
    ret_val = mpfr_get_ld(output, MPFR_RNDD);

    mpfr_clear(output);
    mpfr_clear(alpha);
    mpfr_free_cache();

    return ret_val;
}

long double joint_poisson_pmf_mpfr(int x_1, int x_2, joint_poisson_RV X) {
    double ret_val;
    mpfr_t output, temp_inner, alpha;
    mpfr_init2(output, POISSON_PREC);
    mpfr_init2(temp_inner, POISSON_PREC);
    mpfr_init2(alpha, POISSON_PREC);
    mpfr_set_zero(output, 1);

    for (int i = 0; i <= std::min(x_1, x_2); i++) {

        mpfr_set_zero(temp_inner, 1);

        mpfr_set_d(alpha, X.lambda_vector(0), MPFR_RNDD);
        mpfr_log(alpha, alpha, MPFR_RNDD);
        mpfr_mul_si(alpha, alpha, x_1 - i, MPFR_RNDD);
        mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

        mpfr_set_d(alpha, X.lambda_vector(1), MPFR_RNDD);
        mpfr_log(alpha, alpha, MPFR_RNDD);
        mpfr_mul_si(alpha, alpha, x_2 - i, MPFR_RNDD);
        mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

        mpfr_set_d(alpha, X.lambda_vector(2), MPFR_RNDD);
        mpfr_log(alpha, alpha, MPFR_RNDD);
        mpfr_mul_si(alpha, alpha, i, MPFR_RNDD);
        mpfr_add(temp_inner, temp_inner, alpha, MPFR_RNDD);

        mpfr_set_si(alpha, x_1 - i + 1, MPFR_RNDD);
        mpfr_lngamma(alpha, alpha, MPFR_RNDD);
        mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

        mpfr_set_si(alpha, x_2 - i + 1, MPFR_RNDD);
        mpfr_lngamma(alpha, alpha, MPFR_RNDD);
        mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

        mpfr_set_si(alpha, i + 1, MPFR_RNDD);
        mpfr_lngamma(alpha, alpha, MPFR_RNDD);
        mpfr_sub(temp_inner, temp_inner, alpha, MPFR_RNDD);

        mpfr_exp(temp_inner, temp_inner, MPFR_RNDD);

        mpfr_add(output, output, temp_inner, MPFR_RNDD);
    }
    mpfr_set_d(alpha, (-1.0) * (X.lambda_vector(0) + X.lambda_vector(1) + X.lambda_vector(2)), MPFR_RNDD);
    mpfr_exp(alpha, alpha, MPFR_RNDD);

    mpfr_mul(output, output, alpha, MPFR_RNDD);
    ret_val = mpfr_get_ld(output, MPFR_RNDD);

    mpfr_clear(output);
    mpfr_clear(temp_inner);
    mpfr_clear(alpha);
    mpfr_free_cache();

    return ret_val;
}

long double bivariate_shannon_entropy(joint_poisson_RV X) {
    long double result = 0.0;

    // for (uint x_1 = 0; x_1 <= X.X_1.N; x_1++) {
    //     for (uint x_2 = 0; x_2 <= X.X_2.N; x_2++) {
    //         result += gFunc(joint_poisson_pmf_mpfr(x_1, x_2, X));
    //     }
    // }
    if (MPFR_FLAG) {
        // printf("USING MPFR\n");

        for (int x_1 = 0; x_1 <= X.X_1.N; x_1++) {
            for (int x_2 = 0; x_2 <= X.X_2.N; x_2++) {
                result += gFunc(joint_poisson_pmf_mpfr(x_1, x_2, X));
            }
        }
    } else {

        for (int x_1 = 0; x_1 <= X.X_1.N; x_1++) {
            for (int x_2 = 0; x_2 <= X.X_2.N; x_2++) {
                result += gFunc(joint_poisson_pmf(x_1, x_2, X));
            }
        }
    }

    // if (X.X_1.lambda_shared == 0 && X.X_2.lambda_shared == 0){

    //     for (uint x_1 = 0; x_1 <= X.X_1.N; x_1++) {
    //         for (uint x_2 = 0; x_2 <= X.X_2.N; x_2++) {
    //             result += gFunc(joint_poisson_pmf(x_1, x_2, X));
    //         }
    //     }
    // }



    return (-1.0) * result;
}

long double shannon_entropy(poisson_RV X) {
    long double result = 0.0;

    // for (int x = 0; x <= X.N; x++) {
    //     result += gFunc(poisson_pmf_mpfr(x, X));
    // }

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

// require that this is a vector of size 1
// used for the entropy of a single RV vector
// otherwise should be using sum type (next function)
long double shannon_entropy(vector<poisson_RV> &X) {
    assert(X.size() == 1);
    long double result = 0.0;
    // for (int x = 0; x <= X.at(0).N; x++) {
    //     result += gFunc(poisson_pmf_mpfr(x, X));
    // }
    if (MPFR_FLAG) {
        // printf("USING MPFR\n");

        for (int x = 0; x <= X.at(0).N; x++) {
            result += gFunc(poisson_pmf_mpfr(x, X));
        }
    } else {
        for (int x = 0; x <= X.at(0).N; x++) {
            result += gFunc(poisson_pmf(x, X));
        }
    }

    return (-1.0) * result;
}

long double shannon_entropy(sum_poisson_RVs X) {
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

long double trivariate_joint_shannon_entropy(
    vector<poisson_RV> &target,
    vector<poisson_RV> &shared_s,
    vector<poisson_RV> &unique_1,
    vector<poisson_RV> &unique_2) {

    // long double a, b, c, d;
    long double a, c;
    vector<poisson_RV> X_1(target);

    a = shannon_entropy(target);

    // no spectators across computation
    if (shared_s.size() == 0) {
        // std::cout << "hi" << std::endl;
        sum_poisson_RVs X_S_1 = {{}, unique_1};
        sum_poisson_RVs X_S_2 = {{}, unique_2};
        long double b = shannon_entropy(X_S_1);
        c = shannon_entropy(X_S_2);
        return a + b + c;
    } else {

        // X_T + X_S_1
        X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());

        // sum_poisson_RVs x_1_1 = {shared_s, unique_2};
        // sum_poisson_RVs x_1_2 = {shared_s, X_1};

        // cout << "x_1_1.lambda_shared = " << x_1_1.lambda_shared << endl;
        // cout << "x_1_2.lambda_shared = " << x_1_2.lambda_shared << endl;
        // cout << "x_1_1.lambda_unique = " << x_1_1.lambda_unique << endl;
        // cout << "x_1_2.lambda_unique = " << x_1_2.lambda_unique << endl;
        // cout << "x_1_1.lambda_total = " << x_1_1.lambda_total << endl;
        // cout << "x_1_2.lambda_total = " << x_1_2.lambda_total << endl;

        // cout << "x_1_1.N = " << x_1_1.N << endl;
        // cout << "x_1_2.N = " << x_1_2.N << endl;

        c = bivariate_shannon_entropy({
            {shared_s, unique_2}, // X_1 = X_S + X_S_2
            {shared_s, X_1}       // X_2 = X_S + X_T + X_S_1
        });

        // return a + b + c - d; // OLD
        return a + c;
    }
}

long double poisson_cond_entropy(vector<poisson_RV> &target, vector<poisson_RV> &shared_s, vector<poisson_RV> &unique_1, vector<poisson_RV> &unique_2) {
    // grouping X_T and X_S together
    vector<poisson_RV> X_1(target);
    X_1.insert(X_1.end(), shared_s.begin(), shared_s.end());

    return trivariate_joint_shannon_entropy(target,
                                            shared_s,
                                            unique_1,
                                            unique_2) -
           bivariate_shannon_entropy({
               {X_1, unique_1}, // X_1
               {X_1, unique_2}  // X_2
           });
}

poisson_output single_calculation(poisson_params &target_params, poisson_params &shared_s_params, poisson_params &unique_1_s_params, poisson_params &unique_2_s_params, uint num_shared, uint num_spec_1, uint num_spec_2) {

    vector<poisson_RV> target;
    target.push_back({target_params.lambda, target_params.N});

    vector<poisson_RV> shared_s;
    vector<poisson_RV> unique_1;
    vector<poisson_RV> unique_2;

    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_s_params.lambda, shared_s_params.N});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({unique_1_s_params.lambda, unique_1_s_params.N});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({unique_2_s_params.lambda, unique_2_s_params.N});
    }

    vector<poisson_RV> X_1(target);
    X_1.insert(X_1.end(), shared_s.begin(), shared_s.end());

    long double trivariate_ent, bivariate_ent, target_init, awae, cond;
    auto start = std::chrono::system_clock::now();

    trivariate_ent = trivariate_joint_shannon_entropy(target,
                                                      shared_s,
                                                      unique_1,
                                                      unique_2);
    bivariate_ent = bivariate_shannon_entropy({
        {X_1, unique_1}, // X_1 = (X_T + X_S) + X_S_1
        {X_1, unique_2}  // X_2= (X_T + X_S) + X_S_2
    });
    cond = trivariate_ent - bivariate_ent;
    // X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());

    target_init = shannon_entropy(target);

    awae = target_init + shannon_entropy({shared_s, unique_1}) - shannon_entropy({X_1, unique_1});
    // std::cout << "H_T_new   = " << shannon_entropy(target) << "\t ";
    // std::cout << "H_S_new   = " << shannon_entropy({shared_s, unique_1}) << "\t ";
    // std::cout << "H_T_S_new = " << shannon_entropy({X_1, unique_1}) << "\n";

    // sum_poisson_RVs srv_0 = {shared_s, unique_1};

    // cout << "srv_0.lambda_unique : " << srv_0.lambda_unique << endl;
    // cout << "srv_0.lambda_shared : " << srv_0.lambda_shared << endl;
    // cout << "srv_0.N : " << srv_0.N << endl;

    // sum_poisson_RVs srv = {X_1, unique_1};
    // cout << "srv.lambda_unique : " << srv.lambda_unique << endl;
    // cout << "srv.lambda_shared : " << srv.lambda_shared << endl;
    // cout << "srv.N : " << srv.N << endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    // if (num_spec_1 == 0 && num_spec_2 == 0) {
    //     cond = awae;
    // }

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

    poisson_output output = {
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

// sum of poisson RVs
long double poisson_pmf(int x, sum_poisson_RVs X) {
    return pow(M_E, x * log(X.lambda_total) - (X.lambda_total) - lgamma(x + 1.0));
}

// single poisson RV
long double poisson_pmf(int x, poisson_RV X) {
    return pow(M_E, x * log(X.lambda) - (X.lambda) - lgamma(x + 1.0));
}

long double poisson_pmf(int x, vector<poisson_RV> &X) {
    double lambda = 0;
    // summing up all lambdas in vector
    for (auto &n : X)
        lambda += n.lambda;
    return pow(M_E, x * log(lambda) - (lambda)-lgamma(x + 1.0));
}

long double joint_poisson_pmf(int x_1, int x_2, joint_poisson_RV X) {
    double ret_val = 0.0;
    for (int i = 0; i <= std::min(x_1, x_2); i++) {
        ret_val += pow(M_E,
                       ((x_1 - i) * log(X.lambda_vector(0)) + (x_2 - i) * log(X.lambda_vector(1)) + (i)*log(X.lambda_vector(2)) - lgamma(x_1 - i + 1) - lgamma(x_2 - i + 1) - lgamma(i + 1)));
    }
    return ret_val * pow(M_E, (-1.0) * (X.lambda_vector(0) + X.lambda_vector(1) + X.lambda_vector(2)));
}

void writeData_joint_poisson(vector<poisson_output> out_str, string experiment_name) {
    string distribution;
    string extension = ".csv";
    string experiment = "joint_sum_poisson";
    string dir_path = "../output/" + experiment + "/" + experiment_name;

    // making parent directory for all experiments to go into, if it doesnt exist
    // std::system(("mkdir -p ../output/" + experiment + "/" ));
    std::system(("mkdir -p " + dir_path).c_str());

    // // making directory for this specific experiment
    // // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = dir_path + "/results" + extension;
    // // string param_path = path + "/parameters" + extension;

    // // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "target.lambda, target.N, shared_params.lambda, shared_params.N, ex1_params.lambda, ex1_params.N, ex2_params.lambda, ex2_params.N, num_shared, num_spec_1, num_spec_2, trivariate_shannon_ent, bivariate_shannon_ent, shannon_ent_cond, target_init, awae\n";

    // for (int i = 0; i < (int)awae_results.size(); i++) {
    //     outFile << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << "," << h_T.at(i) << "," << h_S.at(i) << "," << h_T_S.at(i) << endl;
    // }
    for (auto &outpt : out_str) {
        outFile << outpt.target.lambda << "," << outpt.target.N << "," << outpt.shared_params.lambda << "," << outpt.shared_params.N << "," << outpt.ex1_params.lambda << "," << outpt.ex1_params.N << "," << outpt.ex2_params.lambda << "," << outpt.ex2_params.N << "," << outpt.num_shared << "," << outpt.num_spec_1 << "," << outpt.num_spec_2 << "," << outpt.trivariate_shannon_ent << "," << outpt.bivariate_shannon_ent << "," << outpt.shannon_ent_cond << "," << outpt.target_init << "," << outpt.awae << "\n";
    }

    outFile.close();
    outFile.clear();
}