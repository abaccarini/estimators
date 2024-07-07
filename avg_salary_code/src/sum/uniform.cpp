#include "uniform.hpp"

void simple_driver_uniform_test(int N) {
    string experiment = "sum_uniform";

    int numTargets = 1;
    int numSpectators = 0;
    long double H_T;
    long double H_S;
    long double H_T_S;
    long double awae;

    long double prior_value;

    vector<double> awae_results;
    vector<int> spectators;
    vector<double> target_init_entropy;

    int cutoff = 100;
    printf("%5s, %9s, %9s, %9s, %9s, %9s  ---- [N = %i]\n",
           "nSpec", "H_T", "H_S", "H_T_S", "awae", "time(s)", N);
    for (int i = 1; i < cutoff; i++) {

        numSpectators = i;
        // cout << numSpectators << endl;
        auto start = std::chrono::system_clock::now();

        H_T = 0;
        H_S = 0;
        H_T_S = 0;
        awae = 0;

        for (int i = 0; i <= (N - 1); i++) {
            // temp =  uniform_pmf(i, numTargets, N);
            H_T += gFunc(uniform_pmf_mpfr(i, numTargets, N - 1));
            // H_T += temp * log2(temp);
        }
        H_T = (-1.0) * H_T;

        // reduces computational cost by 1/2 so we do not perform the same calculation twice
        if (i > 1) {
            H_S = prior_value;
        } else {
            for (int i = 0; i <= ((N - 1) * numSpectators); i++) {
                // temp =  uniform_pmf_test(i, numSpectators, N);
                H_S += gFunc(uniform_pmf_mpfr(i, numSpectators, N - 1));
                // H_S += temp * log2(temp);
            }
            H_S = (-1.0) * H_S;
        }

        for (int i = 0; i <= (N - 1) * (numSpectators + numTargets); i++) {
            // temp =  uniform_pmf_test(i, numSpectators+numTargets, N);
            H_T_S += gFunc(uniform_pmf_mpfr(i, numSpectators + numTargets, N - 1));
            // H_T_S += temp * log2(temp);

            // cout << i << " - " << temp << ", " << H_T_S << endl;
        }
        H_T_S = (-1.0) * H_T_S;
        prior_value = H_T_S;
        // cout << "-------------" << endl;

        // std::cout << i << ": "
        //           << "H_S = " << H_S << "\t "
        //           << "H_T_S = " << H_T_S << "\t ";

        awae = H_T + H_S - H_T_S;
        // if (!isnan(awae)) {
        awae_results.push_back(awae);
        spectators.push_back(numSpectators);
        target_init_entropy.push_back(H_T);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        // cout << numSpectators << ", " << awae << "\t " << elapsed_seconds.count() << "s" << endl;

        printf("%5d, %.7Lf, %.7Lf, %.7Lf, %.7Lf, %.7f  \n",
               i, H_T, H_S, H_T_S, awae,
               elapsed_seconds.count());

        // cout << numSpectators << ", " << awae << endl;

        writeData_Uniform(experiment, awae_results, target_init_entropy, spectators, N);
        // }
    }
}

void simple_driver_uniform(int N) {
    string experiment = "sum_uniform";

    int numTargets = 1;
    int numSpectators = 0;
    long double H_T;
    long double H_S;
    long double H_T_S;
    long double awae;
    vector<double> awae_results;
    vector<int> spectators;
    vector<double> target_init_entropy;

    int cutoff = 40;
    for (int i = 1; i < cutoff; i++) {

        numSpectators = i;
        // cout << numSpectators << endl;
        auto start = std::chrono::system_clock::now();

        H_T = 0;
        H_S = 0;
        H_T_S = 0;
        awae = 0;

        for (int i = 0; i <= (N - 1); i++) {
            // temp =  uniform_pmf(i, numTargets, N);
            H_T += gFunc(uniform_pmf(i, numTargets, N - 1));
            // H_T += temp * log2(temp);
        }

        H_T = (-1.0) * H_T;
        // cout << "-------------" << endl;
        for (int i = 0; i <= ((N - 1) * numSpectators); i++) {
            // temp =  uniform_pmf(i, numSpectators, N);
            H_S += gFunc(uniform_pmf(i, numSpectators, N - 1));
            // H_S += temp * log2(temp);
        }
        H_S = (-1.0) * H_S;
        // cout << "-------------" << endl;

        for (int i = 0; i <= (N - 1) * (numSpectators + numTargets); i++) {
            // temp =  uniform_pmf(i, numSpectators+numTargets, N);
            H_T_S += gFunc(uniform_pmf(i, numSpectators + numTargets, N - 1));
            // H_T_S += temp * log2(temp);

            // cout << i << " - " << temp << ", " << H_T_S << endl;
        }
        H_T_S = (-1.0) * H_T_S;
        // cout << "-------------" << endl;

        // std::cout << i << ": "
        //           << "H_S = " << H_S << "\t "
        //           << "H_T_S = " << H_T_S << "\t ";

        awae = H_T + H_S - H_T_S;
        // if (!isnan(awae)) {
        awae_results.push_back(awae);
        spectators.push_back(numSpectators);
        target_init_entropy.push_back(H_T);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        // cout << numSpectators << ", " << awae << endl;

        printf("%5d, %.7Lf, %.7Lf, %.7Lf, %.7Lf, %.7f  \n",
               i, H_T, H_S, H_T_S, awae,
               elapsed_seconds.count());

        writeData_Uniform(experiment, awae_results, target_init_entropy, spectators, N);
        // }
    }
}
long double uniform_pmf(int y, int numRVs, int k) {
    long double temp = 0;
    // cout << "upper bound = " << floor(double(y) / double(k + 1)) << endl;
    for (int p = 0; p <= floor((long double)(y) / (long double)(k + 1)); p++) {
        // temp += pow(-1.0, p) * pow(M_E, lgamma(numRVs + y - p * (k + 1)) - lgamma(p + 1) - lgamma(numRVs - p + 1) - lgamma(y - p * (k + 1) + 1)) ;
        // cout << " ---------- " << p << " ---------- " << endl;
        // cout << " lgamma(" << (numRVs) + (y) - (p) * (k + 1) << ") \t " << lgamma((numRVs) + (y) - (p) * (k + 1)) << endl;
        // cout << " lgamma(" << (p) + 1 << ") \t " << lgamma((p) + 1) << endl;
        // cout << " lgamma(" << (numRVs) - (p) + 1 << ") \t " << lgamma((numRVs) - (p) + 1) << endl;
        // cout << " lgamma(" << (y) - (p) * (k + 1) + 1 << ") \t " << lgamma((y) - (p) * (k + 1) + 1) << endl;
        // cout << " exp (" << lgamma(double(numRVs) + double(y) - double(p) * double(k + 1.0)) - lgamma(double(p) + 1.0) - lgamma(double(numRVs) - double(p) + 1.0) - lgamma(double(y) - double(p) * double(k + 1.0) + 1.0) <<")"<<endl;

        temp += pow(-1.0, (p)) * pow(M_E, lgamma(double(numRVs) + double(y) - double(p) * double(k + 1.0)) - lgamma(double(p) + 1.0) - lgamma(double(numRVs) - double(p) + 1.0) - lgamma(double(y) - double(p) * double(k + 1.0) + 1.0));
        // temp += pow(-1.0, (p)) * (tgamma((numRVs) + (y) - (p) * (k + 1)) / (tgamma((p) + 1) * tgamma((numRVs) - (p) + 1) * tgamma((y) - (p) * (k + 1) + 1)));

        // cout << " temp \t\t " << temp << endl;
    }

    // printf("uniform_pmf(y = %i, numRVs = %i, k = %i) = %Lf\n", y, numRVs, k, (numRVs / pow(k + 1, numRVs)) * temp);

    return (double(numRVs) / pow(double(k) + 1.0, numRVs)) * temp;
}

long double uniform_pmf_test(int y, int numRVs, int k) {
    long double temp = 0;
    // cout << "upper bound = " << floor(double(y) / double(k + 1)) << endl;
    for (int p = 0; p <= floor((long double)(y) / (long double)(k + 1)); p++) {
        // temp += pow(-1.0, p) * pow(M_E, lgamma(numRVs + y - p * (k + 1)) - lgamma(p + 1) - lgamma(numRVs - p + 1) - lgamma(y - p * (k + 1) + 1)) ;
        cout << " ---------- " << p << " ---------- " << endl;
        cout << " lgamma(" << (numRVs) + (y) - (p) * (k + 1) << ") \t " << lgamma((numRVs) + (y) - (p) * (k + 1)) << endl;
        cout << " lgamma(" << (p) + 1 << ") \t " << lgamma((p) + 1) << endl;
        cout << " lgamma(" << (numRVs) - (p) + 1 << ") \t " << lgamma((numRVs) - (p) + 1) << endl;
        cout << " lgamma(" << (y) - (p) * (k + 1) + 1 << ") \t " << lgamma((y) - (p) * (k + 1) + 1) << endl;
        cout << " exp (" << lgamma(double(numRVs) + double(y) - double(p) * double(k + 1.0)) - lgamma(double(p) + 1.0) - lgamma(double(numRVs) - double(p) + 1.0) - lgamma(double(y) - double(p) * double(k + 1.0) + 1.0) << ")" << endl;

        temp += pow(-1.0, (p)) * pow(M_E,
                                     lgamma(double(numRVs) + double(y) - double(p) * double(k + 1.0)) - lgamma(double(p) + 1.0) - lgamma(double(numRVs) - double(p) + 1.0) - lgamma(double(y) - double(p) * double(k + 1.0) + 1.0));
        // temp += pow(-1.0, (p)) * (tgamma((numRVs) + (y) - (p) * (k + 1)) / (tgamma((p) + 1) * tgamma((numRVs) - (p) + 1) * tgamma((y) - (p) * (k + 1) + 1)));

        cout << " temp \t\t " << temp << endl;
    }

    printf("uniform_pmf(y = %i, numRVs = %i, k = %i) = %Lf\n", y, numRVs, k, (numRVs / pow(k + 1, numRVs)) * temp);

    return (double(numRVs) / pow(double(k) + 1.0, numRVs)) * (long double)temp;
}

long double uniform_pmf_mpfr(int y, int n, int k) {
    double ret_val;
    mpfr_t temp, alpha, beta, omega, delta;
    mpfr_init2(temp, PREC);
    mpfr_init2(alpha, PREC);
    mpfr_init2(beta, PREC);
    mpfr_init2(omega, PREC);
    mpfr_init2(delta, PREC);
    mpfr_set_zero(temp, 1);

    for (int p = 0; p <= floor((long double)(y) / (long double)(k + 1)); p++) {

        mpfr_set_zero(alpha, 1);
        mpfr_set_zero(beta, 1);
        mpfr_set_zero(omega, 1);
        mpfr_set_zero(delta, 1);

        mpfr_add_si(alpha, alpha, n + y - p * (k + 1), MPFR_RNDD);
        mpfr_add_si(beta, beta, p + 1, MPFR_RNDD);
        mpfr_add_si(omega, omega, n - p + 1, MPFR_RNDD);
        mpfr_add_si(delta, delta, y - p * (k + 1) + 1, MPFR_RNDD);

        mpfr_lngamma(alpha, alpha, MPFR_RNDD);
        mpfr_lngamma(beta, beta, MPFR_RNDD);
        mpfr_lngamma(omega, omega, MPFR_RNDD);
        mpfr_lngamma(delta, delta, MPFR_RNDD);

        mpfr_sub(alpha, alpha, beta, MPFR_RNDD);
        mpfr_sub(alpha, alpha, omega, MPFR_RNDD);
        mpfr_sub(alpha, alpha, delta, MPFR_RNDD);

        mpfr_exp(alpha, alpha, MPFR_RNDD);

        mpfr_mul_si(alpha, alpha, pow(-1.0, (p)), MPFR_RNDD);

        mpfr_add(temp, temp, alpha, MPFR_RNDD);
    }

    mpfr_mul_d(temp, temp, (double(n) / pow(k + 1, n)), MPFR_RNDD);
    ret_val = mpfr_get_ld(temp, MPFR_RNDD);

    mpfr_clear(temp);
    mpfr_clear(alpha);
    mpfr_clear(beta);
    mpfr_clear(omega);
    mpfr_clear(delta);
    mpfr_free_cache();
    return ret_val;
}
