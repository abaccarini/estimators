#include "testing.hpp"
#include "data_io.hpp"
#include "estimator.hpp"
#include "functions.hpp"
#include "knn_est.hpp"
#include "utilities.hpp"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdio.h>

using std::cout;
using std::endl;

double test_fn(double x, void *p) {
    test_obj<double, double> *params = (test_obj<double, double> *)p;
    // double x_t_placeholder = x;
    return params->evalute_fn(x) / sqrt(x);
}

double my_ln(double x) {
    return log(x);
}

void test_gsl() {
    /*
    thoguhts about getting this to work:
        would be better to add the function as a member of the knn_est class
            this way the object knows what function its evaluating
        add number of iterations
        when calling the estimate() method, be able to pass the variable x
            this is going to be used for x_T and/or x_a
                actually, what if x_a is a member of the class?
                such that its referenced automatuica
     */
    // typedef float d_type;
    // std::uniform_real_distribution<d_type> dist(0.0, 1.0);
    // knn_est<d_type, d_type, std::uniform_real_distribution> params(12345, dist);
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

    double result, error;
    double expected = -4.0;
    test_obj<double, double> params(10, my_ln);

    gsl_function F;
    F.function = &test_fn;
    F.params = &params;

    gsl_integration_qags(&F, 0, 1, 0, 1e-7, 1000,
                         w, &result, &error);

    printf("result          = % .18f\n", result);
    printf("exact result    = % .18f\n", expected);
    printf("estimated error = % .18f\n", error);
    printf("actual error    = % .18f\n", result - expected);
    printf("intervals       = %zu\n", w->size);

    gsl_integration_workspace_free(w);
}

void testing_main() {
    // compute_awae();
    // test_knn_est();
    // rng_testing_main();
    // test_est();
    test_gsl();
}

void compute_awae() {
    const long numSamples = 100000;
    const long N = 8;
    const int range_from = 0;
    const int range_to = N - 1;
    const size_t maxNumSpecs = 10;
    const size_t numTargets = 1;
    const size_t numAttackers = 1;
    const std::uint64_t seed = 12345;

    // plug_in_est<long> est(seed, range_from, range_to);

    // std::vector<std::vector<long double>> all_awae;
    // for (size_t numSpecs = 1; numSpecs < maxNumSpecs; numSpecs++) {
    //     std::vector<long double> awae_results;
    //     for (long j = range_from; j <= range_to; j++) {
    //         awae_results.push_back(est.estimate_leakage(static_cast<long (*)(std::map<long, long> &, const size_t &)>(compute_max<long>), numSamples, numTargets, numAttackers, numSpecs, {j}));
    //     }
    //     all_awae.push_back(awae_results);
    //     std::cout << numSpecs << " : " << awae_results << endl;
    //     // print(awae_results);
    // }

    std::uniform_int_distribution<int> dist(range_from, range_to);
    plug_in_est<int, int, std::uniform_int_distribution> est2(seed, dist);

    std::vector<std::vector<long double>> all_awae_new;
    for (size_t numSpecs = 1; numSpecs < maxNumSpecs; numSpecs++) {
        std::vector<long double> awae_results_2;
        for (int j = range_from; j <= range_to; j++) {
            awae_results_2.push_back(est2.estimate_leakage(
                static_cast<int (*)(std::map<int, int> &, const size_t &)>(compute_max<int>), numSamples, numTargets, numAttackers, numSpecs, {j}, range_from, range_to)); //
        }
        all_awae_new.push_back(awae_results_2);
        std::cout << numSpecs << " : " << awae_results_2 << endl;
        // print(awae_results);
    }
}

void test_est() {
    const long qty = 10;
    // const long N = 16;
    std::uniform_int_distribution<int> dist(0, 15);
    plug_in_est<int, int, std::uniform_int_distribution> est(12345, dist);

    std::map<int, int> map = est.generateSamples(qty);
    std::vector<int> x_A = {};
    // std::vector<int> x_A = {0};

    // ++map[x_A[0]]; // adding a known input to dataset
    // for (size_t i = 0; i < qty; i++) {
    //     // est.sample();
    //     ++map[est.sample()];
    // }
    for (const auto &[num, count] : map)
        std::cout << num << " generated " << std::setw(4) << count << " times\n";

    std::cout << "median:" << compute_median_min(map, qty + x_A.size()) << std::endl;
    std::cout << "max:" << compute_max(map, qty + x_A.size()) << std::endl;
    std::cout << "min:" << compute_min(map, qty + x_A.size()) << std::endl;
    std::cout << "mean:" << compute_mean<int, double>(map, qty + x_A.size()) << std::endl;
    std::cout << "stdev:" << compute_var<int, double>(map, qty + x_A.size()) << std::endl;
    //     std::vector<int> x_T = {5};
    //     std::vector<int> x_A_v = {};
    //     est.estimate_H_cond(compute_median_min<int>, 2, 4, x_T, x_A_v);
    //     est.estimate_H_cond(compute_median<int, double>, 2, 4, x_T, x_A_v);
    //     est.estimate_H_cond(compute_max<int>, 2, 4, x_T, x_A_v);
}

void test_knn_est() {
    const std::uint64_t seed = 12345;
    const unsigned long qty = 10;

    // NOTE the types used DO make a difference in speed (e.g. double v long double)
    // typedef float dist_type ;
    std::poisson_distribution<> pois(4);
    // std::cout << dist.max() << std::endl;
    std::cout << pois.max() << std::endl;
    std::cout << pois.min() << std::endl;
    // std::uniform_int_distribution<d_type> dist(0, 15);

    typedef float d_type;
    std::uniform_real_distribution<d_type> dist(0.0, 1.0);
    knn_est<d_type, d_type, std::uniform_real_distribution> est(seed, dist);

    // std::chrono::duration<double> elapsed_seconds;
    // auto start = std::chrono::system_clock::now();
    std::map<int, int> hist;
    std::vector<d_type> samples(qty);
    for (size_t i = 0; i < qty; i++) {
        ++hist[std::round(est.sample_input_data())];
        samples[i] = est.sample_input_data();
    }
    // cout<<samples<<endl;
    sort(samples.begin(), samples.end());
    // printVector(samples);
    cout << samples << endl;

    std::vector<d_type> knn = est.kNN(samples, 2);
    // printVector(knn);
    cout << samples << endl;
    cout << knn << endl;
    for (size_t i = 0; i < qty; i++) {
        cout << "(" << samples[i] << "->" << knn[i] << "), ";
    }
    std::cout << std::endl;
    // delete dist;

    // const std::lognormal_distribution<> dist2(1.6, 0.25);
    // knn_est<long double, std::uniform_real_distribution<>> est2(seed, dist);
    // for (size_t i = 0; i < qty; i++) {
    //     cout<<est.sample()<<endl;
    // }
    // auto end = std::chrono::system_clock::now();
    // elapsed_seconds = end - start;
    // cout << "(knn_rng)   elapsed time for " << qty << " : " << elapsed_seconds.count() << "s" << endl;
}
