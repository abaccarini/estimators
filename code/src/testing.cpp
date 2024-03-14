#include "testing.hpp"
#include "data_io.hpp"
#include "experiment.hpp"
#include "functions.hpp"
#include "knn_est.hpp"
#include "plug-in_est.hpp"
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

vector<int> return_vec() {
    vector<int> x = {1, 2, 3};
    return x;
}

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
                such that its referenced automatuically
     */
    // typedef float d_type;
    // std::uniform_real_distribution<d_type> dist(0.0, 1.0);
    // knn_est<d_type, d_type, std::uniform_real_distribution> params(12345, dist);
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

    double result, error;
    double expected = -4.0;
    test_obj<double, double> params(10, my_ln);
    params.set_x_A(15.0);
    std::cout << params.get_x_A() << std::endl;
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
    // batch_exp_discrete_uniform("max");
    // batch_exp_discrete_uniform("median");
    // batch_exp_discrete_uniform("var");
    // batch_exp_discrete_uniform("var_nd");
    batch_exp_poisson("median");
    batch_exp_poisson("max");
    batch_exp_poisson("var_nd");
    // test_knn_est();
    // rng_testing_main();
    // test_est();
    // test_gsl();
}

void compute_awae() {
    const long numIterations = 500; // how many times we repeat computation to eliminate noise/random deviations
    const long numSamples = 3000;   // number of items in estimator
    const long N = 8;
    const int range_from = 0;
    const int range_to = N - 1;
    const size_t maxNumSpecs = 10;
    const size_t numTargets = 1;
    const size_t numAttackers = 1;
    const std::uint64_t seed = 12345;

    typedef long in_type;
    typedef long out_type;

    std::uniform_int_distribution<in_type> dist(range_from, range_to);
    plug_in_est<in_type, out_type, std::uniform_int_distribution> est2(seed, dist);

    std::map<size_t, std::map<in_type, long double>> awae_data;
    for (size_t numSpecs = 1; numSpecs < maxNumSpecs; numSpecs++) {
        std::map<in_type, long double> awae_vals;
        for (in_type j = range_from; j <= range_to; j++) {
            awae_vals.insert({j, est2.estimate_leakage(static_cast<out_type (*)(std::map<in_type, size_t> &, const size_t &)>(compute_max<in_type>), numSamples, numTargets, numAttackers, numSpecs, {j}, range_from, range_to)});
        }
        awae_data.insert({numSpecs, awae_vals});
    }

    discrete_data<in_type, out_type, std::uniform_int_distribution> dd = {"max", dist, numSamples, numTargets, numAttackers, numIterations, est2.target_init_entropy, awae_data};

    writeJSON_discrete<in_type, out_type, std::uniform_int_distribution>(dd);
}

void test_est() {
    const long qty = 10;
    // const long N = 16;
    std::uniform_int_distribution<int> dist(0, 15);
    plug_in_est<int, int, std::uniform_int_distribution> est(12345, dist);

    std::map<int, size_t> map = est.generateSamples(qty);
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

    std::pair<double, double> var_mean_res(compute_var_mean<int, double>(map, qty + x_A.size()));
    std::cout << "stdev_mu:" << var_mean_res.first << ", " << var_mean_res.second << std::endl;
    //     std::vector<int> x_T = {5};
    //     std::vector<int> x_A_v = {};
    //     est.estimate_H_cond(compute_median_min<int>, 2, 4, x_T, x_A_v);
    //     est.estimate_H_cond(compute_median<int, double>, 2, 4, x_T, x_A_v);
    //     est.estimate_H_cond(compute_max<int>, 2, 4, x_T, x_A_v);
}

void test_knn_est() {
    const std::uint64_t seed = 12346;
    const size_t qty = 3000;

    // NOTE the types used DO make a difference in speed (e.g. double v long double)
    typedef long double d_type;

    const size_t numSpecs = 1;
    const size_t numTargets = 1;
    const uint k = 1;

    // std::uniform_real_distribution<d_type> dist(0.0, 1.0);
    std::normal_distribution<d_type> dist(0.0, 2.0);
    knn_est<d_type, d_type, std::normal_distribution> est(seed, dist, compute_sum<d_type>, qty, numSpecs, numTargets, k);
    vector<d_type> x_A = {0.5};
    est.set_x_A(x_A);
    // cout<<est.generateSamples(numSpecs + numTargets,x_A[0])<<endl;
    // cout<<est.evaluate_pdf(0.5)<<endl;
    size_t trials = 500;
    long double res = 0.0;
    long double error = 0.0;
    long double actual = 0.5 * log2(2.0 * M_PI * M_E * 2.0 * 2.0);
    for (size_t i = 0; i < trials; i++) {
        auto val = est.estimate_h();
        res += val;
        error += (val - actual) * (val - actual);
    }
    res /= static_cast<long double>(trials);
    error /= static_cast<long double>(trials);
    cout << "estimated = " << res << endl;
    cout << "actual    = " << actual << endl;
    cout << "(?) MSE   = " << error << endl;
    cout << datetime() << endl;

    // est.get_func_type();

    // std::map<int, int> hist;
    // std::vector<d_type> samples(qty);
    // for (size_t i = 0; i < qty; i++) {
    //     ++hist[std::round(est.sample_input_data())];
    //     samples[i] = est.sample_input_data();
    // }
    // // cout<<samples<<endl;
    // sort(samples.begin(), samples.end());
    // // printVector(samples);
    // cout << samples << endl;

    // std::vector<d_type> knn = est.kNN(samples, 2);
    // // printVector(knn);
    // cout << samples << endl;
    // cout << knn << endl;
    // for (size_t i = 0; i < qty; i++) {
    //     cout << "(" << samples[i] << "->" << knn[i] << "), ";
    // }
    // std::cout << std::endl;
}
