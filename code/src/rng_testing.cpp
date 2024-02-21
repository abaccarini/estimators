#include "rng_testing.hpp"
#include "XoshiroCpp.hpp"
#include "pcg_random.hpp"
#include <gsl/gsl_rng.h>
#include <iostream>
#include <random>
#include <stdio.h>
#include <chrono>
#include <map>

void rng_testing_main() {


    const unsigned long qty = 1e9;
    const unsigned long Nm1 = 15;
    std::chrono::duration<double> elapsed_seconds;
    auto start = std::chrono::system_clock::now();
    // test_gsl(qty, Nm1);
    auto end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "(test_gsl) elapsed time for " << qty << " : " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    test_xos(qty, Nm1);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "(test_xos) elapsed time for " << qty << " : " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    test_pcg(qty, Nm1);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "(test_pcg) elapsed time for " << qty << " : " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    test_random(qty, Nm1);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "(test_rnd) elapsed time for " << qty << " : " << elapsed_seconds.count() << "s\n";
}

void test_gsl(const unsigned long qty, const unsigned long Nm1) {
    gsl_rng *r; /* global generator */
    const gsl_rng_type *T;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    for (size_t i = 0; i < qty; i++) {
        gsl_rng_uniform_int(r, Nm1 + 1);
    }
    gsl_rng_free(r);
}

// fastest by far
void test_xos(const unsigned long qty, const unsigned long Nm1) {
    const std::uint64_t seed = 12345;
    XoshiroCpp::Xoshiro256PlusPlus rng_xos(seed);
    std::uniform_int_distribution<int> dist(0, Nm1);
    std::map<int, int> hist;

    for (int i = 0; i < qty; ++i) {
        ++hist[(dist(rng_xos))];
        // std::cout<<dist(rng_xos)<<" ";
        // dist(rng_xos);

    }
}
// 2nd fastest
void test_pcg(const unsigned long qty, const unsigned long Nm1) {
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg32 rng(seed_source);
    std::uniform_int_distribution<int> distr(0, Nm1);
    // pcg32 rng_checkpoint = rng;
    std::map<int, int> hist;
    for (size_t i = 0; i < qty; i++) {
        ++hist[(distr(rng))];

        // distr(rng);
    }
}
void test_random(const unsigned long qty, const unsigned long Nm1) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_int_distribution<int> distr(0, Nm1);
    std::map<int, int> hist;
    for (size_t i = 0; i < qty; i++) {
        ++hist[(distr(generator))];
        // distr(generator);
    }
}
