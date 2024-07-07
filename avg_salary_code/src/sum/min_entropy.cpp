#include "min_entropy.hpp"

long double min_entropy(uint x_A, uint numSpectators, double pois_lambda) {

    return 0;
}

void min_ent_driver_poisson(double lambda) {

    uint maxSpectators = 26;

    uint numTargets = 1;
    uint numAttackers = 1;

    uint numSigmas = 10; // cutoff point for "infinity"

    vector<double> awae_results;
    vector<double> target_init_entropy;
    vector<int> spectators;

    for (size_t numSpectators = 1; numSpectators < maxSpectators; numSpectators++) {

        uint attacker_domain = numAttackers * lambda * numSigmas;
        uint target_domain = numTargets * lambda * numSigmas;
        uint spectator_domain = numSpectators * lambda * numSigmas;

        uint output_domain = attacker_domain + target_domain + spectator_domain;

        long double min_ent;
        long double min_ent_target;
        long double max;
        long double temp;
        size_t x_A = 0;
        // for (size_t x_A = 0; x_A < 1; x_A++) {
        min_ent = 0.0;
        for (size_t o = 0; o < output_domain; o++) {
            max = 0.0;
            temp = 0.0;
            for (size_t x_T = 0; x_T < target_domain; x_T++) {
                temp = poisson_pmf(x_T, lambda, numTargets);
                temp *= poisson_pmf(o - x_A - x_T, lambda, numSpectators);
                if (temp > max) {
                    max = temp;
                }
            }
            min_ent += max;
        }

        for (size_t x_T = 0; x_T < target_domain; x_T++) {
            temp = poisson_pmf(x_T, lambda, numTargets);
            if (temp > min_ent_target) {
                min_ent_target = temp;
            }
        }

        cout << "[numSpec, H(X_T), H(X_T | O, X_A = x_a)] = " << numSpectators << ", " << (-1.0) * log2(min_ent_target) << ", " << (-1.0) * log2(min_ent) << endl;

        awae_results.push_back((-1.0) * log2(min_ent));
        spectators.push_back(numSpectators);
        target_init_entropy.push_back((-1.0) * log2(min_ent_target));

        // }
    }
    string experiment = "sum_poisson_min";
    writeData_Poisson(experiment, awae_results, target_init_entropy, spectators, int(lambda), 2 * int(lambda));
}

void min_ent_driver_uniform() {

    uint N = 16;


    uint numTargets = 1;
    uint numAttackers = 1;

    vector<double> awae_results;
    vector<double> target_init_entropy;
    vector<int> spectators;

    // for (size_t numSpectators = 1; numSpectators < maxSpectators; numSpectators++) {
    uint numSpectators = 4;

    uint attacker_domain = numAttackers * (N - 1);
    uint target_domain = numTargets * (N - 1);
    // uint spectator_domain = numSpectators * (N-1);

    uint output_domain = (numAttackers + numTargets + numSpectators) * (N - 1);
    // cout<<"D_O = "<<output_domain<<endl;
    long double min_ent;
    long double min_ent_target;
    long double max;
    long double temp;
    // size_t x_A = 0;
    for (size_t x_A = 0; x_A <= attacker_domain; x_A++) {
        min_ent = 0.0;
        for (size_t o = 0; o <= output_domain; o++) {
            max = 0.0;
            temp = 0.0;
            for (size_t x_T = 0; x_T <= target_domain; x_T++) {

                temp = uniform_pmf_mpfr(x_T, numTargets, N - 1);
                temp *= uniform_pmf_mpfr(o - x_A - x_T, numSpectators, N - 1);
                if (x_T == 0) {
                    max = temp;

                } else if (temp > max) {
                    max = temp;
                }
            }
            min_ent += max;
        }

        for (size_t x_T = 0; x_T <= target_domain; x_T++) {
            temp = uniform_pmf_mpfr(x_T, numTargets, N - 1);
            // cout << temp << endl;
            if (x_T == 0) {
                min_ent_target = temp;
            }
            if (temp > min_ent_target) {
                min_ent_target = temp;
            }
        }

        cout << "[numSpec, H(X_T), H(X_T | O, X_A = x_a)] = " << x_A << ", " << numSpectators << ", " << (-1.0) * log2(min_ent_target) << ", " << (-1.0) * log2(min_ent) << endl;

        // awae_results.push_back( (-1.0) * log2(min_ent) );
        // spectators.push_back(numSpectators);
        // target_init_entropy.push_back( (-1.0) * log2(min_ent_target) );
    }
    // }
    // writeData_Uniform( "sum_uniform_min", awae_results, target_init_entropy, spectators, N);
}
