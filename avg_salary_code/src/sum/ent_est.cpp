#include "ent_est.hpp"

vector<double> generate_lognormal_samples(double mu, double sigma, uint num_samples, uint num_RVs) {

    const gsl_rng_type *T;
    gsl_rng *r;

    vector<double> x(num_samples);

    gsl_rng_env_setup();

    // T = gsl_rng_default;
    // r = gsl_rng_alloc(T);

    struct timeval tv; // Seed generation based on time
    gettimeofday(&tv, 0);
    unsigned long mySeed = tv.tv_sec + tv.tv_usec;
    T = gsl_rng_default; // Generator setup
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, mySeed);

    double sum;
    for (uint i = 0; i < num_samples; i++) {
        sum = 0.0;
        for (size_t j = 0; j < num_RVs; j++) {
            sum += gsl_ran_lognormal(r, mu, sqrt(sigma));
        }

        x.at(i) = sum;
    }

    gsl_rng_free(r);
    return x;
}

// samples.size() must be >= 2
vector<double> k_NN(vector<double> samples, uint k) {

    vector<double> output(samples.size());        // list of kth nearest distances for each sample
    vector<double> distances(samples.size() - 1); // will reuse this every iteration
    // cout << distances.size() << endl;
    for (size_t i = 0; i < samples.size(); i++) {
        // cout<<"hi"<<endl;
        uint ctr = 0;
        for (size_t j = 0; j < samples.size(); j++) {
            if (i != j) {
                distances.at(ctr) = abs(samples.at(i) - samples.at(j));
                ctr++;
            }
        }
        // cout<<"hi"<<endl;
        // list<double> k_NN_distances;
        vector<double> k_NN_distances;
        for (auto &d : distances) {
            if (k_NN_distances.size() < k) { // inserting the first k distances into k_NN_distances
                k_NN_distances.insert(lower_bound(k_NN_distances.begin(), k_NN_distances.end(), d), d);

            } else {
                if (k_NN_distances.back() > d) {
                    k_NN_distances.pop_back();                                                              // deleting the last element bc we know d will populate it
                    k_NN_distances.insert(lower_bound(k_NN_distances.begin(), k_NN_distances.end(), d), d); // adding d and sorting
                }
            }
        }
        output.at(i) = k_NN_distances.back(); // the last element in the list will be the kth N
    }

    return output;
}

void test_estimator() {

    double mu = 0.0;
    double sigma = 4.0;

    uint numRVs = 1;
    uint numSamples = 10000;
    uint numIterations = 100;
    uint k = 1;

    // lognormal_RV X_reference{mu, sigma};
    // double X_diff_ent = single_differential_entropy_lognormal(X_reference);

    // cout << "exact = " << X_diff_ent << " bits" << endl;
    // cout << "est   = " << calculate_estimator (mu, sigma, k, numSamples, numIterations, numRVs)/ log(2.0) << endl; 
}

double calculate_estimator(double mu, double sigma, uint k, uint numSamples, uint numIterations, uint numRVs) {

    uint d = 1, p = 2; // p = 2 for euclidean distance ( in 1d space, dont need the sqrt/pow2)
    double sum, c_d_p, digamma_val, result;

    vector<double> lognormal_samples(numSamples);
    vector<double> knn_distances(numSamples);
    result = 0.0;
    for (size_t i = 0; i < numIterations; i++) {

        lognormal_samples = generate_lognormal_samples(mu, sigma, numSamples, numRVs);
        knn_distances = k_NN(lognormal_samples, k);

        // lognormal_RV X_reference{mu, sigma};
        // double X_diff_ent = single_differential_entropy_lognormal(X_reference);

        c_d_p = (pow(tgamma(1.0 + 1.0 / double(p)), d) / (tgamma(1.0 + double(d) / double(p)))) * pow(2, d);
        // cout<<"c_d_p = "<<c_d_p<<endl;
        sum = 0.0;
        for (size_t j = 0; j < numSamples; j++) {
            sum += log(double(numSamples) * c_d_p * knn_distances.at(j) / double(k)); // this implies d = 1, otherwise we need to raise knn_distances.at(j)^d
        }
        digamma_val = gsl_sf_psi_int(k); // psi(1) = \gamma (euler-mascheroin)
        sum = (1.0 / numSamples) * sum + log(k) - digamma_val;
        result += sum;
    }

    return result / numIterations;
}





