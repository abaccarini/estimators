#include "gaussian_mixed.hpp"

// this is always done on a vector of size 3 (number of groups)
// workflow:
// genereate combination nC3 (n is total number of participants)
// if sum(combination) = n, then its a valid solution
// but we need to permute it for 6 total options

inline uint64_t factorial(uint64_t n) {
    uint64_t ret = 1;
    for (uint64_t i = 1; i <= n; ++i)
        ret *= i;
    return ret;
}

generator<comb_coeff> combination(uint64_t n) {
    for (uint64_t i = 0, ii = 1; i <= n; i++, ii *= (i)) {
        for (uint64_t j = 0, jj = 1; j <= n; j++, jj *= (j)) {
            for (uint64_t k = 0, kk = 1; k <= n; k++, kk *= (k)) {
                if ((i + j + k) == n) {
                    vector<uint64_t> v{i, j, k};
                    // vector<uint64_t> vv{ii, jj, kk};
                    long double c_fl = tgammal(n + 1) / (tgammal(i + 1) * tgammal(j + 1) * tgammal(k + 1));
                    comb_coeff coeff = {v, factorial(n) / (ii * jj * kk), c_fl};

                    // cout <<factorial(n)/(ii*jj*kk)<<", "<< c_fl<<endl;
                    co_yield coeff;
                }
            }
        }
    }
}

// val.second (sigma) is already squared
double gaussian_pdf(double x, void *p) {
    struct gaussian_params *params = (struct gaussian_params *)p;
    double accum = 0;
    double prob_term = pow(1.0 / 3.0, params->n); // 1/ |G|^n
    // this term is the same regardless of the participant group configuration
    // We assume that the likelihood of a participant belonging to group G is equally likely
    // e.g. if |B| = 5, |C|=1, |D| = 0.
    double sigma = 0;
    for (auto comb = combination(params->n); comb.move_next();) {
        // mu = params->mus.at(0) * comb.current_value().at(0) + params->mus.at(1) * comb.current_value().at(1) + params->mus.at(2) * comb.current_value().at(2);

        sigma = params->sigmas.at(0) * comb.current_value().v.at(0) + params->sigmas.at(1) * comb.current_value().v.at(1) + params->sigmas.at(2) * comb.current_value().v.at(2);

        accum += comb.current_value().coeff_fl * (1.0 / (sqrt(sigma * 2.0 * M_PI))) * exp((x * x) / (-2.0 * sigma));
    }
    return prob_term * accum;
}

// computes f(x) log f(X)
double log_gaussian_pdf(double x, void *p) {
    double tmp = gaussian_pdf(x, p);
    if (tmp < 0.000000000000000001) {
        return 0.0;
    } else {
        return (-1.0) * tmp * log2(tmp); // negative sign here bc entropy
    }
}

void single_mixed_gaussian_case_2() {

    uint64_t lower = 1, upper = 50; // the total number of spectators we are considering
    double sigma_base = 16.0;
    vector<double> sigmas{sigma_base, sigma_base * (1.1 * 1.1), sigma_base * (0.9 * 0.9)};
    // vector<double> sigmas{sigma_base, sigma_base, sigma_base};
    vector<double> mus{0.0, 0.0, 0.0};
    int w_size = 5000;

    double h_X_T, h_X_S, h_X_S_T;                              // will be i, i+1
    vector<double> h_X_T_out, h_X_S_out, h_X_S_T_out, abs_out; // will be i, i+1
                                                               // i --> number of spectators
    cout << "i \t h_X_T \t h_X_S_T \t h_X_S \t h_X_S_T - h_X_S \n";

    for (uint64_t i = lower; i < upper + 1; i++) {
        double error;
        if (i == lower) {
            gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);
            gaussian_params params = {sigmas, mus, i};
            gsl_function F;
            F.function = &log_gaussian_pdf;
            F.params = &params;
            gsl_integration_qagi(&F, 0, 1e-9, w_size, w, &h_X_S, &error);

            gaussian_params params_target = {sigmas, mus, 1};
            F.params = &params_target;
            gsl_integration_qagi(&F, 0, 1e-9, w_size, w, &h_X_T, &error);
            gsl_integration_workspace_free(w);
            // h_X_S = 0; // actually compute this term here
        }

        gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);
        gaussian_params params = {sigmas, mus, i+1};
        gsl_function F;
        F.function = &log_gaussian_pdf;
        F.params = &params;
        gsl_integration_qagi(&F, 0, 1e-7, w_size, w, &h_X_S_T, &error);
        gsl_integration_workspace_free(w);

        cout << i << " \t " << h_X_T << " \t " << h_X_S_T << " \t " << h_X_S << " \t " << h_X_S_T - h_X_S << " \n";
        h_X_T_out.push_back(h_X_T);
        h_X_S_out.push_back(h_X_S);
        h_X_S_T_out.push_back(h_X_S_T);
        abs_out.push_back(h_X_S_T - h_X_S);

        h_X_S = h_X_S_T; // setting for the next iteration
    }
    string distribution = "normal_mixed_case2";

    string extension = ".csv";
    string dir_path = "../output/" + distribution + "/" + to_string(upper) + string("_") + to_string(sigma_base);
    std::system(("mkdir -p " + dir_path).c_str());
    string path = dir_path + "/results" + extension;

    ofstream outFile(path);
    outFile << "num_spec, sigma_1, sigma_2, sigma_2, h_X_T, h_X_S_T, h_X_S, abs_loss\n";

    for (uint64_t i = lower; i < upper + 1; i++) {
        outFile << i
                << ", " << sigmas.at(0)
                << ", " << sigmas.at(1)
                << ", " << sigmas.at(2)
                << ", " << h_X_T_out.at(i - lower)
                << ", " << h_X_S_out.at(i - lower)
                << ", " << h_X_S_T_out.at(i - lower)
                << ", " << abs_out.at(i - lower) << "\n";
        ;
    }

    // outFile << outpt.target_params.mu
    //         << ", " << outpt.target_params.sigma
    //         << ", ";
    // for (size_t i = 0; i < outpt.spec_params.size(); i++) {
    //     outFile << outpt.spec_params.at(i).mu << ", " << outpt.spec_params.at(i).sigma << ", ";
    // }

    // outFile << outpt.num_targets << ", ";
    // for (size_t i = 0; i < outpt.num_spec.size(); i++) {
    //     outFile << outpt.num_spec.at(i) << ", ";
    // }
    // outFile << outpt.h_T
    //         << ", " << outpt.h_S
    //         << ", " << outpt.h_T_S
    //         << ", " << outpt.awae_differential << "\n";

    outFile.close();
    outFile.clear();
}

// std::vector<int> v(i + 1);
// std::iota(std::begin(v), std::end(v), 0);
// do {
//     for (auto var : v) {
//         cout << var << ", ";
//     }
// } while (next_permutation(v.begin(), v.end()));
//     cout << endl;

// for (auto word = permute(v); word.move_next();) {

// for (auto word = combination(i); word.move_next();) {
//     for (auto var : word.current_value()) {
//         cout << var << ", ";
//     }
//     cout << endl;
// }
// cout << endl;

// generator<vector<int>> permute(vector<int> &v) {
//     do {
//         co_yield v;
//     } while (next_permutation(v.begin(), v.end()));
// }
