#include "normal_v2.hpp"

void writeData_normal_v2(vector<output_str_shannon_vs_diff_struct> out_str, string experiment_name, string params, string distribution) {
    // string distribution;
    string extension = ".csv";
    // string distribution = "normal_v2";
    string dir_path = "../output/" + distribution + "/" + experiment_name + "/" + params;

    // making parent directory for all experiments to go into, if it doesnt exist
    // std::system(("mkdir -p ../output/" + experiment + "/" ));
    std::system(("mkdir -p " + dir_path).c_str());

    // // making directory for this specific experiment
    // // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = dir_path + "/results" + extension;
    // // string param_path = path + "/parameters" + extension;

    // // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << " target_params_mu, target_params_sigma, spec_params_mu, spec_params_sigma, num_targets, num_spec, H_T, H_S, H_T_S, delta_T, delta_S, delta_T_S, awae_shannon, h_T, h_S, h_T_S, awae_differential \n";

    for (auto &outpt : out_str) {
        outFile << outpt.target_params.mu
                << ", " << outpt.target_params.sigma
                << ", " << outpt.spec_params.mu
                << ", " << outpt.spec_params.sigma
                << ", " << outpt.num_targets
                << ", " << outpt.num_spec
                << ", " << outpt.H_T
                << ", " << outpt.H_S
                << ", " << outpt.H_T_S
                << ", " << outpt.delta_T
                << ", " << outpt.delta_S
                << ", " << outpt.delta_T_S
                << ", " << outpt.awae_shannon
                << ", " << outpt.h_T
                << ", " << outpt.h_S
                << ", " << outpt.h_T_S
                << ", " << outpt.awae_differential << "\n";
    }

    outFile.close();
    outFile.clear();
}

long double get_right_root_normal(gaussian_RV &X, double eps) {
    return X.mu + sqrt(-2.0 * X.sigma * log(eps * sqrt(2 * M_PI * X.sigma)));
}

long double get_right_root_normal(sum_gaussian_RVs_v2 &X, double eps) {
    return X.mu + sqrt(-2.0 * X.sigma * log(eps * sqrt(2 * M_PI * X.sigma)));
}

// double evaluate_normal_pmf(gaussian_RV &X, uint i, double delta, double left_bound) {
//     // needed to ensure we start "integrating" from the leftmost root
//     // left_bound not needed for lognormal, since support is defined for x>0
//     double a = left_bound + delta * i;
//     double b = left_bound + delta * (i + 1);

//     // trapezoidal rule
//     return (b - a) * 0.5 * (normal_pdf(X, a) + normal_pdf(X, b));
// }

double evaluate_normal_pmf(sum_gaussian_RVs_v2 &X, uint i, double delta, double left_bound, uint integ_type) {
    // needed to ensure we start "integrating" from the leftmost root
    // left_bound not needed for lognormal, since support is defined for x>0
    double a = left_bound + delta * i;
    double b = left_bound + delta * (i + 1);
    double result = 0.0;

    // trapezoidal rule

    // double result, error;
    // gsl_function F;
    // F.function = &normal_pdf;
    // F.params = &params;
    // gsl_integration_qags(&F, a, b, 0, 1e-7, w_size, w, &result, &error);
    // return result;

    switch (integ_type) {
    case 0:
        // cout<<"USING TRAP RULE"<<endl;
        // trapezoidal rule
        result = (b - a) * 0.5 * (normal_pdf(X, a) + normal_pdf(X, b));
        break;

    case 1:
        int w_size = 5000;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);
        normal_params params = {X.mu, X.sigma, 0.0};
        double error;

        gsl_function F;
        F.function = &normal_pdf;
        F.params = &params;
        gsl_integration_qags(&F, a, b, 0, 1e-7, w_size, w, &result, &error);
        gsl_integration_workspace_free(w);

        break;
    }

    return result;
}

double normal_pdf(gaussian_RV &X, double x) {
    return (1.0 / (sqrt((X.sigma) * 2 * M_PI))) * exp((-1) * (x - (X.mu)) * (x - (X.mu)) / (2 * (X.sigma)));
}

double normal_pdf(sum_gaussian_RVs_v2 &X, double x) {
    return (1.0 / (sqrt((X.sigma) * 2 * M_PI))) * exp((-1) * (x - (X.mu)) * (x - (X.mu)) / (2 * (X.sigma)));
}

output_str_shannon_vs_diff single_calculation_gaussian(normal_params &target_params, normal_params &spec_params, uint num_spec, uint num_targets, uint N, double eps, uint integ_type) {

    vector<gaussian_RV> target_RV;

    for (size_t i = 0; i < num_targets; i++) {
        target_RV.push_back({target_params.mu, target_params.sigma});
    }
    vector<gaussian_RV> spectators_RVs;
    for (size_t i = 0; i < num_spec; i++) {
        spectators_RVs.push_back({spec_params.mu, spec_params.sigma});
    }
    sum_gaussian_RVs_v2 targets = {target_RV, {}};
    sum_gaussian_RVs_v2 sum_spectators = {{}, spectators_RVs};
    sum_gaussian_RVs_v2 sum_target_spectators = {target_RV, spectators_RVs};

    // cout<<"S     :"<<num_spec<<", "<<sum_spectators.mu<<", "<<sum_spectators.sigma<<endl;
    // cout<<"T + S :"<<num_targets + num_spec<<", "<<sum_target_spectators.mu<<", "<<sum_target_spectators.sigma<<endl;

    double h_T = 0.0, h_S = 0.0, h_T_S = 0.0, awae_differential = 0.0; // shannon entropies
    // calculating differential entropies
    h_T = single_differential_entropy_v2(targets);
    h_S = single_differential_entropy_v2(sum_spectators);
    h_T_S = single_differential_entropy_v2(sum_target_spectators);

    awae_differential = h_T + h_S - h_T_S;

    // calculating Shannon entropies (approximate)
    double delta_T = 0.0, delta_S = 0.0, delta_T_S = 0.0, right_bound = 0.0;
    double H_T = 0.0, H_S = 0.0, H_T_S = 0.0, awae_shannon = 0.0; // shannon entropies

    right_bound = get_right_root_normal(targets, eps);
    delta_T = (2 * right_bound) / (num_targets * (N - 1) + 1);

    // cout<<"("<<right_bound<<", "<<delta_T<<")\n";

    for (int i = 0; i <= (N - 1) * num_targets; i++) {
        // H_T += gFunc(evaluate_normal_pmf(targets, i, delta_T, right_bound * (-1.0), integ_type));
    }
    H_T = (-1.0) * H_T;

    // right_bound = get_right_root_normal(sum_spectators, eps);
    delta_S = (2 * right_bound) / ((num_spec) * (N - 1) + 1);

    // cout<<"("<<right_bound<<", "<<delta_S<<")\n";

    for (int i = 0; i <= (N - 1) * num_spec; i++) {
        // H_S += gFunc(evaluate_normal_pmf(sum_spectators, i, delta_S, right_bound * (-1.0), integ_type));
    }
    H_S = (-1.0) * H_S;

    // right_bound = get_right_root_normal(sum_target_spectators, eps);
    delta_T_S = (2 * right_bound) / ((num_spec + num_targets) * (N - 1) + 1);

    // cout<<"("<<right_bound<<", "<<delta_T_S<<")\n";

    for (int i = 0; i <= (N - 1) * (num_spec + num_targets); i++) {
        // H_T_S += gFunc(evaluate_normal_pmf(sum_target_spectators, i, delta_T_S, right_bound * (-1.0), integ_type));
    }
    H_T_S = (-1.0) * H_T_S;

    awae_shannon = H_T + H_S - H_T_S;

    printf("%5u, %.7f, %.7f, %.7f, %.7f, %.7f, %.7f, %.7f, %.7f \n",
           num_spec, H_T, H_S, H_T_S, awae_shannon,
           h_T, h_S, h_T_S, awae_differential);

    output_str_shannon_vs_diff_struct output = {
        target_params,
        spec_params,
        num_targets,
        num_spec,
        H_T,
        H_S,
        H_T_S,
        delta_T,
        delta_S,
        delta_T_S,
        awae_shannon,
        h_T,
        h_S,
        h_T_S,
        awae_differential};
    return output;
}

void single_exp_gaussian_main() {
    string distribution = "normal_v2";
    string integ_key = "gsl";

    map<string, uint> integ_map{
        {"trap", 0},
        {"gsl", 1}};

    double mu = 0.0;

    vector<double> sigmas{0.25, 0.5, 1.0, 1.1, 1.2, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0};
    // vector<double> sigmas{2.0};

    vector<uint> N_vals{8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    // vector<uint> N_vals{8};
    int counter = 0;
    int upper_bound = 100;
    // int upper_bound = 5;
    vector<uint> num_specs{10, 50, 100};

    int exponent = 5;
    double eps = 1 * pow(10, -1.0 * exponent);

    for (auto &sigma : sigmas) {
        for (auto &N : N_vals) {

            normal_params target_params = {mu, sigma, eps};
            normal_params spec_params = {mu, sigma, eps};

            vector<output_str_shannon_vs_diff_struct> output;
            printf("%5s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s \n",
                   "nSpec", "H_T", "H_S", "H_T_S", "awae_S", "h_T", "h_S", "h_T_S", "awae_diff");
            for (size_t i = 1; i < upper_bound; i++) {
                output.push_back(single_calculation_gaussian(target_params, spec_params, i, 1, N, eps, integ_map.at(integ_key)));
            }
            writeData_normal_v2(output, "sh-v-diff_" + integ_key + string("_e-") + to_string(exponent), to_string(N) + string("_") + to_string(sigma), distribution);
        }
        counter++;
    }
    counter = 0;
}

void single_mixed_gaussian_main() {
    string distribution = "normal_mixed_full";
    // vector<double> sigmas{0.25, 0.5, 1.0};
    // vector<double> sigmas{4.0, 8.0, 16.0};
    double sigma_base = 4.0;

    vector<double> sigmas{sigma_base, sigma_base * (1.1 * 1.1), sigma_base * (0.9 * 0.9)};
    uint total_num_spec = 50;

    // loop is for target sigmas, determining which group he belongs in
    int ctr = 0;
    for (auto &sigma : sigmas) {
        int mod1, mod2, mod3;
        switch (ctr) {
        case 0:
            mod1 = 1;
            mod2 = 0;
            mod3 = 0;
            break;
        case 1:
            mod1 = 0;
            mod2 = 1;
            mod3 = 0;
            break;
        case 2:
            mod1 = 0;
            mod2 = 0;
            mod3 = 1;
            break;
        }
        vector<output_mixed> output;
        for (uint i = 0; i < total_num_spec; i++) {
            for (uint j = 0; j < total_num_spec; j++) {
                for (uint k = 0; k < total_num_spec; k++) {

                    normal_params target_params = {0, sigma};

                    vector<normal_params> spec_params = {{0, sigmas.at(0)}, {0, sigmas.at(1)}, {0, sigmas.at(2)}};

                    // vector<uint> num_specs = {i - mod1, i - mod2, i - mod3};
                    vector<uint> num_specs = {i ,j,k};
                    output.push_back(single_calculation_gaussian_mixed(target_params, spec_params, num_specs, 1));
                    // }
                    // }
                }
            }
        }
        writeData_normal_mixed(output, to_string(total_num_spec) + string("_") + to_string(sigma), distribution);
        ctr += 1;
    }
}

output_mixed single_calculation_gaussian_mixed(normal_params &target_params, vector<normal_params> &spec_params, vector<uint> num_spec, uint num_targets) {

    vector<gaussian_RV> target_RV;

    for (size_t i = 0; i < num_targets; i++) {
        target_RV.push_back({target_params.mu, target_params.sigma});
    }

    vector<gaussian_RV> spectators_RVs;
    assert(spec_params.size() == num_spec.size());
    for (size_t i = 0; i < spec_params.size(); i++) {
        for (size_t j = 0; j < num_spec.at(i); j++) {
            spectators_RVs.push_back({spec_params.at(i).mu, spec_params.at(i).sigma});
        }
    }

    sum_gaussian_RVs_v2 targets = {target_RV, {}};
    sum_gaussian_RVs_v2 sum_spectators = {{}, spectators_RVs};
    sum_gaussian_RVs_v2 sum_target_spectators = {target_RV, spectators_RVs};

    double h_T = 0.0, h_S = 0.0, h_T_S = 0.0, awae_differential = 0.0; // shannon entropies
    h_T = single_differential_entropy_v2(targets);
    h_S = single_differential_entropy_v2(sum_spectators);
    h_T_S = single_differential_entropy_v2(sum_target_spectators);

    awae_differential = h_T + h_S - h_T_S;

    output_mixed output = {
        target_params,
        spec_params,
        num_targets,
        num_spec,
        h_T,
        h_S,
        h_T_S,
        awae_differential};
    return output;
}

void writeData_normal_mixed(vector<output_mixed> out_str, string params, string distribution) {
    string extension = ".csv";
    string dir_path = "../output/" + distribution + "/" + params;
    // making parent directory for all experiments to go into, if it doesnt exist
    std::system(("mkdir -p " + dir_path).c_str());

    // // making directory for this specific experiment
    string path = dir_path + "/results" + extension;

    ofstream outFile(path);
    outFile << "target_params_mu, target_params_sigma, ";
    for (size_t i = 0; i < out_str.at(0).spec_params.size(); i++) {
        outFile << "spec_params_mu_" << to_string(i + 1) << ", spec_params_sigma_" << to_string(i + 1) << ", ";
    }

    outFile << "num_targets, ";
    for (size_t i = 0; i < out_str.at(0).spec_params.size(); i++) {
        outFile << "num_spec_" << to_string(i + 1) << ", ";
    }
    outFile << "h_T, h_S, h_T_S, awae_differential \n";

    for (auto &outpt : out_str) {
        outFile << outpt.target_params.mu
                << ", " << outpt.target_params.sigma
                << ", ";
        for (size_t i = 0; i < outpt.spec_params.size(); i++) {
            outFile << outpt.spec_params.at(i).mu << ", " << outpt.spec_params.at(i).sigma << ", ";
        }

        outFile << outpt.num_targets << ", ";
        for (size_t i = 0; i < outpt.num_spec.size(); i++) {
            outFile << outpt.num_spec.at(i) << ", ";
        }
        outFile << outpt.h_T
                << ", " << outpt.h_S
                << ", " << outpt.h_T_S
                << ", " << outpt.awae_differential << "\n";
    }

    outFile.close();
    outFile.clear();
}