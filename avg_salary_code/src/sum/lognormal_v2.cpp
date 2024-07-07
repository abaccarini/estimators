#include "lognormal_v2.hpp"

long double get_right_root_lognormal(lognormal_RV &X, double eps) {
    return exp(-(1.0) * (X.sigma - X.mu) + sqrt(X.sigma * X.sigma - 2 * X.sigma * X.mu - 2 * X.sigma * log(eps * sqrt(2 * M_PI * X.sigma))));
}

long double get_right_root_lognormal(sum_lognormal_RVs &X, double eps) {
    return exp(-(1.0) * (X.sigma - X.mu) + sqrt(X.sigma * X.sigma - 2 * X.sigma * X.mu - 2 * X.sigma * log(eps * sqrt(2 * M_PI * X.sigma))));
}

long double single_differential_entropy_lognormal(sum_lognormal_RVs X) {
    return (X.mu + 0.5 * log(2 * M_PI * M_E * X.sigma)) / log(2.0);
}

long double single_differential_entropy_lognormal(lognormal_RV X) {
    return (X.mu + 0.5 * log(2 * M_PI * M_E * X.sigma)) / log(2.0);
}

long double normal_pdf(lognormal_RV &X, double x) {
    return (1.0 / (sqrt((X.sigma) * 2 * M_PI))) * exp((-1) * (x - (X.mu)) * (x - (X.mu)) / (2 * (X.sigma)));
}

long double ln_pdf(sum_lognormal_RVs &X, double x) {
    return (1.0 / (sqrt((X.sigma) * 2 * M_PI))) * exp((-1) * (x - (X.mu)) * (x - (X.mu)) / (2 * (X.sigma)));
}

double ln_pdf_v2(double x, void *p) {
    normal_params *params = (normal_params *)p;
    return (1 / (x * sqrt((params->sigma) * 2 * M_PI))) * exp((-1) * (log(x) - (params->mu)) * (log(x) - (params->mu)) / (2 * (params->sigma)));
}

double evaluate_lognormal_pmf(sum_lognormal_RVs &X, uint i, double delta, double left_bound, uint integ_type) {
    // needed to ensure we start "integrating" from the leftmost root
    // left_bound not needed for lognormal, since support is defined for x>0
    double a = delta * i;
    double b = delta * (i + 1);
    double result = 0.0;

    switch (integ_type) {
    case 0:
        // cout<<"USING TRAP RULE"<<endl;
        // trapezoidal rule
        result = (b - a) * 0.5 * (ln_pdf(X, a) + ln_pdf(X, b));
        break;

    case 1:

        int w_size = 5000;
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);
        normal_params params = {X.mu, X.sigma, 0.0};
        double error;

        gsl_function F;
        F.function = &ln_pdf_v2;
        F.params = &params;
        gsl_integration_qags(&F, a, b, 0, 1e-7, w_size, w, &result, &error);
        gsl_integration_workspace_free(w);

        break;
        // default:
        //     printf("INVALID INTEGRATION FLAG\n");
        //     break;
    }

    return result;
}

output_str_shannon_vs_diff single_calculation_lognormal(normal_params &target_params, normal_params &spec_params, uint num_spec, uint num_targets, uint N, double eps, uint integ_type) {

    sum_lognormal_RVs target = {target_params.mu, target_params.sigma, num_targets};
    sum_lognormal_RVs sum_spectators = {spec_params.mu, spec_params.sigma, num_spec};
    sum_lognormal_RVs sum_target_spectators = {spec_params.mu, spec_params.sigma, num_spec + num_targets};

    // cout << "T     :" << num_targets << ", " << target_params.mu << ", " << target_params.sigma << endl;
    // cout << "S     :" << num_spec << ", " << sum_spectators.mu << ", " << sum_spectators.sigma << endl;
    // cout << "T + S :" << num_targets + num_spec << ", " << sum_target_spectators.mu << ", " << sum_target_spectators.sigma << endl;

    double h_T = 0.0, h_S = 0.0, h_T_S = 0.0, awae_differential = 0.0; // shannon entropies
    // calculating differential entropies
    h_T = single_differential_entropy_lognormal(target);
    h_S = single_differential_entropy_lognormal(sum_spectators);
    h_T_S = single_differential_entropy_lognormal(sum_target_spectators);

    uint numSamples = 4000;
    uint numIterations = 100;
    uint k = 1;

    // h_T = single_differential_entropy_lognormal(target); // exactly calculated
    // h_S = calculate_estimator(spec_params.mu, spec_params.sigma, k, numSamples, numIterations, num_spec);
    // h_T_S = calculate_estimator(spec_params.mu, spec_params.sigma, k, numSamples, numIterations, num_spec + 1);

    awae_differential = h_T + h_S - h_T_S;

    // calculating Shannon entropies (approximate)
    double delta_T = 0.0, delta_S = 0.0, delta_T_S = 0.0, right_bound = 0.0;
    double H_T = 0.0, H_S = 0.0, H_T_S = 0.0, awae_shannon = 0.0; // shannon entropies

    // right_bound = get_right_root_lognormal(target, eps);
    delta_T = (right_bound) / (num_targets * (N - 1) + 1);

    // cout << "(" << right_bound << ", " << delta_T << ") = " << ln_pdf(right_bound, &target_params) << endl;

    // for (uint i = 0; i <= (N - 1) * num_targets; i++) {
    //     H_T += gFunc(evaluate_lognormal_pmf(target, i, delta_T, right_bound * (-1.0), integ_type));
    // }
    H_T = (-1.0) * H_T;

    // right_bound = get_right_root_lognormal(sum_spectators, eps);
    delta_S = right_bound / (num_spec * (N - 1) + 1);

    // cout<<"("<<right_bound<<", "<<delta_S<<")\n";

    // for (uint i = 0; i <= (N - 1) * num_spec; i++) {
    //     H_S += gFunc(evaluate_lognormal_pmf(sum_spectators, i, delta_S, right_bound * (-1.0), integ_type));
    // }
    H_S = (-1.0) * H_S;

    // right_bound = get_right_root_lognormal(sum_target_spectators, eps);
    delta_T_S = right_bound / ((num_spec + num_targets) * (N - 1) + 1);

    // cout<<"("<<right_bound<<", "<<delta_T_S<<")\n";

    // for (uint i = 0; i <= (N - 1) * (num_spec + num_targets); i++) {
    //     H_T_S += gFunc(evaluate_lognormal_pmf(sum_target_spectators, i, delta_T_S, right_bound * (-1.0), integ_type));
    // }
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

output_diff_est single_calculation_lognormal_est(normal_params &target_params, normal_params &spec_params, uint num_spec, uint num_targets, uint numSamples, uint numIterations, uint k, vector<output_diff_est> &output) {

    sum_lognormal_RVs target = {target_params.mu, target_params.sigma, num_targets};
    sum_lognormal_RVs sum_spectators = {spec_params.mu, spec_params.sigma, num_spec};
    sum_lognormal_RVs sum_target_spectators = {spec_params.mu, spec_params.sigma, num_spec + num_targets};

    double h_T = 0.0, h_S = 0.0, h_T_S = 0.0, awae_differential = 0.0; // shannon entropies
    // calculating differential entropies
    // h_T = single_differential_entropy_lognormal(target);
    // h_S = single_differential_entropy_lognormal(sum_spectators);
    // h_T_S = single_differential_entropy_lognormal(sum_target_spectators);

    h_T = single_differential_entropy_lognormal(target); // exactly calculated
    if (output.size() == 0) {

        h_S = calculate_estimator(spec_params.mu, spec_params.sigma, k, numSamples, numIterations, num_spec);
    } else {
        h_S = output.back().h_T_S;
    }
    h_T_S = calculate_estimator(spec_params.mu, spec_params.sigma, k, numSamples, numIterations, num_spec + 1);

    awae_differential = h_T + h_S - h_T_S;

    printf("%5u, %.7f, %.7f, %.7f, %.7f \n",
           num_spec, h_T, h_S, h_T_S, awae_differential);

    output_diff_est new_output = {
        target_params,
        spec_params,
        num_targets,
        num_spec,
        numSamples,
        numIterations,
        k,
        h_T,
        h_S,
        h_T_S,
        awae_differential};
    return new_output;
}

void single_exp_lognormal_main() {
    string distribution = "lognormal_v2";

    // string integ_key = "trap";
    string integ_key = "gsl";

    map<string, uint> integ_map{
        {"trap", 0},
        {"gsl", 1}};

    double mu = 0.0;
    // vector<double> sigmas{0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0};
    // vector<double> sigmas{0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
    // vector<double> sigmas{2.0};
    // vector<double> sigmas{0.25, 0.5, 1.0};
    // vector<double> sigmas{0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
    vector<double> sigmas{ 2.0, 4.0, 8.0, 16.0};

    vector<uint> N_vals{8, 16, 32, 64, 128, 256, 512, 1024};
    // vector<uint> N_vals{8};
    int counter = 0;
    uint upper_bound = 50;
    // int upper_bound = 3;
    // vector<uint> num_specs{10, 50, 100};
    vector<uint> num_specs{10};

    uint numSamples = 4000;
    uint numIterations = 100;
    uint k = 1;

    int exponent = 5;
    double eps = 1 * pow(10, -1.0 * exponent);

    // for (auto &sigma : sigmas) {
    //     for (auto &N : N_vals) {

    //         normal_params target_params = {mu, sigma, eps};
    //         normal_params spec_params = {mu, sigma, eps};

    //         vector<output_str_shannon_vs_diff_struct> output;
    //         printf("%5s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s \n",
    //                "nSpec", "H_T", "H_S", "H_T_S", "awae_S", "h_T", "h_S", "h_T_S", "awae_diff");
    //         for (size_t i = 1; i < upper_bound; i++) {
    //             output.push_back(single_calculation_lognormal(target_params, spec_params, i, 1, N, eps, integ_map.at(integ_key)));
    //         }
    //         writeData_normal_v2(output, "sh-v-diff_" + integ_key + string("_e-") + to_string(exponent), to_string(N) + string("_") + to_string(sigma), distribution);
    //     }
    //     counter++;
    // }
    // counter = 0;

    // counter = 1;
    // for (auto &sigma : sigmas) {
    //     normal_params target_params = {mu, sigma, eps};
    //     normal_params spec_params = {mu, sigma, eps};

    //     vector<output_diff_est> output;
    //     printf("%5s, %9s, %9s, %9s, %9s \n",
    //            "nSpec", "h_T", "h_S", "h_T_S", "awae_diff");
    //     for (size_t i = 1; i < upper_bound; i++) {
    //         output.push_back(single_calculation_lognormal_est(target_params, spec_params, i, 1, numSamples*(2*counter), numIterations, k, output));
    //     }
    //     writeData_lognormal_est(output, "diff_est", to_string(mu) + "_" + to_string(sigma), distribution);
    //     counter++;
    // }
// 0.3815
    normal_params target_params = {1.6702, 0.3815*0.3815, eps};
    normal_params spec_params = {1.6702, 0.3815*0.3815, eps};

    uint N = 8;
    vector<output_diff_est> output;
    numSamples = 4000;
    printf("%5s, %9s, %9s, %9s, %9s \n",
            "nSpec", "h_T", "h_S", "h_T_S", "awae_diff");
    // for (size_t i = 1; i < upper_bound; i++) {
    //     output.push_back(single_calculation_lognormal_est(target_params, spec_params, i, 1, numSamples, numIterations, k, output));
    // }
    // writeData_lognormal_est(output, "diff_est_real", to_string(1.6702) + "_" + to_string(0.3815*0.3815), distribution);
    
    vector<output_str_shannon_vs_diff_struct> output2;
    for (size_t i = 1; i < upper_bound; i++) {
        output2.push_back(single_calculation_lognormal(target_params, spec_params, i, 1, N, eps, integ_map.at(integ_key)));

    }
    writeData_normal_v2(output2, "sh-v-diff-real_" + integ_key + string("_e-") + to_string(exponent), to_string(N) + string("_") + to_string(0.3815*0.3815), distribution);

    // writeData_lognormal_est(output, "diff_est_real", to_string(1.6702) + "_" + to_string(0.3815*0.3815), distribution);
    
}

void writeData_lognormal_est(vector<output_diff_est> out_str, string experiment_name, string params, string distribution) {
    string extension = ".csv";
    string dir_path = "../output/" + distribution + "/" + experiment_name + "/" + params;

    // making parent directory for all experiments to go into, if it doesnt exist
    std::system(("mkdir -p " + dir_path).c_str());

    // // making directory for this specific experiment
    string path = dir_path + "/results" + extension;

    ofstream outFile(path);
    outFile << "target_params_mu, target_params_sigma, spec_params_mu, spec_params_sigma, num_targets, num_spec, numSamples, numIterations, k, h_T, h_S, h_T_S, awae_differential \n";

    for (auto &outpt : out_str) {
        outFile << outpt.target_params.mu
                << ", " << outpt.target_params.sigma
                << ", " << outpt.spec_params.mu
                << ", " << outpt.spec_params.sigma
                << ", " << outpt.num_targets
                << ", " << outpt.num_spec
                << ", " << outpt.numSamples
                << ", " << outpt.numIterations
                << ", " << outpt.k
                << ", " << outpt.h_T
                << ", " << outpt.h_S
                << ", " << outpt.h_T_S
                << ", " << outpt.awae_differential << "\n";
    }

    outFile.close();
    outFile.clear();
}