#include "gamma.hpp"

long double single_differential_entropy_gamma(sum_gamma_RVs X) {

    // std::cout << boost::math::digamma(3.14) << "\n";

    return X.shape_k + log(X.scale_theta) + lgamma(X.shape_k) + (1 - X.shape_k) * (boost::math::digamma(X.shape_k));
}

output_str_gamma single_calculation_gamma(gamma_params &target_params, gamma_params &spec_params, uint num_spec, uint num_targets) {

    sum_gamma_RVs target = {target_params.shape_k, target_params.scale_theta, num_targets};
    sum_gamma_RVs sum_spectators = {spec_params.shape_k, spec_params.scale_theta, num_spec};
    sum_gamma_RVs sum_target_spectators = {spec_params.shape_k, spec_params.scale_theta, num_spec + num_targets};

    double h_T = 0.0, h_S = 0.0, h_T_S = 0.0, awae_differential = 0.0; //

    h_T = single_differential_entropy_gamma(target);
    h_S = single_differential_entropy_gamma(sum_spectators);
    h_T_S = single_differential_entropy_gamma(sum_target_spectators);
    awae_differential = h_T + h_S - h_T_S;

    cout << sum_spectators.shape_k << ", " << sum_spectators.scale_theta << ", " << sum_target_spectators.shape_k << ", " << sum_target_spectators.scale_theta << ", " << awae_differential << ", ";
    cout << h_T << ", " << h_S << ", " << h_T_S << ", " << awae_differential << endl;

    output_str_gamma output = {
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

void simple_driver_gamma() {

    string distribution = "gamma";
    double shape_k = 7.3854;;
    double scale_theta = 0.7710;

    uint upper_bound = 50;
    vector<output_str_gamma> output;
    gamma_params target_params = {shape_k, scale_theta};
    gamma_params spec_params = {shape_k, scale_theta};
    for (size_t i = 1; i < upper_bound; i++) {
        output.push_back(single_calculation_gamma(target_params, spec_params, i, 1));
    }
    writeData_gamma_est(output, "gamma_real", to_string(shape_k) + string("_") + to_string(scale_theta), distribution);

    // std::cout << boost::math::digamma(3.14) << "\n";
}

void writeData_gamma_est(vector<output_str_gamma> out_str, string experiment_name, string params, string distribution) {
    string extension = ".csv";
    string dir_path = "../output/" + distribution  ;

    // making parent directory for all experiments to go into, if it doesnt exist
    std::system(("mkdir -p " + dir_path).c_str());

    // // making directory for this specific experiment
    string path = dir_path + "/results" + extension;

    ofstream outFile(path);
    outFile << "target_params_shape_k, target_params_scale_theta, spec_params_shape_k, spec_params_scale_theta, num_targets, num_spec, h_T, h_S, h_T_S, awae_differential \n";

    for (auto &outpt : out_str) {
        outFile << outpt.target_params.shape_k
                << ", " << outpt.target_params.scale_theta
                << ", " << outpt.spec_params.shape_k
                << ", " << outpt.spec_params.scale_theta
                << ", " << outpt.num_targets
                << ", " << outpt.num_spec
                << ", " << outpt.h_T
                << ", " << outpt.h_S
                << ", " << outpt.h_T_S
                << ", " << outpt.awae_differential << "\n";
    }

    outFile.close();
    outFile.clear();
}