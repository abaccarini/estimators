#include "joint.hpp"

// entropy returned is in NATS (not bits)
double single_differential_entropy(gaussian_RV X) {
    return log(sqrt(X.sigma * 2.0 * M_PI * M_E)) / log(2.0);
}

double single_differential_entropy_spec(sum_gaussian_RVs X) {
    return log(sqrt((X.sigma_shared + X.sigma_unique) * 2.0 * M_PI * M_E)) / log(2.0);
}

double single_differential_entropy_spec_target(sum_gaussian_RVs X) {
    // uses X.sigma which is the sum of all sigmas (X_T, X_S, X_S_1)
    return log(sqrt((X.sigma) * 2.0 * M_PI * M_E)) / log(2.0);
}

// entropy returned is in NATS (not bits)
double joint_differential_entropy(joint_exps_gaussian_RVs X_vec) {

    // used to avoid using std::pow()
    switch (X_vec.mean_vector.size()) {
    case 2:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X_vec.cov_matrix.determinant()) / log(2.0);
    case 3:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X_vec.cov_matrix.determinant()) / log(2.0);
    default:
        return 0.0;
    }
}

// H(X_T | X_1, X_2) = H(X_T, X_1, X_2) - H(X_1, X_2)
double cond_entropy(joint_exps_gaussian_RVs X_T_exps, joint_exps_gaussian_RVs X_exps) {

    return joint_differential_entropy(X_T_exps) - joint_differential_entropy(X_exps);
}

output_str single_calculation(normal_params &target_params, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2) {
    gaussian_RV target = {target_params.mu, target_params.sigma};

    vector<gaussian_RV> shared; // still needs to be declared and passed even if there are no shared spectators
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;

    for (size_t i = 0; i < num_shared; i++) {
        shared.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }

    joint_exps_gaussian_RVs joint_T_exps(target, {target, shared, unique_1}, {target, shared, unique_2});
    joint_exps_gaussian_RVs joint_exps({target, shared, unique_1}, {target, shared, unique_2});

    double diff_ent_joint_T_exps = joint_differential_entropy(joint_T_exps);
    double diff_ent_joint_exps = joint_differential_entropy(joint_exps);




    double target_init = single_differential_entropy(target);
    double awae = target_init + single_differential_entropy_spec({target, shared, unique_1}) - single_differential_entropy_spec_target({target, shared, unique_1});
    double diff_end_cond = diff_ent_joint_T_exps - diff_ent_joint_exps;


    // cout<<"1: bivariate_RV.mean_vector:\n"<<joint_exps.mean_vector<<endl;
    // cout<<"1: bivariate_RV.cov_matrix:\n"<<joint_exps.cov_matrix<<endl; 
    // cout<<"1: trivariate_ent = "<<diff_ent_joint_T_exps<<endl;
    // cout<<"1: bivariate_ent = "<<diff_ent_joint_exps<<endl;
    // cout<<"1: cond = "<<diff_end_cond<<endl;


    if (num_spec_1 == 0 and num_spec_2 == 0)
    {   
        diff_end_cond = awae;
    }
    

    output_str output = {
        target_params,
        shared_params,
        ex1_params,
        ex2_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        diff_ent_joint_T_exps,
        diff_ent_joint_exps,
        diff_end_cond,
        target_init,
        awae};

    return output;
}

void joint_main() {
    double mu = 0.0;
    vector<double> sigmas{0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0};
    vector<double> sigmas_2{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    vector<double> sigmas_3{1.0, 4.0, 6.0, 9.0, 13.0};
    int counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params = {mu, sigma, 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t i = 1; i < 100; i++) {
            output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, 0, i, i));
        }
        writeData_joint_norm(output, "no_common_spec_" + to_string(counter));
        counter++;
    }
    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params = {mu, sigma, 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t i = 1; i < 100; i++) {
            output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, 1, i, i));
        }
        writeData_joint_norm(output, "one_common_spec_" + to_string(counter));
        counter++;
    }
    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params = {mu, sigma, 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t j = 0; j < 100; j++) {
            for (size_t i = 1; i < 100; i++) {
                output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, j, i, i));
            }
        }
        writeData_joint_norm(output, "vary_shared_and_unique" + string("_") + to_string(counter));
        counter++;

        
    }

    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params = {mu, sigmas.at(0), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t j = 0; j < 100; j++) {
            for (size_t i = 1; i < 100; i++) {
                output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, j, i, i));
            }
        }
        writeData_joint_norm(output, "vary_low_target_shared_and_unique" + string("_") + to_string(counter));
        counter++;

        
    }

    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params = {mu, sigmas.back(), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t j = 0; j < 100; j++) {
            for (size_t i = 1; i < 100; i++) {
                output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, j, i, i));
            }
        }
        writeData_joint_norm(output, "vary_high_target_shared_and_unique" + string("_") + to_string(counter));
        counter++;

        
    }

    vector<uint> num_specs{10, 50, 100};
    for (auto &total_num_spec : num_specs) {

        counter = 0;
        for (auto &sigma : sigmas) {

            normal_params target_params = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (int i = 0; i <= total_num_spec; i++) {
                // num_shared = 2 * i;
                num_shared = i;
                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                                                    num_shared,
                                                    num_ex1,
                                                    num_ex2));
            }
            writeData_joint_norm(output, to_string(total_num_spec) + string("_") + "vary_spec_percent_same_target_" + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params = {mu, sigmas.back(), 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (int i = 0; i <= total_num_spec; i++) {
                num_shared = i;
                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                                                    num_shared,
                                                    num_ex1,
                                                    num_ex2));
            }
            writeData_joint_norm(output, to_string(total_num_spec) + string("_") + "vary_spec_percent_high_target_" + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params = {mu, sigmas.at(0), 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (int i = 0; i <= total_num_spec; i++) {
                num_shared = i;
                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                                                    num_shared,
                                                    num_ex1,
                                                    num_ex2));
            }
            writeData_joint_norm(output, to_string(total_num_spec) + string("_") + "vary_spec_percent_low_target_" + to_string(counter));
            counter++;
        }
    }
    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params = {mu, sigmas.at(0), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t i = 1; i < 100; i++) {
            output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, 0, i, i));
        }
        writeData_joint_norm(output, "low_target_sigma_" + to_string(counter));
        counter++;
    }
    counter = 0;
    for (auto &sigma : sigmas) {

        normal_params target_params = {mu, sigmas.back(), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t i = 1; i < 100; i++) {
            output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, 0, i, i));
        }
        writeData_joint_norm(output, "high_target_sigma_" + to_string(counter));
        counter++;
    }
    counter = 0;
    for (auto &sigma : sigmas_2) {

        normal_params target_params = {mu, sigmas_2.back(), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t i = 1; i < 100; i++) {
            output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, 0, i, i));
        }
        writeData_joint_norm(output, "high_target_nonlinear_sigma2_" + to_string(counter));
        counter++;
    }
    counter = 0;
    for (auto &sigma : sigmas_3) {

        normal_params target_params = {mu, sigmas_3.back(), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;

        // int ctr = 0;
        for (size_t i = 1; i < 100; i++) {
            output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params, 0, i, i));
        }
        writeData_joint_norm(output, "high_target_nonlinear_sigma3_" + to_string(counter));
        counter++;
    }
    counter = 0;





}


void test_single_v1() {
    cout<<"========== VERSION 1 =============\n";

    double mu = 0.0;
    double sigma = 0.25;

    normal_params target_params_1 = {mu, sigma, 0.0};
    normal_params shared_params = {mu, sigma, 0.0};
    normal_params ex_1_params = {mu, sigma, 0.0};
    normal_params ex_2_params = {mu, sigma, 0.0};


    uint num_shared = 0;
    uint num_ex1 = 10;
    uint num_ex2 = 10;
    single_calculation(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2);
}

void writeData_joint_norm(vector<output_struct> out_str, string experiment_name) {
    string distribution;
    string extension = ".csv";
    string experiment = "joint_sum_normal";
    string dir_path = "../output/" + experiment + "/" + experiment_name;

    // making parent directory for all experiments to go into, if it doesnt exist
    // std::system(("mkdir -p ../output/" + experiment + "/" ));
    std::system(("mkdir -p " + dir_path).c_str());

    // // making directory for this specific experiment
    // // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = dir_path + "/results" + extension;
    // // string param_path = path + "/parameters" + extension;

    // // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "target.mu, target.sigma, shared_params.mu, shared_params.sigma, ex1_params.mu, ex1_params.sigma, ex2_params.mu, ex2_params.sigma, num_shared, num_spec_1, num_spec_2, diff_ent_joint_T_exps, diff_ent_joint_exps, diff_ent_cond, target_init, awae \n";

    // for (int i = 0; i < (int)awae_results.size(); i++) {
    //     outFile << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << "," << h_T.at(i) << "," << h_S.at(i) << "," << h_T_S.at(i) << endl;
    // }
    for (auto &outpt : out_str) {
        outFile << outpt.target.mu << "," << outpt.target.sigma << "," << outpt.shared_params.mu << "," << outpt.shared_params.sigma << "," << outpt.ex1_params.mu << "," << outpt.ex1_params.sigma << "," << outpt.ex2_params.mu << "," << outpt.ex2_params.sigma << "," << outpt.num_shared << "," << outpt.num_spec_1 << "," << outpt.num_spec_2 << "," << outpt.diff_ent_joint_T_exps << "," << outpt.diff_ent_joint_exps << "," << outpt.diff_ent_cond << "," << outpt.target_init << "," << outpt.awae << "\n";
    }

    outFile.close();
    outFile.clear();
}

void writeData_joint_norm_two_targets(vector<output_struct> out_str, string experiment_name) {
    string distribution;
    string extension = ".csv";
    string experiment = "joint_sum_normal_two_targets";
    // string dir_path = "../output/" + experiment + "/" + experiment_name;
    string dir_path = "../output/" + experiment + "/" + experiment_name;

    // making parent directory for all experiments to go into, if it doesnt exist
    // std::system(("mkdir -p ../output/" + experiment + "/" ));
    std::system(("mkdir -p " + dir_path).c_str());

    // // making directory for this specific experiment
    // // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = dir_path + "/results" + extension;
    // cout<<path<<endl;
    // // string param_path = path + "/parameters" + extension;

    // // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "target.mu, target.sigma, shared_params.mu, shared_params.sigma, ex1_params.mu, ex1_params.sigma, ex2_params.mu, ex2_params.sigma, num_shared, num_spec_1, num_spec_2, diff_ent_joint_T_exps, diff_ent_joint_exps, diff_ent_cond, target_init, awae \n";

    // for (int i = 0; i < (int)awae_results.size(); i++) {
    //     outFile << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << "," << h_T.at(i) << "," << h_S.at(i) << "," << h_T_S.at(i) << endl;
    // }
    outFile << std::fixed << setprecision(5) << endl;

    for (auto &outpt : out_str) {
        outFile << outpt.target.mu << "," << outpt.target.sigma << "," << outpt.shared_params.mu << "," << outpt.shared_params.sigma << "," << outpt.ex1_params.mu << "," << outpt.ex1_params.sigma << "," << outpt.ex2_params.mu << "," << outpt.ex2_params.sigma << "," << outpt.num_shared << "," << outpt.num_spec_1 << "," << outpt.num_spec_2 << "," << outpt.diff_ent_joint_T_exps << "," << outpt.diff_ent_joint_exps << "," << outpt.diff_ent_cond << "," << outpt.target_init << "," << outpt.awae << "\n";
    }

    outFile.close();
    outFile.clear();
}


