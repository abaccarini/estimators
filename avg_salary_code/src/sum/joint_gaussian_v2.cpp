#include "joint_gaussian_v2.hpp"

// new version is for two RVs always
long double joint_differential_entropy_v2(joint_exps_gaussian_RVs_v2 X_vec) {
    assert(X_vec.mean_vector.size() == 2);
    return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X_vec.cov_matrix.determinant()) / log(2.0);
}
long double single_differential_entropy_v2(sum_gaussian_RVs_v2 X) {
    // cout<<"hi\n";
    return log(sqrt(X.sigma * 2.0 * M_PI * M_E)) / log(2.0);
}

long double single_differential_entropy_v2(gaussian_RV X) {
    return log(sqrt(X.sigma * 2.0 * M_PI * M_E)) / log(2.0);
}

// vector size of one, used for target entropy only
// otherwise, need to construct a sum_gaussian_RV struct
long double single_differential_entropy_v2(vector<gaussian_RV> &X) {
    assert(X.size() == 1);
    return log(sqrt(X.at(0).sigma * 2.0 * M_PI * M_E)) / log(2.0);
}

long double multivariate_differential_entropy(multivariate_joint_gaussian_RV X) {
    switch (X.mean_vector.size()) {
    case 2:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    case 3:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    case 4:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    default:
        cout << "BASE" << endl;
        return 0.0;
    }
}

long double gaussian_cond_entropy(
    vector<gaussian_RV> &target_1,
    vector<gaussian_RV> &target_2,
    vector<gaussian_RV> &shared_s,
    vector<gaussian_RV> &unique_1,
    vector<gaussian_RV> &unique_2) {

    // grouping X_T_1 and unique_1 together
    // grouping X_T_2 and unique_2 together
    vector<gaussian_RV> X_1(target_1);
    vector<gaussian_RV> X_2(target_2);
    X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());
    X_2.insert(X_2.end(), unique_2.begin(), unique_2.end());

    return trivariate_joint_gassian_entropy(target_1, target_2, shared_s, unique_1, unique_2) -
           joint_differential_entropy_v2({
               {shared_s, X_1}, // X_1
               {shared_s, X_2}  // X_2
           });
}

long double trivariate_joint_gassian_entropy(
    vector<gaussian_RV> &target_1,
    vector<gaussian_RV> &target_2,
    vector<gaussian_RV> &shared_s,
    vector<gaussian_RV> &unique_1,
    vector<gaussian_RV> &unique_2) {

    long double a = 0.0, c = 0.0;

    a = single_differential_entropy_v2(target_1);

    vector<gaussian_RV> X_2(target_2);
    X_2.insert(X_2.end(), unique_2.begin(), unique_2.end());

    c = joint_differential_entropy_v2({{shared_s, X_2},
                                       {shared_s, unique_1}});

    return a + c;
    // d = single_differential_entropy_v2({shared_s, unique_1});
}

// independent of whether or not the target participates multiple times
// or if other targets exist (handled when we create X when we call this function)
long double term_1(
    vector<gaussian_RV> &target_1,
    multivariate_joint_gaussian_RV X) {

    long double a = single_differential_entropy_v2(target_1);
    long double c = multivariate_differential_entropy(X);

    return a + c;
}

output_str single_calculation_two_targets(normal_params &target_params_1, normal_params &target_params_2, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2) {
    // gaussian_RV target_1 = {target_params_1.mu, target_params_1.sigma};
    // gaussian_RV target_2 = {target_params_2.mu, target_params_2.sigma};
    vector<gaussian_RV> target_1;
    vector<gaussian_RV> target_2;
    target_1.push_back({target_params_1.mu, target_params_1.sigma});
    target_2.push_back({target_params_2.mu, target_params_2.sigma});

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }
    vector<gaussian_RV> X_1(target_1);
    vector<gaussian_RV> X_2(target_2);
    X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());
    X_2.insert(X_2.end(), unique_2.begin(), unique_2.end());

    long double trivariate_ent, bivariate_ent, target_1_init, awae, cond;

    trivariate_ent = trivariate_joint_gassian_entropy(target_1,
                                                      target_2,
                                                      shared_s,
                                                      unique_1,
                                                      unique_2);

    bivariate_ent = joint_differential_entropy_v2({
        {shared_s, X_1}, // X_1 = X_S + (X_T_1 + X_S_1)
        {shared_s, X_2}  // X_2 = X_S + (X_T_2 + X_S_2)
    });

    cond = trivariate_ent - bivariate_ent;

    target_1_init = single_differential_entropy_v2(target_1);

    awae = target_1_init + single_differential_entropy_v2({shared_s, unique_1}) - single_differential_entropy_v2({X_1, shared_s});

    output_str output = {
        target_params_1,
        shared_params,
        ex1_params,
        ex2_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        trivariate_ent,
        bivariate_ent,
        cond,
        target_1_init,
        awae};

    return output;
}

// target particiaptes only once in exp_1
output_str single_calculation_one_target_one_exp(normal_params &target_params_1, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2) {
    // gaussian_RV target_1 = {target_params_1.mu, target_params_1.sigma};
    // gaussian_RV target_2 = {target_params_2.mu, target_params_2.sigma};
    vector<gaussian_RV> target_1;
    target_1.push_back({target_params_1.mu, target_params_1.sigma});

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }
    vector<gaussian_RV> X_1(target_1);
    X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());

    long double trivariate_ent, bivariate_ent, target_1_init, awae, cond;

    trivariate_ent = trivariate_joint_gassian_entropy_one_target(target_1,
                                                                 shared_s,
                                                                 unique_1,
                                                                 unique_2);

    bivariate_ent = joint_differential_entropy_v2({
        {shared_s, X_1},     // X_1 = X_S + (X_T_1 + X_S_1)
        {shared_s, unique_2} // X_2 = X_S + X_S_2
    });

    cond = trivariate_ent - bivariate_ent;

    vector<gaussian_RV> X_1_exp_1(target_1);
    X_1_exp_1.insert(X_1_exp_1.end(), shared_s.begin(), shared_s.end());

    target_1_init = single_differential_entropy_v2(target_1);
    awae = target_1_init + single_differential_entropy_v2({shared_s, unique_1}) - single_differential_entropy_v2({X_1_exp_1, unique_1});

    if (num_spec_1 == 0 and num_spec_2 == 0) {
        // cond = 0;
    }

    output_str output = {
        target_params_1,
        shared_params,
        ex1_params,
        ex2_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        trivariate_ent,
        bivariate_ent,
        cond,
        target_1_init,
        awae};

    return output;
}

// target particiaptes only once in exp_1
output_str single_calculation_one_target_one_exp_second_not_first(normal_params &target_params_1, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2) {
    // gaussian_RV target_1 = {target_params_1.mu, target_params_1.sigma};
    // gaussian_RV target_2 = {target_params_2.mu, target_params_2.sigma};
    vector<gaussian_RV> target_1;
    target_1.push_back({target_params_1.mu, target_params_1.sigma});

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }
    vector<gaussian_RV> X_1(target_1);
    X_1.insert(X_1.end(), unique_2.begin(), unique_2.end());

    long double trivariate_ent, bivariate_ent, target_1_init, awae, cond;

    trivariate_ent = trivariate_joint_gassian_entropy_one_target(target_1,
                                                                 shared_s,
                                                                 unique_1,
                                                                 unique_2);

    bivariate_ent = joint_differential_entropy_v2({
        {shared_s, unique_1}, // X_1 = X_S +  X_S_1
        {shared_s, X_1}       // X_2 = X_S + (X_T_1 + X_S_2)
    });

    cond = trivariate_ent - bivariate_ent;

    vector<gaussian_RV> X_1_exp_1(target_1);
    X_1_exp_1.insert(X_1_exp_1.end(), shared_s.begin(), shared_s.end());

    target_1_init = single_differential_entropy_v2(target_1);
    awae = target_1_init + single_differential_entropy_v2({shared_s, unique_2}) - single_differential_entropy_v2({X_1_exp_1, unique_2});

    if (num_spec_1 == 0 and num_spec_2 == 0) {
        // cond = 0;
    }

    output_str output = {
        target_params_1,
        shared_params,
        ex1_params,
        ex2_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        trivariate_ent,
        bivariate_ent,
        cond,
        target_1_init,
        awae};

    return output;
}
// this should be the same as the v1 joint experioments
output_str single_calculation_one_target_two_exp(normal_params &target_params_1, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, uint num_shared, uint num_spec_1, uint num_spec_2) {
    // gaussian_RV target_1 = {target_params_1.mu, target_params_1.sigma};
    // gaussian_RV target_2 = {target_params_2.mu, target_params_2.sigma};
    vector<gaussian_RV> target_1;
    target_1.push_back({target_params_1.mu, target_params_1.sigma});
    // cout<<"target -{"<<target_1.at(0).mu<<", "<<target_1.at(0).sigma<<"}\n";

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }
    vector<gaussian_RV> X_1(target_1);
    // vector<gaussian_RV> X_2(target_1);
    X_1.insert(X_1.end(), shared_s.begin(), shared_s.end());
    // X_2.insert(X_2.end(), unique_2.begin(), unique_2.end());

    long double trivariate_ent, bivariate_ent, target_1_init, awae, cond;

    trivariate_ent = trivariate_joint_gassian_entropy_one_target(target_1,
                                                                 shared_s,
                                                                 unique_1,
                                                                 unique_2);

    bivariate_ent = joint_differential_entropy_v2({
        // {shared_s, X_1}, // X_1 = X_S + (X_T + X_S_1)
        // {shared_s, X_2}  // X_2 = X_S + (X_T + X_S_2)
        {X_1, unique_1}, // X_1 = (X_S + X_T) + X_S_1)
        {X_1, unique_2}  // X_2 = (X_S + X_T) + X_S_2)

    });

    cond = trivariate_ent - bivariate_ent;

    vector<gaussian_RV> X_1_exp_1(target_1);
    X_1_exp_1.insert(X_1_exp_1.end(), shared_s.begin(), shared_s.end());

    target_1_init = single_differential_entropy_v2(target_1);
    awae = target_1_init + single_differential_entropy_v2({shared_s, unique_1}) - single_differential_entropy_v2({X_1_exp_1, unique_1});

    if (num_spec_1 == 0 and num_spec_2 == 0) {
        cond = awae;
    }

    output_str output = {
        target_params_1,
        shared_params,
        ex1_params,
        ex2_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        trivariate_ent,
        bivariate_ent,
        cond,
        target_1_init,
        awae};

    return output;
}

// target only participates in exp_1
long double trivariate_joint_gassian_entropy_one_target(
    vector<gaussian_RV> &target_1,
    vector<gaussian_RV> &shared_s,
    vector<gaussian_RV> &unique_1,
    vector<gaussian_RV> &unique_2) {

    long double a = 0.0, c = 0.0;

    a = single_differential_entropy_v2(target_1);

    c = joint_differential_entropy_v2({{shared_s, unique_1},
                                       {shared_s, unique_2}});

    return a + c;
    // d = single_differential_entropy_v2({shared_s, unique_1});
}

void joint_main_two_targets() {
    double mu = 0.0;
    vector<double> sigmas{0.25, 0.5, 1.0, 2.0, 4.0};
    // vector<double> sigmas{0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0};

    vector<uint> num_specs{
        5,
        6,
        7,
        8,
        9,
        10,
        20,
        24,
        30,
        31,
        32,
        33,
        34,
        36,
        38,
        39,
        40,
        41,
        42,
    };

    // vector<double> sigmas{0.25, 0.5};
    int counter = 0;
    size_t upper_bound = 100;
    for (auto &sigma : sigmas) {

        normal_params target_params_1 = {mu, sigma, 0.0};
        normal_params target_params_2 = {mu, sigma, 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;
        // printf("%9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s,\n",
        //        "shared",
        //        "spec_1",
        //        "spec_2",
        //        "H_T",
        //        "awae",
        //        "trivar",
        //        "bivar",
        //        "cond",
        //        "time(s)");

        // int ctr = 0;
        for (size_t i = 1; i < upper_bound; i++) {
            output.push_back(single_calculation_two_targets(target_params_1, target_params_2, shared_params, ex_1_params, ex_2_params, 0, i, i)); // i + 1
        }
        writeData_joint_norm_two_targets(output, "two_targets_ncs" + string("_") + to_string(counter));
        counter++;
    }
    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params_1 = {mu, sigma, 0.0};
        normal_params target_params_2 = {mu, sigma, 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;
        // int ctr = 0;
        for (size_t i = 1; i < upper_bound; i++) {
            output.push_back(single_calculation_two_targets(target_params_1, target_params_2, shared_params, ex_1_params, ex_2_params, 1, i, i)); // i + 1
        }
        writeData_joint_norm_two_targets(output, "two_targets_one_cs" + string("_") + to_string(counter));
        counter++;
    }
    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params_1 = {mu, sigmas.back(), 0.0};
        normal_params target_params_2 = {mu, sigmas.back(), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;
        // printf("%9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s,\n",
        //        "shared",
        //        "spec_1",
        //        "spec_2",
        //        "H_T",
        //        "awae",
        //        "trivar",
        //        "bivar",
        //        "cond",
        //        "time(s)");

        // int ctr = 0;
        for (size_t i = 1; i < upper_bound; i++) {
            output.push_back(single_calculation_two_targets(target_params_1, target_params_2, shared_params, ex_1_params, ex_2_params, 1, i, i)); // i + 1
        }
        writeData_joint_norm_two_targets(output, "two_targets_one_high_target_sigma_cs" + string("_") + to_string(counter));
        counter++;
    }

    counter = 0;

    for (auto &sigma : sigmas) {

        normal_params target_params_1 = {mu, sigmas.at(0), 0.0};
        normal_params target_params_2 = {mu, sigmas.at(0), 0.0};
        normal_params shared_params = {mu, sigma, 0.0};
        normal_params ex_1_params = {mu, sigma, 0.0};
        normal_params ex_2_params = {mu, sigma, 0.0};

        vector<output_str> output;
        // printf("%9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s,\n",
        //        "shared",
        //        "spec_1",
        //        "spec_2",
        //        "H_T",
        //        "awae",
        //        "trivar",
        //        "bivar",
        //        "cond",
        //        "time(s)");

        // int ctr = 0;
        for (size_t i = 1; i < upper_bound; i++) {
            output.push_back(single_calculation_two_targets(target_params_1, target_params_2, shared_params, ex_1_params, ex_2_params, 1, i, i)); // i + 1
        }
        // writeData_joint_norm_two_targets(output, "two_targets_one_low_target_sigma_cs" + string("_") + to_string(counter));
        counter++;
    }
    counter = 0;
    // uint total_num_spec = 10;
    for (auto &total_num_spec : num_specs) {

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigma, 0.0};
            normal_params target_params_2 = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_two_targets(target_params_1, target_params_2, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "two_targets_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigmas.at(0), 0.0};
            normal_params target_params_2 = {mu, sigmas.at(0), 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_two_targets(target_params_1, target_params_2, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            // writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "two_targets_low_target_sigma_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_one_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_one_exp_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_two_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_two_exp_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;
        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_one_exp_second_not_first(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_one_exp_second_not_first_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigmas.at(0), 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;

                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_one_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            // cout<< to_string(total_num_spec) + string("_") + "one_target_low_one_exp_vary_spec_percent" + string("_") + to_string(counter)<<endl;

            // writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_low_one_exp_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigmas.at(0), 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_two_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            // writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_low_two_exp_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigmas.back(), 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_one_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            // writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_high_one_exp_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigmas.back(), 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};

            vector<output_str> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_two_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            // writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_high_two_exp_vary_spec_percent" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;
    }

    // uint total_num_spec = 10;
    // for (auto &sigma : sigmas) {

    //     normal_params target_params_1 = {mu, sigma, 0.0};
    //     normal_params shared_params = {mu, sigma, 0.0};
    //     normal_params ex_1_params = {mu, sigma, 0.0};
    //     normal_params ex_2_params = {mu, sigma, 0.0};

    //     vector<output_str> output;

    //     // int ctr = 0;
    //     uint num_shared = 0,
    //          num_ex1 = 0,
    //          num_ex2 = 0;
    //     for (uint i = 0; i <= total_num_spec; i++) {
    //         num_shared = i;

    //         num_ex1 = total_num_spec - i;
    //         num_ex2 = total_num_spec - i;
    //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
    //         output.push_back(single_calculation_one_target_one_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

    //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
    //         //                                     num_shared,
    //         //                                     num_ex1,
    //         //                                     num_ex2));
    //     }
    //     writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_one_exp_vary_spec_percent" + string("_") + to_string(counter));
    //     counter++;
    // }
    // counter = 0;

    // for (auto &sigma : sigmas) {

    //     normal_params target_params_1 = {mu, sigma, 0.0};
    //     normal_params shared_params = {mu, sigma, 0.0};
    //     normal_params ex_1_params = {mu, sigma, 0.0};
    //     normal_params ex_2_params = {mu, sigma, 0.0};

    //     vector<output_str> output;

    //     // int ctr = 0;
    //     uint num_shared = 0,
    //          num_ex1 = 0,
    //          num_ex2 = 0;
    //     for (uint i = 0; i <= total_num_spec; i++) {
    //         num_shared = i;

    //         num_ex1 = total_num_spec - i;
    //         num_ex2 = total_num_spec - i;
    //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
    //         output.push_back(single_calculation_one_target_two_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2)); // i + 1

    //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
    //         //                                     num_shared,
    //         //                                     num_ex1,
    //         //                                     num_ex2));
    //     }
    //     writeData_joint_norm_two_targets(output, to_string(total_num_spec) + string("_") + "one_target_two_exp_vary_spec_percent" + string("_") + to_string(counter));
    //     counter++;
    // }
    // counter = 0;
}

void test_single_v2() {
    cout << "========== VERSION 2 =============\n";
    double mu = 0.0;
    double sigma = 0.25;

    normal_params target_params_1 = {mu, sigma, 0.0};
    normal_params shared_params = {mu, sigma, 0.0};
    normal_params ex_1_params = {mu, sigma, 0.0};
    normal_params ex_2_params = {mu, sigma, 0.0};

    vector<output_str> output;

    uint num_shared = 0;
    uint num_ex1 = 10;
    uint num_ex2 = 10;
    single_calculation_one_target_two_exp(target_params_1, shared_params, ex_1_params, ex_2_params, num_shared, num_ex1, num_ex2);
}

void joint_main_multiple_exps() {

    double mu = 1.0, sigma = 2.0;

    uint num_shared = 1, num_spec_1 = 3, num_spec_2 = 3, num_spec_3 = 3;

    normal_params target_params_1 = {mu, sigma, 0.0};
    normal_params shared_params = {mu, sigma, 0.0};
    normal_params ex_1_params = {mu, sigma, 0.0};
    normal_params ex_2_params = {mu, sigma, 0.0};
    normal_params ex_3_params = {mu, sigma, 0.0};

    vector<gaussian_RV> target;
    vector<gaussian_RV> target_2;
    target.push_back({target_params_1.mu, target_params_1.sigma});

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    vector<gaussian_RV> unique_3;
    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex_1_params.mu, ex_1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex_2_params.mu, ex_2_params.sigma});
    }
    for (size_t i = 0; i < num_spec_3; i++) {
        unique_3.push_back({ex_3_params.mu, ex_3_params.sigma});
    }

    vector<gaussian_RV> X_T_S(target);

    vector<gaussian_RV> X_1(target);
    vector<gaussian_RV> X_2(target);
    vector<gaussian_RV> X_3(target);
    X_T_S.insert(X_T_S.end(), shared_s.begin(), shared_s.end());

    X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());

    X_2.insert(X_2.end(), unique_2.begin(), unique_2.end());
    X_3.insert(X_3.end(), unique_3.begin(), unique_3.end());

    multivariate_joint_gaussian_RV X_trivar = {{{shared_s, X_1},
                                                {shared_s, unique_2},
                                                {shared_s, unique_3}}}; // (x_t + x_s + x_s_1, x_t + x_s + x_s_2, x_t + x_s + x_s_3)

    cout << "x_trivar.mu_vec: \n";
    cout << X_trivar.mean_vector << endl;

    cout << "x_trivar.cov_matrix: \n";
    cout << X_trivar.cov_matrix << endl;
}

// X_T participates in ALL experiments
output_str_3exp single_calculation_one_target_three_out_of_three_exp(normal_params &target_params_1, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, normal_params &ex3_params, uint num_shared, uint num_spec_1, uint num_spec_2, uint num_spec_3) {
    // gaussian_RV target_1 = {target_params_1.mu, target_params_1.sigma};
    // gaussian_RV target_2 = {target_params_2.mu, target_params_2.sigma};
    vector<gaussian_RV> target_1;
    target_1.push_back({target_params_1.mu, target_params_1.sigma});
    // cout<<"target -{"<<target_1.at(0).mu<<", "<<target_1.at(0).sigma<<"}\n";

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    vector<gaussian_RV> unique_3;

    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_3.push_back({ex3_params.mu, ex3_params.sigma});
    }
    vector<gaussian_RV> X_1(target_1);

    X_1.insert(X_1.end(), shared_s.begin(), shared_s.end());

    long double t_1 = 0.0, t_2 = 0.0, target_1_init, one_exp, three_exp, two_exp;

    t_1 = trivariate_joint_gassian_entropy_one_target(target_1,
                                                      shared_s,
                                                      unique_1,
                                                      unique_2);
    t_2 = joint_differential_entropy_v2({
        {X_1, unique_1}, // X_1 = (X_S + X_T) + X_S_1)
        {X_1, unique_2}  // X_2 = (X_S + X_T) + X_S_2)

    });
    two_exp = t_1 - t_2;

    t_1 = term_1(target_1,
                 {{{shared_s, unique_1},
                   {shared_s, unique_2},
                   {shared_s, unique_3}}});
    t_2 = multivariate_differential_entropy({{
        {X_1, unique_1}, // X_1 = (X_S + X_T) + X_S_1
        {X_1, unique_2}, // X_2 = (X_S + X_T) + X_S_2
        {X_1, unique_3}  // X_2 = (X_S + X_T) + X_S_3
    }});
    three_exp = t_1 - t_2;

    target_1_init = single_differential_entropy_v2(target_1);
    one_exp = target_1_init + single_differential_entropy_v2({shared_s, unique_1}) - single_differential_entropy_v2({X_1, unique_1});

    if (num_spec_1 == 0 and num_spec_2 == 0 and num_spec_3 == 0) {
        two_exp = one_exp;
        three_exp = one_exp;
    }

    output_str_3exp output = {
        target_params_1,
        shared_params,
        ex1_params,
        ex2_params,
        ex3_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        num_spec_3,
        target_1_init,
        one_exp,
        two_exp,
        three_exp};

    return output;
}

// X_T participates in TWO OUT OF THREE experiments (X_1 and X_2 but NOT X_3)
output_str_3exp single_calculation_one_target_two_out_of_three_exp(normal_params &target_params_1, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, normal_params &ex3_params, uint num_shared, uint num_spec_1, uint num_spec_2, uint num_spec_3) {
    // gaussian_RV target_1 = {target_params_1.mu, target_params_1.sigma};
    // gaussian_RV target_2 = {target_params_2.mu, target_params_2.sigma};
    vector<gaussian_RV> target_1;
    target_1.push_back({target_params_1.mu, target_params_1.sigma});
    // cout<<"target -{"<<target_1.at(0).mu<<", "<<target_1.at(0).sigma<<"}\n";

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    vector<gaussian_RV> unique_3;

    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_3.push_back({ex3_params.mu, ex3_params.sigma});
    }
    vector<gaussian_RV> X_1(target_1);
    X_1.insert(X_1.end(), shared_s.begin(), shared_s.end());

    long double t_1 = 0.0, t_2 = 0.0, target_1_init, one_exp, three_exp, two_exp;

    t_1 = trivariate_joint_gassian_entropy_one_target(target_1,
                                                      shared_s,
                                                      unique_1,
                                                      unique_2);
    t_2 = joint_differential_entropy_v2({
        {X_1, unique_1}, // X_1 = (X_S + X_T) + X_S_1)
        {X_1, unique_2}  // X_2 = (X_S + X_T) + X_S_2)

    });
    two_exp = t_1 - t_2;

    t_1 = term_1(target_1,
                 {{{shared_s, unique_1},
                   {shared_s, unique_2},
                   {shared_s, unique_3}}});
    // cout << "term_1 = " << t_1 << endl;
    sum_gaussian_RVs_v2 sum_spectators = {shared_s, {}}; // only doing this so we can retrieve the sum of all shared_Spec sigmas

    // sum_gaussian_RVs_v2 sum_1 = {shared_s, unique_1}; // only doing this so we can retrieve the sum of all shared_Spec sigmas
    // sum_gaussian_RVs_v2 sum_2 = {shared_s, unique_2};
    sum_gaussian_RVs_v2 sum_3 = {shared_s, unique_3};

    // doing this so we can modify cov_matrix for this specific configuration
    // NEED TO DO SEMI_MAJOR REVISION TO SPECIFY SHARED RVS among 3 exps
    multivariate_joint_gaussian_RV mv_gaussian = {{
        {X_1, unique_1}, // X_1 = (X_S) + X_T + X_S_1
        {X_1, unique_2}, // X_2 = (X_S) + X_T + X_S_2
        {X_1, unique_3}  // X_2 = (X_S)       + X_S_3
    }};

    mv_gaussian.cov_matrix(0, 2) = sum_spectators.sigma;
    mv_gaussian.cov_matrix(1, 2) = sum_spectators.sigma;
    mv_gaussian.cov_matrix(2, 0) = sum_spectators.sigma;
    mv_gaussian.cov_matrix(2, 1) = sum_spectators.sigma;

    mv_gaussian.cov_matrix(2, 2) = sum_3.sigma;
    // cout << mv_gaussian.cov_matrix << endl;
    // cout << "x_vec size = " << mv_gaussian.X_vec.size() << endl;
    // cout << "x_vec size = " << mv_gaussian.mean_vector.size() << endl;
    t_2 = multivariate_differential_entropy(mv_gaussian);
    // cout << "t_2 = " << t_2 << endl;

    three_exp = t_1 - t_2; // PROBLEM HERE PROBLEM HERE PROBLEM HERE PROBLEM HERE PROBLEM HERE

    vector<gaussian_RV> X_1_exp_1(target_1);
    X_1_exp_1.insert(X_1_exp_1.end(), shared_s.begin(), shared_s.end());

    target_1_init = single_differential_entropy_v2(target_1);
    one_exp = target_1_init + single_differential_entropy_v2({shared_s, unique_1}) - single_differential_entropy_v2({X_1_exp_1, unique_1});

    if (num_spec_1 == 0 and num_spec_2 == 0 and num_spec_3 == 0) {
        two_exp = one_exp;
        // three_exp = 0;
    }

    output_str_3exp output = {
        target_params_1,
        shared_params,
        ex1_params,
        ex2_params,
        ex3_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        num_spec_3,
        target_1_init,
        one_exp,
        two_exp,
        three_exp};

    return output;
}

// X_T participates in ONE OUT OF THREE experiments (X_1 but NOT X_2 and X_3)
output_str_3exp single_calculation_one_target_one_out_of_three_exp(normal_params &target_params_1, normal_params &shared_params, normal_params &ex1_params, normal_params &ex2_params, normal_params &ex3_params, uint num_shared, uint num_spec_1, uint num_spec_2, uint num_spec_3) {
    // gaussian_RV target_1 = {target_params_1.mu, target_params_1.sigma};
    // gaussian_RV target_2 = {target_params_2.mu, target_params_2.sigma};
    vector<gaussian_RV> target_1;
    target_1.push_back({target_params_1.mu, target_params_1.sigma});
    // cout<<"target -{"<<target_1.at(0).mu<<", "<<target_1.at(0).sigma<<"}\n";

    vector<gaussian_RV> shared_s;
    vector<gaussian_RV> unique_1;
    vector<gaussian_RV> unique_2;
    vector<gaussian_RV> unique_3;

    for (size_t i = 0; i < num_shared; i++) {
        shared_s.push_back({shared_params.mu, shared_params.sigma});
    }
    for (size_t i = 0; i < num_spec_1; i++) {
        unique_1.push_back({ex1_params.mu, ex1_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_2.push_back({ex2_params.mu, ex2_params.sigma});
    }
    for (size_t i = 0; i < num_spec_2; i++) {
        unique_3.push_back({ex3_params.mu, ex3_params.sigma});
    }
    vector<gaussian_RV> X_1(target_1);

    // X_1.insert(X_1.end(), shared_s.begin(), shared_s.end());
    X_1.insert(X_1.end(), unique_1.begin(), unique_1.end());

    long double t_1 = 0.0, t_2 = 0.0, target_1_init, one_exp, three_exp, two_exp;

    t_1 = trivariate_joint_gassian_entropy_one_target(target_1,
                                                      shared_s,
                                                      unique_1,
                                                      unique_2);
    t_2 = joint_differential_entropy_v2({
        {shared_s, X_1},     // X_1 = X_S + (X_T_1 + X_S_1)
        {shared_s, unique_2} // X_2 = X_S + X_S_2
    });

    two_exp = t_1 - t_2;

    t_1 = term_1(target_1,
                 {{{shared_s, unique_1},
                   {shared_s, unique_2},
                   {shared_s, unique_3}}});

    t_2 = multivariate_differential_entropy({{
        {shared_s, X_1},      // X_1 = X_S + (X_T + X_S_1)
        {shared_s, unique_2}, // X_2 = X_S + X_S_2
        {shared_s, unique_3}  // X_2 = X_S + X_S_3
    }});

    three_exp = t_1 - t_2;

    vector<gaussian_RV> X_1_exp_1(target_1);
    X_1_exp_1.insert(X_1_exp_1.end(), shared_s.begin(), shared_s.end());

    target_1_init = single_differential_entropy_v2(target_1);
    one_exp = target_1_init + single_differential_entropy_v2({shared_s, unique_1}) - single_differential_entropy_v2({X_1_exp_1, unique_1});

    if (num_spec_1 == 0 and num_spec_2 == 0 and num_spec_3 == 0) {
        two_exp = 0;
        three_exp = 0;
    }

    output_str_3exp output = {
        target_params_1,
        shared_params,
        ex1_params,
        ex2_params,
        ex3_params,
        num_shared,
        num_spec_1,
        num_spec_2,
        num_spec_3,
        target_1_init,
        one_exp,
        two_exp,
        three_exp};

    return output;
}

void joint_main_3exp() {

    double mu = 0.0;
    vector<double> sigmas{0.25, 0.5, 1.0, 2.0, 4.0, 8.0};
    // vector<double> sigmas{0.25, 0.5};
    int counter = 0;
    size_t upper_bound = 100;
    vector<uint> num_specs{6, 10, 24};
    // vector<uint> num_specs{6};

    // for (auto &sigma : sigmas) {

    //     normal_params target_params_1 = {mu, sigma, 0.0};
    //     normal_params shared_params = {mu, sigma, 0.0};
    //     normal_params ex_1_params = {mu, sigma, 0.0};
    //     normal_params ex_2_params = {mu, sigma, 0.0};
    //     normal_params ex_3_params = {mu, sigma, 0.0};

    //     vector<output_str_3exp> output;

    //     for (size_t i = 1; i < upper_bound; i++) {
    //         output.push_back(single_calculation_one_target_three_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, 0, i, i, i));
    //     }
    //     writeData_joint_norm_three_exps(output, "one_T_3_3_no_common_spec" + string("_") + to_string(counter));
    //     counter++;
    // }
    // counter = 0;

    // for (auto &sigma : sigmas) {

    //     normal_params target_params_1 = {mu, sigma, 0.0};
    //     normal_params shared_params = {mu, sigma, 0.0};
    //     normal_params ex_1_params = {mu, sigma, 0.0};
    //     normal_params ex_2_params = {mu, sigma, 0.0};
    //     normal_params ex_3_params = {mu, sigma, 0.0};

    //     vector<output_str_3exp> output;

    //     for (size_t i = 1; i < upper_bound; i++) {
    //         output.push_back(single_calculation_one_target_three_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, 1, i, i, i));
    //     }
    //     writeData_joint_norm_three_exps(output, "one_T_3_3_one_common_spec" + string("_") + to_string(counter));
    //     counter++;
    // }
    counter = 0;

    for (auto &total_num_spec : num_specs) {

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};
            normal_params ex_3_params = {mu, sigma, 0.0};

            vector<output_str_3exp> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0,
                 num_ex3 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                num_ex3 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_three_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_3_3_vsp" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;

        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};
            normal_params ex_3_params = {mu, sigma, 0.0};

            vector<output_str_3exp> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0,
                 num_ex3 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                num_ex3 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_two_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_2_3_vsp" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;
        for (auto &sigma : sigmas) {

            normal_params target_params_1 = {mu, sigma, 0.0};
            normal_params shared_params = {mu, sigma, 0.0};
            normal_params ex_1_params = {mu, sigma, 0.0};
            normal_params ex_2_params = {mu, sigma, 0.0};
            normal_params ex_3_params = {mu, sigma, 0.0};

            vector<output_str_3exp> output;

            // int ctr = 0;
            uint num_shared = 0,
                 num_ex1 = 0,
                 num_ex2 = 0,
                 num_ex3 = 0;
            for (uint i = 0; i <= total_num_spec; i++) {
                num_shared = i;

                num_ex1 = total_num_spec - i;
                num_ex2 = total_num_spec - i;
                num_ex3 = total_num_spec - i;
                // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
                output.push_back(single_calculation_one_target_one_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

                // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
                //                                     num_shared,
                //                                     num_ex1,
                //                                     num_ex2));
            }
            writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_1_3_vsp" + string("_") + to_string(counter));
            counter++;
        }
        counter = 0;
        // /////////////////////////////////

        // for (auto &sigma : sigmas) {

        //     normal_params target_params_1 = {mu, sigmas.back(), 0.0};
        //     normal_params shared_params = {mu, sigma, 0.0};
        //     normal_params ex_1_params = {mu, sigma, 0.0};
        //     normal_params ex_2_params = {mu, sigma, 0.0};
        //     normal_params ex_3_params = {mu, sigma, 0.0};

        //     vector<output_str_3exp> output;

        //     // int ctr = 0;
        //     uint num_shared = 0,
        //          num_ex1 = 0,
        //          num_ex2 = 0,
        //          num_ex3 = 0;
        //     for (uint i = 0; i <= total_num_spec; i++) {
        //         num_shared = i;

        //         num_ex1 = total_num_spec - i;
        //         num_ex2 = total_num_spec - i;
        //         num_ex3 = total_num_spec - i;
        //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
        //         output.push_back(single_calculation_one_target_three_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

        //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
        //         //                                     num_shared,
        //         //                                     num_ex1,
        //         //                                     num_ex2));
        //     }
        //     writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_high_T_3_3_vsp" + string("_") + to_string(counter));
        //     counter++;
        // }
        // counter = 0;

        // for (auto &sigma : sigmas) {

        //     normal_params target_params_1 = {mu, sigmas.back(), 0.0};
        //     normal_params shared_params = {mu, sigma, 0.0};
        //     normal_params ex_1_params = {mu, sigma, 0.0};
        //     normal_params ex_2_params = {mu, sigma, 0.0};
        //     normal_params ex_3_params = {mu, sigma, 0.0};

        //     vector<output_str_3exp> output;

        //     // int ctr = 0;
        //     uint num_shared = 0,
        //          num_ex1 = 0,
        //          num_ex2 = 0,
        //          num_ex3 = 0;
        //     for (uint i = 0; i <= total_num_spec; i++) {
        //         num_shared = i;

        //         num_ex1 = total_num_spec - i;
        //         num_ex2 = total_num_spec - i;
        //         num_ex3 = total_num_spec - i;
        //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
        //         output.push_back(single_calculation_one_target_two_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

        //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
        //         //                                     num_shared,
        //         //                                     num_ex1,
        //         //                                     num_ex2));
        //     }
        //     writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_high_T_2_3_vsp" + string("_") + to_string(counter));
        //     counter++;
        // }
        // counter = 0;
        // for (auto &sigma : sigmas) {

        //     normal_params target_params_1 = {mu, sigmas.back(), 0.0};
        //     normal_params shared_params = {mu, sigma, 0.0};
        //     normal_params ex_1_params = {mu, sigma, 0.0};
        //     normal_params ex_2_params = {mu, sigma, 0.0};
        //     normal_params ex_3_params = {mu, sigma, 0.0};

        //     vector<output_str_3exp> output;

        //     // int ctr = 0;
        //     uint num_shared = 0,
        //          num_ex1 = 0,
        //          num_ex2 = 0,
        //          num_ex3 = 0;
        //     for (uint i = 0; i <= total_num_spec; i++) {
        //         num_shared = i;

        //         num_ex1 = total_num_spec - i;
        //         num_ex2 = total_num_spec - i;
        //         num_ex3 = total_num_spec - i;
        //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
        //         output.push_back(single_calculation_one_target_one_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

        //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
        //         //                                     num_shared,
        //         //                                     num_ex1,
        //         //                                     num_ex2));
        //     }
        //     writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_high_T_1_3_vsp" + string("_") + to_string(counter));
        //     counter++;
        // }
        // counter = 0;
        // /////////////////////////////
        // for (auto &sigma : sigmas) {

        //     normal_params target_params_1 = {mu, sigmas.at(0), 0.0};
        //     normal_params shared_params = {mu, sigma, 0.0};
        //     normal_params ex_1_params = {mu, sigma, 0.0};
        //     normal_params ex_2_params = {mu, sigma, 0.0};
        //     normal_params ex_3_params = {mu, sigma, 0.0};

        //     vector<output_str_3exp> output;

        //     // int ctr = 0;
        //     uint num_shared = 0,
        //          num_ex1 = 0,
        //          num_ex2 = 0,
        //          num_ex3 = 0;
        //     for (uint i = 0; i <= total_num_spec; i++) {
        //         num_shared = i;

        //         num_ex1 = total_num_spec - i;
        //         num_ex2 = total_num_spec - i;
        //         num_ex3 = total_num_spec - i;
        //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
        //         output.push_back(single_calculation_one_target_three_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

        //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
        //         //                                     num_shared,
        //         //                                     num_ex1,
        //         //                                     num_ex2));
        //     }
        //     writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_low_T_3_3_vsp" + string("_") + to_string(counter));
        //     counter++;
        // }
        // counter = 0;

        // for (auto &sigma : sigmas) {

        //     normal_params target_params_1 = {mu, sigmas.at(0), 0.0};
        //     normal_params shared_params = {mu, sigma, 0.0};
        //     normal_params ex_1_params = {mu, sigma, 0.0};
        //     normal_params ex_2_params = {mu, sigma, 0.0};
        //     normal_params ex_3_params = {mu, sigma, 0.0};

        //     vector<output_str_3exp> output;

        //     // int ctr = 0;
        //     uint num_shared = 0,
        //          num_ex1 = 0,
        //          num_ex2 = 0,
        //          num_ex3 = 0;
        //     for (uint i = 0; i <= total_num_spec; i++) {
        //         num_shared = i;

        //         num_ex1 = total_num_spec - i;
        //         num_ex2 = total_num_spec - i;
        //         num_ex3 = total_num_spec - i;
        //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
        //         output.push_back(single_calculation_one_target_two_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

        //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
        //         //                                     num_shared,
        //         //                                     num_ex1,
        //         //                                     num_ex2));
        //     }
        //     writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_low_T_2_3_vsp" + string("_") + to_string(counter));
        //     counter++;
        // }
        // counter = 0;
        // for (auto &sigma : sigmas) {

        //     normal_params target_params_1 = {mu, sigmas.at(0), 0.0};
        //     normal_params shared_params = {mu, sigma, 0.0};
        //     normal_params ex_1_params = {mu, sigma, 0.0};
        //     normal_params ex_2_params = {mu, sigma, 0.0};
        //     normal_params ex_3_params = {mu, sigma, 0.0};

        //     vector<output_str_3exp> output;

        //     // int ctr = 0;
        //     uint num_shared = 0,
        //          num_ex1 = 0,
        //          num_ex2 = 0,
        //          num_ex3 = 0;
        //     for (uint i = 0; i <= total_num_spec; i++) {
        //         num_shared = i;

        //         num_ex1 = total_num_spec - i;
        //         num_ex2 = total_num_spec - i;
        //         num_ex3 = total_num_spec - i;
        //         // cout << num_shared << " -- " << num_ex1 << " -- " << num_ex2 << " -- " << num_shared + +num_ex1 + num_ex2 << endl;
        //         output.push_back(single_calculation_one_target_one_out_of_three_exp(target_params_1, shared_params, ex_1_params, ex_2_params, ex_3_params, num_shared, num_ex1, num_ex2, num_ex3)); // i + 1

        //         // output.push_back(single_calculation(target_params, shared_params, ex_1_params, ex_2_params,
        //         //                                     num_shared,
        //         //                                     num_ex1,
        //         //                                     num_ex2));
        //     }
        //     writeData_joint_norm_three_exps(output, to_string(total_num_spec) + string("_") + "one_T_low_T_1_3_vsp" + string("_") + to_string(counter));
        //     counter++;
        // }
        // counter = 0;
    }
}

void writeData_joint_norm_three_exps(vector<output_str_3exp> out_str, string experiment_name) {
    string distribution;
    string extension = ".csv";
    string experiment = "joint_sum_normal_three_exp";
    string dir_path = "../output/" + experiment + "/" + experiment_name;

    // making parent directory for all experiments to go into, if it doesnt exist
    // std::system(("mkdir -p ../output/" + experiment + "/" ));

    int systemRet = std::system(("mkdir -p " + dir_path).c_str());
    if (systemRet == -1) {
        // The system method failed
    }
    // // making directory for this specific experiment
    // // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = dir_path + "/results" + extension;
    // // string param_path = path + "/parameters" + extension;

    // // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "target.mu, target.sigma, shared_params.mu, shared_params.sigma, ex1_params.mu, ex1_params.sigma, ex2_params.mu, ex2_params.sigma, ex3_params.mu, ex3_params.sigma, num_shared, num_spec_1, num_spec_2, num_spec_3, target_init, one_exp, two_exp, three_exp \n";

    // for (uint i = 0; i < (int)awae_results.size(); i++) {
    //     outFile << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << "," << h_T.at(i) << "," << h_S.at(i) << "," << h_T_S.at(i) << endl;
    // }
    for (auto &outpt : out_str) {
        outFile << outpt.target.mu << "," << outpt.target.sigma << "," << outpt.shared_params.mu << "," << outpt.shared_params.sigma << "," << outpt.ex1_params.mu << "," << outpt.ex1_params.sigma << "," << outpt.ex2_params.mu << "," << outpt.ex2_params.sigma << "," << outpt.ex3_params.mu << "," << outpt.ex3_params.sigma << "," << outpt.num_shared << "," << outpt.num_spec_1 << "," << outpt.num_spec_2 << "," << outpt.num_spec_3 << "," << outpt.target_init << "," << outpt.one_exp << "," << outpt.two_exp << "," << outpt.three_exp << "\n";
    }

    outFile.close();
    outFile.clear();
}
