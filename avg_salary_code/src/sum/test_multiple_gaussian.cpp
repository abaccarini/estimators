#include "test_multiple_gaussian.hpp"

vector<vector<uint>> t_confs;

long double mv_differential_entropy(mv_gaussian X) {
    switch (X.cov_matrix.cols()) {
    case 2:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);

    case 3:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    default:
        cout << "BASE" << endl;
        return 0.0;
    }
}

long double mv_differential_entropy(N_gaussian X) {
    // int n = X.cov_matrix.cols();
    switch (X.cov_matrix.cols()) {
    case 1:
        return (0.5) * log((2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    case 2:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);

    case 3:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    case 4: {
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    }
    case 5:
        return (0.5) * log((2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * (2.0 * M_PI * M_E) * X.cov_matrix.determinant()) / log(2.0);
    default:
        cout << "BASE" << endl;
        return 0.0;
    }
}

long double single_differential_entropy(uint n, double sigma) {
    return log(sqrt(n * sigma * 2.0 * M_PI * M_E)) / log(2.0);
}

output_str_3exp_full single_calculation(double sigma, uint S_1, uint S_2, uint S_3, uint S_12, uint S_13, uint S_23, uint S_123, uint T, vector<uint> T_flags) {

    long double t_1 = 0.0, t_2 = 0.0, H_X_T = 0.0, H_X_T_O_1 = 0.0, H_X_T_O_12 = 0.0, H_X_T_O_123 = 0.0;

    H_X_T = single_differential_entropy(T, sigma);
    H_X_T_O_1 = single_differential_entropy(T, sigma) + single_differential_entropy(S_1 + S_12 + S_13 + S_123, sigma) - single_differential_entropy(T + S_1 + S_12 + S_13 + S_123, sigma);

    mv_gaussian Spec_12 = {sigma,
                           S_1,
                           S_2,
                           S_12,
                           S_13,
                           S_23,
                           S_123,
                           0,
                           T_flags};

    mv_gaussian O_12 = {sigma,
                        S_1,
                        S_2,
                        S_12,
                        S_13,
                        S_23,
                        S_123,
                        T,
                        T_flags};

    t_1 = H_X_T + mv_differential_entropy(Spec_12);
    t_2 = mv_differential_entropy(O_12);
    H_X_T_O_12 = t_1 - t_2;

    mv_gaussian Spec_123 = {sigma,
                            S_1,
                            S_2,
                            S_3,
                            S_12,
                            S_13,
                            S_23,
                            S_123,
                            0,
                            T_flags};

    mv_gaussian O_123 = {sigma,
                         S_1,
                         S_2,
                         S_3,
                         S_12,
                         S_13,
                         S_23,
                         S_123,
                         T,
                         T_flags};

    // cout << "Spec_123 = \n"
    //      << Spec_123.cov_matrix << endl;
    // cout << "O_123 = \n"
    //      << O_123.cov_matrix << endl;

    t_1 = H_X_T + mv_differential_entropy(Spec_123);
    t_2 = mv_differential_entropy(O_123);
    H_X_T_O_123 = t_1 - t_2;

    // cout << "H_X_T       = " << H_X_T << endl;
    // cout << "H_X_T_O_1   = " << H_X_T_O_1 << endl;
    // cout << "H_X_T_O_12  = " << H_X_T_O_12 << endl;
    // cout << "H_X_T_O_123 = " << H_X_T_O_123 << endl;

    output_str_3exp_full output = {
        sigma,
        S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags.at(0), T_flags.at(1), T_flags.at(2),
        H_X_T,
        H_X_T_O_1,
        H_X_T_O_12,
        H_X_T_O_123};

    return output;
}

void reset_sets(uint &S_1, uint &S_2, uint &S_3, uint &S_12, uint &S_13, uint &S_23, uint &S_123, uint max_per_set) {
    S_1 = max_per_set, S_2 = max_per_set, S_3 = max_per_set;
    S_12 = max_per_set, S_13 = max_per_set, S_23 = max_per_set;
    S_123 = max_per_set;
}

void test_mult_gaussian_main() {

    double sigma = 4;
    // uint total_num_spec = 20;
    // uint max_per_set = 10;
    // uint S_1, S_2, S_3, S_12, S_13, S_23, S_123, T;

    // uint T = 1;
    vector<output_str_3exp_full> output;
    vector<uint> T_flags(3);
    fill(T_flags.begin(), T_flags.end(), 1);
    uint topK = 10;
    brute_force(24, 1, sigma, topK);
    output.clear();

    // for (uint j = 0; j < 3; j++) {
    //     reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);
    //     // total_num_spec = S_1 + S_12 + S_13 + S_123;

    //     total_num_spec = 20;
    //     switch (j) {
    //     case 0:
    //         fill(T_flags.begin(), T_flags.end(), 1);
    //         break;
    //     case 1:
    //         T_flags.at(2) = 0;
    //         break;
    //     case 2:
    //         T_flags.at(1) = 0;
    //         T_flags.at(2) = 0;
    //         break;
    //     }
    //     uint topK = 10;
    //     brute_force(total_num_spec, 1, sigma, topK, output);
    //     // writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "brute_force_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // resetting
    //     output.clear();
    //     // varying S_13 0 -> (max_set + max_set)
    //     for (uint i = 0; i <= (2 * max_per_set); i++) {
    //         S_13 = i;
    //         S_1 = (2 * max_per_set) - i;
    //         S_3 = (2 * max_per_set) - i;
    //         assert(total_num_spec == S_1 + S_12 + S_13 + S_123);
    //         assert(total_num_spec == S_2 + S_12 + S_23 + S_123);
    //         assert(total_num_spec == S_3 + S_23 + S_13 + S_123);
    //         output.push_back(single_calculation(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags));
    //     }
    //     writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "vary_S13_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // resetting
    //     output.clear();
    //     reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);

    //     // varying S_12 0 -> (max_set + max_set)
    //     for (uint i = 0; i <= (2 * max_per_set); i++) {
    //         S_12 = i;
    //         S_1 = (2 * max_per_set) - i;
    //         S_2 = (2 * max_per_set) - i;
    //         assert(total_num_spec == S_1 + S_12 + S_13 + S_123);
    //         assert(total_num_spec == S_2 + S_12 + S_23 + S_123);
    //         assert(total_num_spec == S_3 + S_23 + S_13 + S_123);

    //         output.push_back(single_calculation(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags));
    //     }
    //     writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "vary_S12_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // resetting
    //     output.clear();
    //     reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);
    //     S_123 = 8, S_13 = 8, S_23 = 8;
    //     S_3 = 16;
    //     // varying S_12 0 -> (max_set + max_set), 40% overlap for exp3

    //     for (uint i = 0; i <= (2 * (max_per_set) + 4); i++) {
    //         S_12 = i;
    //         S_1 = (2 * (max_per_set) + 4) - i;
    //         S_2 = (2 * (max_per_set) + 4) - i;
    //         // cout<<S_12<<", "<<S_1<<", "<<S_2<<", "<<S_123<<"\n";
    //         // printf("S_1(%u) + S_12(%u) + S_13(%u) + S_123(%u) = %u\n",S_1 , S_12 , S_13 , S_123, S_1 + S_12 + S_13 + S_123);
    //         assert(total_num_spec == S_1 + S_12 + S_13 + S_123);

    //         // printf("S_2(%u) + S_12(%u) + S_23(%u) + S_123(%u) = %u\n",S_2 , S_12 , S_23 , S_123, S_2 + S_12 + S_23 + S_123);
    //         assert(total_num_spec == S_2 + S_12 + S_23 + S_123);

    //         // printf("S_3(%u) + S_23(%u) + S_13(%u) + S_123(%u) = %u\n",S_3 , S_23 , S_13 , S_123, S_3 + S_23 + S_13 + S_123);
    //         assert(total_num_spec == S_3 + S_23 + S_13 + S_123);

    //         output.push_back(single_calculation(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags));
    //     }
    //     writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "vary_S12_40_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // resetting
    //     output.clear();
    //     reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);

    //     S_123 = 12, S_13 = 12, S_23 = 12;
    //     S_3 = 4;
    //     // varying S_12 0 -> (max_set + max_set), 40% overlap for exp3

    //     for (uint i = 0; i <= (2 * (max_per_set)-4); i++) {
    //         S_12 = i;
    //         S_1 = (2 * (max_per_set)-4) - i;
    //         S_2 = (2 * (max_per_set)-4) - i;
    //         // cout<<S_12<<", "<<S_1<<", "<<S_2<<", "<<S_123<<"\n";
    //         // printf("S_1(%u) + S_12(%u) + S_13(%u) + S_123(%u) = %u\n",S_1 , S_12 , S_13 , S_123, S_1 + S_12 + S_13 + S_123);
    //         assert(total_num_spec == S_1 + S_12 + S_13 + S_123);

    //         // printf("S_2(%u) + S_12(%u) + S_23(%u) + S_123(%u) = %u\n",S_2 , S_12 , S_23 , S_123, S_2 + S_12 + S_23 + S_123);
    //         assert(total_num_spec == S_2 + S_12 + S_23 + S_123);

    //         // printf("S_3(%u) + S_23(%u) + S_13(%u) + S_123(%u) = %u\n",S_3 , S_23 , S_13 , S_123, S_3 + S_23 + S_13 + S_123);
    //         assert(total_num_spec == S_3 + S_23 + S_13 + S_123);

    //         output.push_back(single_calculation(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags));
    //     }
    //     writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "vary_S12_60_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // resetting
    //     output.clear();
    //     reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);

    //     // varying S_23 0 -> (max_set + max_set)
    //     for (uint i = 0; i <= (2 * max_per_set); i++) {
    //         S_23 = i;
    //         S_2 = (2 * max_per_set) - i;
    //         S_3 = (2 * max_per_set) - i;
    //         assert(total_num_spec == S_1 + S_12 + S_13 + S_123);
    //         assert(total_num_spec == S_2 + S_12 + S_23 + S_123);
    //         assert(total_num_spec == S_3 + S_23 + S_13 + S_123);

    //         output.push_back(single_calculation(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags));
    //     }
    //     writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "vary_S23_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // resetting
    //     output.clear();
    //     reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);

    //     // varying S_123 0 -> (max_set + max_set)
    //     for (uint i = 0; i <= (2 * max_per_set); i++) {
    //         S_123 = i;
    //         S_1 = (2 * max_per_set) - i;
    //         S_2 = (2 * max_per_set) - i;
    //         S_3 = (2 * max_per_set) - i;
    //         assert(total_num_spec == S_1 + S_12 + S_13 + S_123);
    //         assert(total_num_spec == S_2 + S_12 + S_23 + S_123);
    //         assert(total_num_spec == S_3 + S_23 + S_13 + S_123);

    //         output.push_back(single_calculation(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags));
    //     }
    //     writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "vary_S123_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // resetting
    //     output.clear();
    //     reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);

    //     // uint temp_total = 36;
    //     // // varying S_123 0 -> (max_set + max_set)
    //     // for (uint i = 0; i <= 6; i++) {
    //     //     S_1 = 0, S_2 = 0, S_3 = 0;

    //     //     S_12 = temp_total/2 - 3*i;
    //     //     S_13 = temp_total/2 - 3*i;
    //     //     S_23 = temp_total/2 - 3*i;
    //     //     S_123 = i*6;

    //     //     assert(temp_total == S_1 + S_12 + S_13 + S_123);
    //     //     assert(temp_total == S_2 + S_12 + S_23 + S_123);
    //     //     assert(temp_total == S_3 + S_23 + S_13 + S_123);

    //     //     output.push_back(single_calculation(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, T_flags));
    //     // }
    //     // writeData_joint_normal_3xp(output, to_string(temp_total) + string("_") + "vary_S123_S12_S13_S23_" + to_string(T_flags.at(0)) + to_string(T_flags.at(1)) + to_string(T_flags.at(2)));
    //     // // resetting
    //     // output.clear();
    //     // reset_sets(S_1, S_2, S_3, S_12, S_13, S_23, S_123, max_per_set);
    // }
}

void brute_force(uint total_num_spec, uint num_targets, double sigma, uint topK) {

    uint S_1, S_2, S_3;
    output_str_3exp_full temp_out;
    vector<uint> T_flags(3);
    fill(T_flags.begin(), T_flags.end(), 1);

    vector<output_str_3exp_brute_force> output;

    for (uint S_12 = 0; S_12 <= (total_num_spec); S_12++) {
        for (uint S_23 = 0; S_23 <= (total_num_spec - S_12); S_23++) {
            for (uint S_13 = 0; S_13 <= (total_num_spec - S_23 - S_12); S_13++) {
                for (uint S_123 = 0; S_123 <= (total_num_spec - S_12 - S_23 - S_13); S_123++) {

                    S_1 = total_num_spec - S_12 - S_13 - S_123;
                    S_2 = total_num_spec - S_12 - S_23 - S_123;
                    S_3 = total_num_spec - S_23 - S_13 - S_123;
                    assert(total_num_spec == S_1 + S_12 + S_13 + S_123);
                    assert(total_num_spec == S_2 + S_12 + S_23 + S_123);
                    assert(total_num_spec == S_3 + S_23 + S_13 + S_123);

                    output.push_back(single_calculation_brute_force(sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, 1));

                    // if (output.size() <= topK) {
                    //     output.push_back(temp_out);
                    // } else {
                    //     if (temp_out.H_X_T_O_123 > output.back().H_X_T_O_123) {

                    //         output.push_back(temp_out);
                    //         // sorting in DESCENDING ORDER
                    //         sort(output.begin(), output.end(), compareBy_H_T_O_123);

                    //         // deleteing new smallest value (WHICH IS AT THE END OF THE VECTOR)
                    //         output.pop_back();
                    //     }
                    //     // otherwise, don't bother adding/sorting, since it has a larger loss than the configurations already present in output
                    // }
                }
            }
        }
    }

    writeData_joint_normal_3xp(output, to_string(total_num_spec) + string("_") + "brute_force_all");
}
// SORTS IN DESCENDING ORDER
bool compareBy_H_T_O_123(const output_str_3exp_full &a, const output_str_3exp_full &b) {
    return a.H_X_T_O_123 > b.H_X_T_O_123;
}

output_str_3exp_brute_force single_calculation_brute_force(double sigma, uint S_1, uint S_2, uint S_3, uint S_12, uint S_13, uint S_23, uint S_123, uint T) {

    long double t_1 = 0.0, t_2 = 0.0, H_X_T = 0.0, H_X_T_O_1 = 0.0, H_X_T_O_12_t11 = 0.0, H_X_T_O_12_t10 = 0.0, H_X_T_O_123_t111 = 0.0, H_X_T_O_123_t110 = 0.0, H_X_T_O_123_t101 = 0.0, H_X_T_O_123_t011 = 0.0, H_X_T_O_123_t100 = 0.0, H_X_T_O_123_t010 = 0.0, H_X_T_O_123_t001 = 0.0;

    H_X_T = single_differential_entropy(T, sigma);
    H_X_T_O_1 = single_differential_entropy(T, sigma) + single_differential_entropy(S_1 + S_12 + S_13 + S_123, sigma) - single_differential_entropy(T + S_1 + S_12 + S_13 + S_123, sigma);

    mv_gaussian Spec_12_t11 = {sigma,
                               S_1,
                               S_2,
                               S_12,
                               S_13,
                               S_23,
                               S_123,
                               0,
                               {1, 1}};

    mv_gaussian O_12_t11 = {sigma,
                            S_1,
                            S_2,
                            S_12,
                            S_13,
                            S_23,
                            S_123,
                            T,
                            {1, 1}};

    mv_gaussian Spec_12_t10 = {sigma,
                               S_1,
                               S_2,
                               S_12,
                               S_13,
                               S_23,
                               S_123,
                               0,
                               {1, 0}};

    mv_gaussian O_12_t10 = {sigma,
                            S_1,
                            S_2,
                            S_12,
                            S_13,
                            S_23,
                            S_123,
                            T,
                            {1, 0}};

    t_1 = H_X_T + mv_differential_entropy(Spec_12_t11);
    t_2 = mv_differential_entropy(O_12_t11);
    H_X_T_O_12_t11 = t_1 - t_2;

    t_1 = H_X_T + mv_differential_entropy(Spec_12_t10);
    t_2 = mv_differential_entropy(O_12_t10);
    H_X_T_O_12_t10 = t_1 - t_2;

    mv_gaussian Spec_123_t111 = {sigma,
                                 S_1,
                                 S_2,
                                 S_3,
                                 S_12,
                                 S_13,
                                 S_23,
                                 S_123,
                                 0,
                                 {1, 1, 1}};

    mv_gaussian O_123_t111 = {sigma,
                              S_1,
                              S_2,
                              S_3,
                              S_12,
                              S_13,
                              S_23,
                              S_123,
                              T,
                              {1, 1, 1}};

    mv_gaussian Spec_123_t110 = {sigma,
                                 S_1,
                                 S_2,
                                 S_3,
                                 S_12,
                                 S_13,
                                 S_23,
                                 S_123,
                                 0,
                                 {1, 1, 0}};

    mv_gaussian Spec_123_t101 = {sigma,
                                 S_1,
                                 S_2,
                                 S_3,
                                 S_12,
                                 S_13,
                                 S_23,
                                 S_123,
                                 0,
                                 {1, 0, 1}};

    mv_gaussian Spec_123_t011 = {sigma,
                                 S_1,
                                 S_2,
                                 S_3,
                                 S_12,
                                 S_13,
                                 S_23,
                                 S_123,
                                 0,
                                 {0, 1, 1}};

    mv_gaussian O_123_t110 = {sigma,
                              S_1,
                              S_2,
                              S_3,
                              S_12,
                              S_13,
                              S_23,
                              S_123,
                              T,
                              {1, 1, 0}};
    mv_gaussian O_123_t101 = {sigma,
                              S_1,
                              S_2,
                              S_3,
                              S_12,
                              S_13,
                              S_23,
                              S_123,
                              T,
                              {1, 0, 1}};
    mv_gaussian O_123_t011 = {sigma,
                              S_1,
                              S_2,
                              S_3,
                              S_12,
                              S_13,
                              S_23,
                              S_123,
                              T,
                              {0, 1, 1}};

    mv_gaussian Spec_123_t100 = {sigma,
                                 S_1,
                                 S_2,
                                 S_3,
                                 S_12,
                                 S_13,
                                 S_23,
                                 S_123,
                                 0,
                                 {1, 0, 0}};

    mv_gaussian O_123_t100 = {sigma,
                              S_1,
                              S_2,
                              S_3,
                              S_12,
                              S_13,
                              S_23,
                              S_123,
                              T,
                              {1, 0, 0}};

    mv_gaussian Spec_123_t010 = {sigma,
                                 S_1,
                                 S_2,
                                 S_3,
                                 S_12,
                                 S_13,
                                 S_23,
                                 S_123,
                                 0,
                                 {0, 1, 0}};

    mv_gaussian O_123_t010 = {sigma,
                              S_1,
                              S_2,
                              S_3,
                              S_12,
                              S_13,
                              S_23,
                              S_123,
                              T,
                              {0, 1, 0}};

    mv_gaussian Spec_123_t001 = {sigma,
                                 S_1,
                                 S_2,
                                 S_3,
                                 S_12,
                                 S_13,
                                 S_23,
                                 S_123,
                                 0,
                                 {0, 0, 1}};

    mv_gaussian O_123_t001 = {sigma,
                              S_1,
                              S_2,
                              S_3,
                              S_12,
                              S_13,
                              S_23,
                              S_123,
                              T,
                              {0, 0, 1}};

    t_1 = H_X_T + mv_differential_entropy(Spec_123_t111);
    t_2 = mv_differential_entropy(O_123_t111);
    H_X_T_O_123_t111 = t_1 - t_2;

    t_1 = H_X_T + mv_differential_entropy(Spec_123_t110);
    t_2 = mv_differential_entropy(O_123_t110);
    H_X_T_O_123_t110 = t_1 - t_2;

    t_1 = H_X_T + mv_differential_entropy(Spec_123_t101);
    t_2 = mv_differential_entropy(O_123_t101);
    H_X_T_O_123_t101 = t_1 - t_2;

    t_1 = H_X_T + mv_differential_entropy(Spec_123_t011);
    t_2 = mv_differential_entropy(O_123_t011);
    H_X_T_O_123_t011 = t_1 - t_2;

    t_1 = H_X_T + mv_differential_entropy(Spec_123_t100);
    t_2 = mv_differential_entropy(O_123_t100);
    H_X_T_O_123_t100 = t_1 - t_2;

    t_1 = H_X_T + mv_differential_entropy(Spec_123_t010);
    t_2 = mv_differential_entropy(O_123_t010);
    H_X_T_O_123_t010 = t_1 - t_2;

    t_1 = H_X_T + mv_differential_entropy(Spec_123_t001);
    t_2 = mv_differential_entropy(O_123_t001);
    H_X_T_O_123_t001 = t_1 - t_2;

    output_str_3exp_brute_force output = {
        sigma,
        S_1, S_2, S_3, S_12, S_13, S_23, S_123, T,
        H_X_T,
        H_X_T_O_1,
        H_X_T_O_12_t11,
        H_X_T_O_12_t10,
        H_X_T_O_123_t111,
        H_X_T_O_123_t110,
        H_X_T_O_123_t101,
        H_X_T_O_123_t011,
        H_X_T_O_123_t100,
        H_X_T_O_123_t010,
        H_X_T_O_123_t001};

    return output;
}

void writeData_joint_normal_3xp(vector<output_str_3exp_full> out_str, string experiment_name) {
    string distribution;
    string extension = ".csv";
    string experiment = "joint_normal_3xp_full";
    string dir_path = "../output/" + experiment + "/" + experiment_name;
    // making parent directory for all experiments to go into, if it doesnt exist
    // system(("mkdir -p ../output/" + experiment + "/" ));

    int systemRet = system(("mkdir -p " + dir_path).c_str());
    if (systemRet == -1) {
        // The system method failed
    }
    // // making directory for this specific experiment
    // // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = dir_path + "/results" + extension;
    // // string param_path = path + "/parameters" + extension;
    // cout<<path<<endl;
    // // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, t_1, t_2, t_3, H_X_T, H_X_T_O_1, H_X_T_O_12, H_X_T_O_123 \n";
    // for (uint i = 0; i < (int)awae_results.size(); i++) {
    //     outFile << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << "," << h_T.at(i) << "," << h_S.at(i) << "," << h_T_S.at(i) << endl;
    // }
    outFile.precision(10);
    for (auto &outpt : out_str) {
        outFile << outpt.sigma << "," << outpt.S_1 << "," << outpt.S_2 << "," << outpt.S_3 << "," << outpt.S_12 << "," << outpt.S_13 << "," << outpt.S_23 << "," << outpt.S_123 << "," << outpt.T << "," << outpt.t_1 << "," << outpt.t_2 << "," << outpt.t_3 << "," << outpt.H_X_T << "," << outpt.H_X_T_O_1 << "," << outpt.H_X_T_O_12 << "," << outpt.H_X_T_O_123 << "\n";
    }

    outFile.close();
    outFile.clear();
}
void writeData_joint_normal_3xp(vector<output_str_3exp_brute_force> out_str, string experiment_name) {

    string distribution;
    string extension = ".csv";
    string experiment = "joint_normal_3xp_full";
    string dir_path = "../output/" + experiment + "/" + experiment_name;
    int systemRet = std::system(("mkdir -p " + dir_path).c_str());
    if (systemRet == -1) {
        // The system method failed
    }
    // // making directory for this specific experiment
    // // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = dir_path + "/results" + extension;
    // // string param_path = path + "/parameters" + extension;
    // cout<<path<<endl;
    // // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "sigma, S_1, S_2, S_3, S_12, S_13, S_23, S_123, T, H_X_T, H_X_T_O_1, H_X_T_O_12_t11, H_X_T_O_12_t10, H_X_T_O_123_t111, H_X_T_O_123_t110, H_X_T_O_123_t101, H_X_T_O_123_t011, H_X_T_O_123_t100, H_X_T_O_123_t010, H_X_T_O_123_t001 \n";

    // for (uint i = 0; i < (int)awae_results.size(); i++) {
    //     outFile << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << "," << h_T.at(i) << "," << h_S.at(i) << "," << h_T_S.at(i) << endl;
    // }
    outFile.precision(10);
    for (auto &outpt : out_str) {
        outFile << outpt.sigma << "," << outpt.S_1 << "," << outpt.S_2 << "," << outpt.S_3 << "," << outpt.S_12 << "," << outpt.S_13 << "," << outpt.S_23 << "," << outpt.S_123 << "," << outpt.T << "," << outpt.H_X_T << "," << outpt.H_X_T_O_1 << "," << outpt.H_X_T_O_12_t11 << "," << outpt.H_X_T_O_12_t10 << "," << outpt.H_X_T_O_123_t111 << "," << outpt.H_X_T_O_123_t110 << "," << outpt.H_X_T_O_123_t101 << "," << outpt.H_X_T_O_123_t011 << "," << outpt.H_X_T_O_123_t100 << "," << outpt.H_X_T_O_123_t010 << "," << outpt.H_X_T_O_123_t001 << "\n";
    }

    outFile.close();
    outFile.clear();
}

void print_vector(const std::vector<int> &array) {
    for (const int item : array) {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}

vector<string> generate_spec_combo(uint N) {
    std::vector<int> int_vec(N);           // vector with N+1 ints.
    iota(begin(int_vec), end(int_vec), 1); // Fill with 1, ..., N.
    vector<string> spec_power_set(pow(2, N));
    generate_power_set(int_vec, N, spec_power_set);
    return spec_power_set;
}

void generate_power_set(vector<int> set, int n, vector<string> &output) {
    bool *contain = new bool[n]{0};
    // Empty subset
    uint ctr = 0;
    for (int i = 0; i < n; i++) {
        contain[i] = 1;
        // All permutation
        do {
            for (int j = 0; j < n; j++)
                if (contain[j]) {
                    output.at(ctr).append(to_string(set[j]));
                }
            ctr++;
        } while (prev_permutation(contain, contain + n));
    }
    output.pop_back(); // last element is empty string (empty set)
}

vector<vector<uint>> target_comb(int N) {
    vector<vector<uint>> result(pow(2, N) - 1, vector<uint>(N, 0));
    for (uint i = 1; i < (pow(2, N)); i++) {
        for (size_t j = 0; j < N; j++) {
            result.at(i - 1).at(j) = GET_BIT(i, j);
        }
    }
    return result;
}

struct AddValues
{
  template<class Value, class Pair> 
  Value operator()(Value value, const Pair& pair) const
  {
    return value + pair.second;
  }
};

//adds values from spec_map(start) to spec_map(end - 1) (inclusive)
int acc(vector<pair<string, int>> &spec_map,int start, int end){
    int accum = 0;
    for (size_t i = start; i < end; i++) {
        accum += spec_map.at(i).second;
    }
    return accum;
}


int counter = 0;
void test_N_exp_main() {

    double sigma = 4.0;
    // uint Delta = 10;
    uint N = 1;
    for (uint Delta = 1; Delta < 51; Delta++) {
       cout<<"Delta = "<<Delta<<endl; 
    // for (uint Delta = 25; Delta < 26; Delta++) {

        vector<string> spec_power_set = generate_spec_combo(N); // length of 2^N - 1 (removed empty set)
        t_confs = target_comb(N);
        // vector<vector<uint>> t_confs = target_comb(N);

        vector<pair<string, int>> spec_map;
        for (auto const &x : spec_power_set) {
            spec_map.push_back(make_pair(x, 0));
        }
        // for (auto const &x : spec_map) {
        //     cout<<x.first<<", "<<x.second<<endl;
        // }
        // spec_map.at(0).second = 4;
        // spec_map.at(1).second = 1;
        // spec_map.at(2).second = 2;
        // cout<<acc(spec_map,0,2)<<endl;

        uint top_N = 1;
        vector<N_exp_data> out_data;
        doRecursion(spec_map.size() - 1, Delta, 0, N, sigma, spec_map, top_N, out_data);
        // evaluate_4_exp(Delta, N, sigma, spec_map, top_N, out_data); //use for N = 4

        // sorting the results
        std::sort_heap(out_data.begin(), out_data.end(), compareByCondEnt);
        wirteData_N_exp(out_data, spec_map, sigma, N, Delta);
        cout << spec_map.size() << endl;
        cout << "counter = " << counter << endl;
    }
}

// 417,225,900 options for N = 4, delta = 24
// starting index, the end of the vector
// each recursion, we move towards the front of the vector
// once we reach the point were its single spectator vars (e.g. s1, s2,...), we're done
// and can explicitly define the remaining variables
// minis_term just set to zero
void doRecursion(int index, int Delta, int minus_term, int N, double sigma, vector<pair<string, int>> &spec_map, uint top_N, vector<N_exp_data> &out_data) {

    if (index == (N - 1)) {
        // this is where we set the s1, s2, s3, ..., si, ...., sN variables, since all others have been set at this point/
        // can be done by iterating through the list to see i is in string of mapping (key), if yes, subtract the value stored there
        for (size_t i = 0; i < N; i++) {
            spec_map.at(i).second = Delta;
            for (size_t j = N; j < spec_map.size(); j++) {
                // if (spec_map.at(j).first.contains(to_string(i + 1))) {
                if (spec_map.at(j).first.find(to_string(i + 1)) != std::string::npos) {
                    spec_map.at(i).second -= spec_map.at(j).second;
                }
            }
        }
        // for (size_t i = 0; i < N; i++) {
        //     uint temp = 0;
        //     for (size_t j = 0; j < spec_map.size(); j++) {
        //         if (spec_map.at(j).first.contains(to_string(i + 1))) {
        //             temp += spec_map.at(j).second;
        //         }
        //     }
        //     assert(Delta == temp);
        // }
        // do actual computation here
        counter += 1;
        // cout<<"\r counter = "<<counter;
        single_calculation(sigma, N, Delta, spec_map, top_N, out_data);

        return;
    }

    for (size_t i = 0; i <= (Delta - minus_term); i++) {
        spec_map.at(index).second = i;
        doRecursion(index - 1, Delta, minus_term + i, N, sigma, spec_map, top_N, out_data);
    }
}
void single_calculation(double sigma, uint N, uint Delta, vector<pair<string, int>> &spec_map, uint top_N, vector<N_exp_data> &out_data) {
    uint T = 1;
    double H_X_T = single_differential_entropy(T, sigma);
    double t_1, t_2, H_cond_t_vec;
    // stream << sigma;
    // for (auto &[key, value] : spec_map) {
    //     stream << "," << value;
    // }
    // stream.precision(10);
    // stream << "," << H_X_T;

    vector<int> spec_values(spec_map.size(), 0);
    // extracting all of the spectator values
    for (size_t i = 0; i < spec_map.size(); i++) {
        spec_values.at(i) = spec_map.at(i).second;
    }

    // vector<double> entropies(t_confs.size(), 0);
    // for (auto const &t_vec : t_confs) {
    double min = 0.0;
    bool flag = 0;
    for (size_t i = 0; i < t_confs.size(); i++) {
        N_gaussian Spec_N = {sigma, N, Delta, 0, t_confs.at(i), spec_map};
        N_gaussian O_N = {sigma, N, Delta, 1, t_confs.at(i), spec_map};
        t_1 = H_X_T + mv_differential_entropy(Spec_N);
        t_2 = mv_differential_entropy(O_N);
        H_cond_t_vec = t_1 - t_2;
        if (isfinite(H_cond_t_vec)) {
            if (flag) { // isn't the first value
                if (H_cond_t_vec < min) {
                    min = H_cond_t_vec;
                }
            } else { // first value
                min = H_cond_t_vec;
                flag = 1;
            }
        }
    }
    // cout<<min<<endl;
    // if the flag was never flipped to one, that means none of the entropies were finite
    // and we skip this block entirely
    if (flag) {
        // if the output data size is less than the top_N we want, just insert
        if (out_data.size() < top_N) {
            out_data.push_back({spec_values, min});
            if (out_data.size() == top_N) {
                std::make_heap(out_data.begin(), out_data.end(), compareByCondEnt);
            }
        } else {
            // optionally inserting the new value into the heap
            if (min > out_data[0].min_cond_entropy) {
                pop_heap(out_data.begin(), out_data.end(), compareByCondEnt);
                out_data.back() = {spec_values, min};
                push_heap(out_data.begin(), out_data.end(), compareByCondEnt);
            }
        }
    }
}
bool compareByCondEnt(const N_exp_data &a, const N_exp_data &b) {
    return a.min_cond_entropy > b.min_cond_entropy;
}

void wirteData_N_exp(vector<N_exp_data> out_data, vector<pair<string, int>> &spec_map, double sigma, uint N, uint Delta) {
    string experiment_name = to_string(N) + "_" + to_string(Delta) + "_all";
    string distribution;
    string experiment = "N_experiments";
    string dir_path = "../output/" + experiment + "/" + experiment_name;
    // string dir_path = "/Users/alessandrobaccarini/output/" + experiment + "/" + experiment_name; // storing locally

    double H_X_T = single_differential_entropy(1, sigma);

    int systemRet = std::system(("mkdir -p " + dir_path).c_str());
    if (systemRet == -1) {
        printf("directory creation failed\n");
        return;
    }

    string path = dir_path + "/results.csv";
    ofstream outFile(path);
    // column names
    outFile << "sigma";
    for (auto &[key, value] : spec_map) {
        outFile << ",s" << key;
    }
    outFile << ",H_X_T,min_cond_entropy\n";

    // writing actual data
    for (auto &outpt : out_data) {
        outFile << sigma;
        for (auto &v : outpt.spec_values) {
            outFile << "," << v;
        }
        outFile.precision(10);
        outFile << "," << H_X_T << "," << outpt.min_cond_entropy << "\n";
    }
}

void doOpNoRecursion(int index, int Delta, int minus_term, int N, double sigma, vector<pair<string, int>> &spec_map, ofstream &stream) {

    vector<int> indices(spec_map.size() - N + 1, 0);          // if "n" is not known before hand, then this array will need to be created dynamically.
    vector<int> upper_bounds(spec_map.size() - N + 1, Delta); // if "n" is not known before hand, then this array will need to be created dynamically.
    int p = 0;
    while (indices.back() == 0) {

        // main computation
        for (size_t i = 0; i < N; i++) {
            spec_map.at(i).second = Delta;
            for (size_t j = N; j < spec_map.size(); j++) {
                // if (spec_map.at(j).first.contains(to_string(i + 1))) {
                if (spec_map.at(j).first.find(to_string(i + 1)) != std::string::npos) {
                    spec_map.at(i).second -= spec_map.at(j).second;
                }
            }
        }
        counter += 1;
        // single_calculation(sigma, N, Delta, spec_map, stream);
        indices[0]++;

        while (indices[p] == upper_bounds.at(p)) { //(or "MAX[p]" if each "for" loop is different. Note that from an English point of view, this is more like "if(indices[p]==MAX". (Initially indices[0]) If this is true, then indices[p] is reset to 0, and indices[p+1] is incremented.
            indices[p] = 0;
            indices[++p]++; // increase p by 1, and increase the next (p+1)th index
            upper_bounds.at(++p)--;
            if (indices[p] != upper_bounds.at(p))
                p = 0; // Alternatively, "p=0" can be inserted above (currently commented-out). This one's more efficient though, since it only resets p when it actually needs to be reset!
        }
    }

    // if (index == (N - 1)) {
    //     // this is where we set the s1, s2, s3, ..., si, ...., sN variables, since all others have been set at this point/
    //     // can be done by iterating through the list to see i is in string of mapping (key), if yes, subtract the value stored there
    //     // for (size_t i = 0; i < N; i++) {
    //     //     uint temp = 0;
    //     //     for (size_t j = 0; j < spec_map.size(); j++) {
    //     //         if (spec_map.at(j).first.contains(to_string(i + 1))) {
    //     //             temp += spec_map.at(j).second;
    //     //         }
    //     //     }
    //     //     assert(Delta == temp);
    //     // }
    //     // do actual computation here

    //     return;
    // }

    // for (size_t i = 0; i <= (Delta - minus_term); i++) {
    //     spec_map.at(index).second = i;
    //     doRecursion(index - 1, Delta, minus_term + i, N, sigma, spec_map, stream);
    // }
}

// void doRecursion(int index, int Delta, int minus_term, int N, double sigma, vector<pair<string, int>> &spec_map, uint top_N, ) 

void evaluate_4_exp(int Delta, int N, double sigma, vector<pair<string, int>> &svar, uint top_N,vector<N_exp_data> &out_data) {
// // spec_map.at(index).second

// for (spec_map.at(index).second = 0; i < count; i++)
// {
//     /* code */
// }


for (svar.at(N+0).second = 0;    svar.at(N+0).second <= Delta - 0; svar.at(N+0).second++){
for (svar.at(N+1).second = 0;    svar.at(N+1).second <= Delta - acc(svar,N,1); svar.at(N+1).second++){
for (svar.at(N+2).second = 0;    svar.at(N+2).second <= Delta - acc(svar,N,2); svar.at(N+2).second++){
for (svar.at(N+3).second = 0;    svar.at(N+3).second <= Delta - acc(svar,N,3); svar.at(N+3).second++){
for (svar.at(N+4).second = 0;    svar.at(N+4).second <= Delta - acc(svar,N,4); svar.at(N+4).second++){
for (svar.at(N+5).second = 0;    svar.at(N+5).second <= Delta - acc(svar,N,5); svar.at(N+5).second++){
for (svar.at(N+6).second = 0;    svar.at(N+6).second <= Delta - acc(svar,N,6); svar.at(N+6).second++){
for (svar.at(N+7).second = 0;    svar.at(N+7).second <= Delta - acc(svar,N,7); svar.at(N+7).second++){
for (svar.at(N+8).second = 0;    svar.at(N+8).second <= Delta - acc(svar,N,8); svar.at(N+8).second++){
for (svar.at(N+9).second = 0;    svar.at(N+9).second <= Delta - acc(svar,N,9); svar.at(N+9).second++){
for (svar.at(N+10).second = 0;  svar.at(N+10).second <= Delta - acc(svar,N,10); svar.at(N+10).second++){


    for (size_t i = 0; i < N; i++) {
            svar.at(i).second = Delta;
            for (size_t j = N; j < svar.size(); j++) {
                // if (svar.at(j).first.contains(to_string(i + 1))) {
                if (svar.at(j).first.find(to_string(i + 1)) != std::string::npos) {
                    svar.at(i).second -= svar.at(j).second;
                }
            }
        }
        single_calculation(sigma, N, Delta, svar, top_N, out_data);


}
}
}
}
}
}
}
}
}
}
}

}
