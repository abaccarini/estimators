#ifndef _DATA_IO_HPP_
#define _DATA_IO_HPP_

#include <string>
#include <vector>
#include <fstream>
#include <json/writer.h>

using std::vector;
using std::string;


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
    string path = dir_path + "/results" + extension;

    std::ofstream outFile(path);
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


#endif // _DATA_IO_HPP_