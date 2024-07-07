#include "plot.hpp"

string getDist(uint distFlag) {
    string x;
    if (distFlag == 0) {
        x = "u";
        return x;
    } else {
        x = "b";
        return x;
    }
}

// only called once per function we evaluate
// takes all the generated data and plots them into a single, multi-page pdf
void plot(string experiment, int N) {

    // sleeping just in case
    usleep(400);

    string command = "python3 ../py/plot.py ../output_old/" + experiment + "/n" + to_string(N);
    cout << command << endl;
    int systemRet = system(command.c_str());
}

void writeData_Poisson(string experiment, vector<double> awae_results, vector<double> target_init_entropy, vector<int> spectators, double lambda, uint N) {
    string distribution;
    string extension = ".csv";
    string binomial_prob;

    // making parent directory for all experiments to go into, if it doesnt exist
    int systemRet = system(("mkdir -p ../output/" + experiment).c_str());

    // making directory for this specific experiment
    systemRet = system(("mkdir -p ../output/" + experiment + "/" + to_string(N) + '_' + to_string(lambda) ).c_str());
    string path = "../output/" + experiment + "/" + to_string(N) + '_' + to_string(lambda);
    string awae_path = path + "/results" + extension;
    string param_path = path + "/parameters" + extension;

    systemRet = system(("mkdir -p " + path).c_str());

    ofstream outFile(awae_path);
    outFile << "spectators, awae_results, target_init_entropy" << endl;

    for (int i = 0; i < (int)awae_results.size(); i++) {
        outFile << std::fixed << std::setprecision(10) << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << endl;
    }
    outFile.close();
    outFile.clear();
}

void writeData_Uniform(string experiment, vector<double> awae_results, vector<double> target_init_entropy, vector<int> spectators, uint N) {
    // void plot(string experiment, functionNoSpectator &func_ns){
    string distribution;
    string extension = ".csv";
    string binomial_prob;
    int systemRet = system(("mkdir -p ../output/" + experiment).c_str());
     systemRet = system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());

    string path = "../output/" + experiment + "/" + to_string(N) + "/" + experiment;
    string awae_path = path + "/results" + extension;
    string param_path = path + "/parameters" + extension;
     systemRet = system(("mkdir -p " + path).c_str());

    ofstream outFile(awae_path);
    outFile << "spectators, awae_results, target_init_entropy" << endl;
    for (int i = 0; i < (int)awae_results.size(); i++) {
        outFile << spectators.at(i) << "," << awae_results.at(i) << "," << target_init_entropy.at(i) << endl;
    }
    outFile.close();
    outFile.clear();
}

void writeData_lnorm(string experiment, vector<double> awae_results, vector<double> target_init_entropy, vector<int> spectators, vector<double> h_T, vector<double> h_S, vector<double> h_T_S, vector<double> differential_awae, double mu, double sigma, uint N) {
    string distribution;
    string extension = ".csv";
    string binomial_prob;

    // making parent directory for all experiments to go into, if it doesnt exist
    int systemRet = system(("mkdir -p ../output/" + experiment + "/" + to_string(mu) + "_" + to_string(sigma)).c_str());

    // making directory for this specific experiment
    // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = "../output/" + experiment + "/" +to_string(mu) + "_" + to_string(sigma)+"/"+ to_string(N) + extension;
    // string param_path = path + "/parameters" + extension;

    // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "spectators, awae_results, target_init_entropy, h_T, h_S, h_T_S, differential_awae" << endl;

    for (int i = 0; i < (int)awae_results.size(); i++) {
        outFile << spectators.at(i) << ", " << awae_results.at(i) << ", " << target_init_entropy.at(i) << ", " << h_T.at(i) << ", " << h_S.at(i) << ", " << h_T_S.at(i) << ", " << differential_awae.at(i) << endl;
    }
    outFile.close();
    outFile.clear();
}



void writeData_lnorm_differential(string experiment, vector<int> spectators, vector<double> h_T, vector<double> h_S, vector<double> h_T_S, vector<double> differential_awae, double mu, double sigma, uint N) {
    string distribution;
    string extension = ".csv";
    string binomial_prob;

    // making parent directory for all experiments to go into, if it doesnt exist
    int systemRet = system(("mkdir -p ../output/" + experiment + "/" + to_string(mu) + "_" + to_string(sigma)).c_str());

    // making directory for this specific experiment
    // system(("mkdir -p ../output/" + experiment + "/" + to_string(N)).c_str());
    string path = "../output/" + experiment + "/" +to_string(mu) + "_" + to_string(sigma)+"/"+ to_string(N) + extension;
    // string param_path = path + "/parameters" + extension;

    // system(("mkdir -p " + path).c_str());

    ofstream outFile(path);
    outFile << "spectators, h_T, h_S, h_T_S, differential_awae" << endl;

    for (int i = 0; i < (int)spectators.size(); i++) {
        outFile << spectators.at(i) << ", " <<  h_T.at(i) << ", " << h_S.at(i) << ", " << h_T_S.at(i) << ", " << differential_awae.at(i) << endl;
    }
    outFile.close();
    outFile.clear();
}

