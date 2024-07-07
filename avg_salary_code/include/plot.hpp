#ifndef PLOT_H
#define PLOT_H

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <map>

// #include "joint.hpp"

using namespace std;

string getDist(uint);
void plot(string, int);
void plot(string, int, vector<int>);

typedef struct output_max_struct {
    int num_T;
    int num_A;
    int num_S;
    int N;
    std::map<int,double> awae_map;
    std::map<int,double> twae_map;
} output_max;

template <typename T>
void writeData(string experiment, T &func_ns) {
    // void plot(string experiment, functionNoSpectator &func_ns){
    string distribution;
    string extension = ".csv";
    string binomial_prob;
    // string num_participants;

    // making parent directory for all experiments to go into, if it doesnt exist
    int systemRet = system(("mkdir -p ../output_old/" + experiment).c_str());
    if (systemRet == -1) {
        // The system method failed
    }

    // making directory for this specific experiment
    systemRet = system(("mkdir -p ../output_old/" + experiment + "/n" + to_string(func_ns.N)).c_str());

    // checking if we have any spectators present
    // if no, don't include in output filename
    int check = 0;
    if (func_ns.numSpectators == 0) {
        check = 1;
    }

    // notation: targetDist, targetProb, attackerDist, attackerProb, spectatorDist, spectatorProb
    for (int i = 0; i < (int)func_ns.distFlags.size() - check; i++) {
        distribution += getDist(func_ns.distFlags.at(i));
        binomial_prob += to_string((int)(func_ns.distProbabilities[i] * 10));
    }

    string path = "../output_old/" + experiment + "/n" + to_string(func_ns.N) + "/" + experiment + "_" + distribution + "_" + binomial_prob + "_" + to_string(func_ns.attacker_range.at(0)) + to_string(func_ns.attacker_range.at(1)) + "_t" + to_string(func_ns.numTargets) + "_a" + to_string(func_ns.numAttackers) + "_s" + to_string(func_ns.numSpectators);

    cout << path << endl;

    string awae_path = path + "/awae" + extension;
    string twae_path = path + "/twae" + extension;
    string param_path = path + "/parameters" + extension;

    systemRet = system(("mkdir -p " + path).c_str());

    ofstream outFile(awae_path);

    for (int i = 0; i < (int)func_ns.x_A_combinations.size(); i++) {
        for (int j = 0; j < (int)func_ns.x_A_combinations.at(i).size(); j++) {
            outFile << func_ns.x_A_combinations.at(i).at(j) << ",";
        }
        outFile << func_ns.awae_values.at(i) << endl;
    }
    outFile.close();
    outFile.clear();
    outFile.open(twae_path);

    for (int i = 0; i < (int)func_ns.x_T_combinations.size(); i++) {
        for (int j = 0; j < (int)func_ns.x_T_combinations.at(i).size(); j++) {
            outFile << func_ns.x_T_combinations.at(i).at(j) << ",";
        }
        outFile << func_ns.twae_values.at(i) << endl;
    }

    outFile.close();

    outFile.clear();
    outFile.open(param_path);

    // because fuck learning JSON, right?
    outFile << "func_ns.N," << func_ns.N << endl;
    outFile << "func_ns.numTargets," << func_ns.numTargets << endl;
    outFile << "func_ns.numAttackers," << func_ns.numAttackers << endl;
    outFile << "func_ns.numSpectators," << func_ns.numSpectators << endl;
    outFile << "func_ns.attacker_range[0]," << func_ns.attacker_range[0] << endl;
    outFile << "func_ns.attacker_range[1]," << func_ns.attacker_range[1] << endl;
    outFile << "TargetDist," << func_ns.distFlags.at(0) << endl;
    outFile << "AttackerDist," << func_ns.distFlags.at(1) << endl;
    outFile << "SpectatorDist," << func_ns.distFlags.at(2) << endl;
    outFile << "TargetP," << func_ns.distProbabilities.at(0) << endl;
    outFile << "AttackerP," << func_ns.distProbabilities.at(1) << endl;
    outFile << "SpectatorP," << func_ns.distProbabilities.at(2) << endl;

    outFile.close();

    // usleep(1000);

    // string command = "python3 ../py/plot.py " + awae_path + " " + twae_path;
    // cout << command << endl;
    // system(command.c_str());
}

void writeData_Poisson(string experiment, vector<double> awae_results, vector<double> target_init_entropy, vector<int> spectators, double lambda, uint N);
void writeData_Uniform(string experiment, vector<double> awae_results, vector<double> target_init_entropy, vector<int> spectators, uint N);

void writeData_lnorm_differential(string experiment, vector<int> spectators, vector<double> h_T, vector<double> h_S, vector<double> h_T_S, vector<double> differential_awae, double mu, double sigma, uint N);

void writeData_lnorm(string experiment, vector<double> awae_results, vector<double> target_init_entropy, vector<int> spectators, vector<double> h_T, vector<double> h_S, vector<double> h_T_S, vector<double> differential_awae, double mu, double sigma, uint N);
#endif