#ifndef UTIL_H_
#define UTIL_H_

#include "constants.hpp"
#include "types.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <vector>

using namespace std;

// used for protection against division by zero
void print_vec(vector<int> int_vec);

class utilities {

public:
    utilities();

    // overloading
    void print_nl(vector<int> int_vec);

    void print(vector<int>);
    void print(vector<vector<int>>);
    void print(vector<double>);
    void print(vector<string>);
    void print(int);
    void print(string);
    void print(double);
    void print(Lint);

    Lint factorial(Lint n);
    void assignProbabilities(vector<double> &res, uint N, double p, uint flag);
    void assignProbabilities(vector<double> &res, uint _N, uint NumPart, double lam);
    double binomialPMF(uint k, uint n, double p);
    double poissonPMF(uint k, uint N_RV, double lambda);
    Lint nCk(Lint n, Lint k);
    double gFunc(double n);
    void makeCombiUtil(vector<vector<int>> &ans, vector<int> &tmp, int n, int left, int k);
    vector<vector<int>> makeCombi(int lower, int upper, int k);
};

long double gFunc(long double n);

template <typename T>
std::string to_string_prec(const T a_value, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

#endif