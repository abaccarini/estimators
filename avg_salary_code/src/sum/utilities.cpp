#include "utilities.hpp"


void print_vec(vector<int> int_vec) {

    cout << '[';
    for (int i = 0; i < (int)int_vec.size(); i++) {
        cout << int_vec.at(i);
        if (i < (int)int_vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << ']' << endl;
}

utilities::utilities() {}

void utilities::print(vector<int> int_vec) {

    cout << '[';
    for (int i = 0; i < (int)int_vec.size(); i++) {
        cout << int_vec.at(i);
        if (i < (int)int_vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << ']' << endl;
}


void utilities::print_nl(vector<int> int_vec) {

    cout << '[';
    for (int i = 0; i < (int)int_vec.size(); i++) {
        cout << int_vec.at(i);
        if (i < (int)int_vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << '] ';
}

void utilities::print(vector<vector<int>> int_vec) {

    cout << '[';
    for (int i = 0; i < (int)int_vec.size(); i++) {
        cout << '[';
        for (int j = 0; j < (int)int_vec[i].size(); j++) {
            cout << int_vec.at(i).at(j);
            if (j < (int)int_vec[i].size() - 1) {
                cout << ", ";
            }
        }
        cout << "]";

        if (i < (int)int_vec.size() - 1) {
            cout << ' ';
        }
    }
    cout << "]" << endl;
}

void utilities::print(vector<double> double_vec) {
    cout << '[';
    for (int i = 0; i < (int)double_vec.size(); i++) {
        cout << double_vec.at(i);
        if (i < (int)double_vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << ']' << endl;
}

void utilities::print(vector<string> string_vec) {
    cout << '[';
    for (int i = 0; i < (int)string_vec.size(); i++) {
        cout << string_vec.at(i);
        if (i < (int)string_vec.size() - 1) {
            cout << ", ";
        }
    }
    cout << ']' << endl;
}

void utilities::print(int x) {
    cout << x << endl;
}
void utilities::print(double x) {
    cout << x << endl;
}

void utilities::print(string x) {
    cout << x << endl;
}

void utilities::print(Lint x) {
    cout << x << endl;
}

// source
// https://www.geeksforgeeks.org/make-combinations-size-k/
void utilities::makeCombiUtil(vector<vector<int>> &ans,
                              vector<int> &tmp, int n, int left, int k) {

    // Pushing this vector to a vector of vector
    if (k == 0) {
        // this makes an empty vector non-empty
        ans.push_back(tmp);
        return;
    }

    // i iterates from left to n. First time
    // left will be 1
    for (int i = left; i < n; ++i) {
        tmp.push_back(i);
        makeCombiUtil(ans, tmp, n, left, k - 1);

        // Popping out last inserted element
        // from the vector
        tmp.pop_back();
    }
}

// Prints all combinations of size k of numbers
// from 1 to n.
vector<vector<int>> utilities::makeCombi(int lower, int upper, int k) {
    vector<vector<int>> ans;
    vector<int> tmp;
    // left we pass considers attackers range
    makeCombiUtil(ans, tmp, upper, lower, k);
    return ans;
}

double utilities::gFunc(double n) {
    // chosen by convention, since lim_{n->0} n log n = 0
    if (n == 0) {
        return 0;
    } else {
        return (-1) * n * log2(n);
    }
}

Lint utilities::factorial(Lint n) {
    // chosen by convention, since lim_{n->0} n log n = 0
    Lint ans = 1;
    for (Lint i = 1; i < n; i++) {
        ans *= i;
    }
    return ans;
}

double utilities::binomialPMF(uint k, uint n, double p) {
    if (k > n) {
        printf("Error: n must be >= k\n");
        return -1;
    } else {
        return nCk(n, k) * pow(p, k) * pow(1 - p, n - k);
    }
}

double utilities::poissonPMF(uint x, uint N_RV, double lambda) {
    // return exp(-(1.0) * lambda * N_RV) * pow(lambda * N_RV, x) / factorial(x);
    // return (( exp(-(1.0) * lambda ) * pow(lambda, x) )/ (double) factorial(x));
    return pow(M_E, x * log(lambda * N_RV) - (lambda * N_RV) - lgamma(x + 1.0));
    // return x * log(lambda * N_RV) - (lambda * N_RV) - lgamma(x + 1.0);
}

// assigns the probabilities for the parties
void utilities::assignProbabilities(vector<double> &res, uint _N, double p, uint flag) {
    double temp;
    switch (flag) {
    case 0:
        temp = (double)1 / (double)_N;
        for (uint i = 0; i < _N; i++) {
            res[i] = temp;
        }
        break;
    case 1:
        for (uint i = 0; i < _N; i++) {
            res[i] = binomialPMF(i, _N - 1, p);
        }
        break;
    }
}

void utilities::assignProbabilities(vector<double> &res, uint _N, uint NumPart, double lam) {
    for (uint i = 0; i < _N; i++) {
        res[i] = poissonPMF(i, NumPart, lam);
    }
}

// calculates the binomial coefficient
// still overflows for n = 64
Lint utilities::nCk(Lint n, Lint k) {

    if (k > n) {
        printf("Error: n must be >= k\n");
        return -1;
    } else {
        Lint res = 1;
        // Since C(n, k) = C(n, n-k)
        if (k > n - k) {
            k = n - k;
        }
        // Calculate value of
        // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
        for (Lint i = 0; i < k; ++i) {
            res *= ((Lint)n - i);
            res /= (i + (Lint)1);
        }
        return res;
    }
}

long double gFunc(long double n) {
    // chosen by convention, since lim_{n->0} n log n = 0
    if (n < EPSILON) {
        return 0.0;
    } else {
        return n * log2(n);
    }
}
