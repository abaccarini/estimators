#ifndef ENTROPY_H_
#define ENTROPY_H_

#include "utilities.hpp"
#include <map>
#include <ranges>

using namespace std;

#define UNIFORM 0
#define BINOMIAL 1
#define POISSON 2

template <typename IntegralType>
typename std::enable_if<std::is_integral<IntegralType>::value, bool>::type
equal(const IntegralType &a, const IntegralType &b) {
    return a == b;
}

template <typename FloatingType>
typename std::enable_if<std::is_floating_point<FloatingType>::value, bool>::type
equal(const FloatingType &a, const FloatingType &b) {
    return std::fabs(a - b) < std::numeric_limits<FloatingType>::epsilon();
}

template <typename T>
class functionEvaluation : public utilities {
public:
    functionEvaluation(){};  // constructor
    ~functionEvaluation(){}; // destructor

    // constructor overloading
    // called if output domain must be generated
    functionEvaluation(int, int, int, int, vector<uint>, vector<double>, vector<int>, function<T(vector<int>, vector<int>, vector<int>)>);

    // called if output domain is provided
    functionEvaluation(int, int, int, int, vector<uint>, vector<double>, vector<int>, function<T(vector<int>, vector<int>, vector<int>)>, vector<T>);

    void sharedConstructor();

    double evaluateTWAE(vector<int> &x_T);
    void generateTWAEs();

    double evaluateAWAE_new();
    double H_T();
    double H_S();
    double H_T_S();
    void generateAWAEs_new();
    double evaluateAWAE(vector<int> &x_A);
    void generateAWAEs();

    double evaluateJWAE(vector<int> &, vector<int> &);

    double ent(vector<int> &x_A, const T output, double);

    double singleConditional(const T, vector<int> &, const string &);
    double dualConditional(const T, vector<int> &, vector<int> &);
    double getTargetProbabilitiy(vector<int> &);
    double getAttackerProbabilitiy(vector<int> &);
    double getSpectatorProbabilitiy(vector<int> &);

    // used for generating combinations of possible values

    // used to generate all of the possible outputs if not known
    // useful if a large number of outcomes exists for all inputs
    void generateD_o();

    // MEMBERS ////////////////////////////////////////
    // private:

    // the values in this vector are ordered according to how x_A_combinations are ordered
    std::map<int, double> awae_map;
    std::map<int, double> twae_map;
    vector<double> awae_values;
    vector<double> awae_values_new;
    vector<double> twae_values;

    // associated probabilities for the attackers, targets, and spectators
    vector<double> p_A;
    vector<double> p_T;
    vector<double> p_S;

    // attacker's suspected range of target's values
    // two-element vector [a,b)
    vector<int> attacker_range;

    // possible output range
    vector<T> output_domain;

    // list of all target possible permutations
    // will be generated only once
    // and is based on the attacker's prior knowledge (for x_T)

    vector<vector<int>> x_T_combinations;
    vector<vector<int>> x_A_combinations;
    vector<vector<int>> x_S_combinations;

    // function we are evaluating
    function<T(vector<int>, vector<int>, vector<int>)> funct;

    // range we consider [0..N-1]
    int N;
    int numTargets;
    int numAttackers;
    int numSpectators;

    // success probability use for binomial distribution (if selected)
    vector<double> distProbabilities;
    vector<uint> distFlags;
    // double pr_A;
    // double pr_T;
    // double pr_S;

    // distribution flag
    // uint distFlag;
};

template <typename T>
functionEvaluation<T>::functionEvaluation(int _N, int _numTargets, int _numAttackers,
                                          int _numSpectators,
                                          vector<uint> _distFlags, vector<double> _distProbabilities, vector<int> _attacker_range,
                                          function<T(vector<int>, vector<int>, vector<int>)> _funct) : p_A(_N, 0), p_T(_N, 0), p_S(_N, 0), attacker_range{_attacker_range}, funct{_funct}, N{_N}, numTargets{_numTargets}, numAttackers{_numAttackers}, numSpectators{_numSpectators}, distProbabilities{_distProbabilities}, distFlags{_distFlags} {
    sharedConstructor();

    // generating the output domain
    generateD_o();
    std::cout << "Generated output domain:" << std::endl;
    print(output_domain);
    // generating the awae values
    // generateAWAEs();
}

template <typename T>
functionEvaluation<T>::functionEvaluation(int _N, int _numTargets, int _numAttackers,
                                          int _numSpectators,
                                          vector<uint> _distFlags, vector<double> _distProbabilities, vector<int> _attacker_range,
                                          function<T(vector<int>, vector<int>, vector<int>)> _funct,
                                          vector<T> _output_domain) : p_A(_N, 0), p_T(_N, 0), p_S(_N, 0), attacker_range{_attacker_range}, output_domain{_output_domain}, funct{_funct}, N{_N}, numTargets{_numTargets}, numAttackers{_numAttackers}, numSpectators{_numSpectators}, distProbabilities{_distProbabilities}, distFlags{_distFlags} {
    sharedConstructor();

    // std::cout << "Output domain provided." << std::endl;
    // generateAWAEs();
}
template <typename T>
void functionEvaluation<T>::sharedConstructor() {
    // print(attacker_range.at(1) - attacker_range.at(0));
    if (distFlags[1] == 2) {

        // not needed if we are doing poisson, since we use the simplified version based on the pmf
        // assignProbabilities(p_T, attacker_range.at(1) - attacker_range.at(0), numTargets, distProbabilities[0]);
        // assignProbabilities(p_A, N, numAttackers, distProbabilities[1]);
        // assignProbabilities(p_S, N, numSpectators, distProbabilities[2]);

    } else {
        assignProbabilities(p_T, attacker_range.at(1) - attacker_range.at(0), distProbabilities[0], distFlags[0]);
        assignProbabilities(p_A, N, distProbabilities[1], distFlags[1]);
        assignProbabilities(p_S, N, distProbabilities[2], distFlags[2]);
    }

    // cannot initialize using init_lists because size is unknown
    x_T_combinations = makeCombi(attacker_range.at(0), attacker_range.at(1), numTargets);
    x_A_combinations = makeCombi(0, N, numAttackers);
    x_S_combinations = makeCombi(0, N, numSpectators);

    // used to make output csv formatting easier
    for (int i = 0; i < (int)distProbabilities.size(); i++) {
        if (distFlags.at(i) == 0) {
            distProbabilities.at(i) = 0.0;
        }
    }
}

template <typename T>
void functionEvaluation<T>::generateD_o() {
    T output;
    for (int i = 0; i < (int)x_T_combinations.size(); i++) {
        for (int j = 0; j < (int)x_A_combinations.size(); j++) {
            for (int k = 0; k < (int)x_S_combinations.size(); k++) {
                // generating an output with the (i,j) input combination
                output = funct(x_T_combinations.at(i), x_A_combinations.at(j), x_S_combinations.at(k));

                // print(x_T_combinations.at(i));
                // print(x_A_combinations.at(j));
                // print(x_S_combinations.at(k));
                // std::cout << "output = " <<output<< std::endl;

                // checking if output exists in the current domain
                // if it is not in output_domain, add it to it
                // and continue generating potential values

                bool check = false;
                for (auto &var : output_domain) {
                    if (equal(var, output)) {
                        check = true;
                        break;
                    }
                }
                if (!check) {
                    output_domain.push_back(output);
                }
                // non templated version
                // if (!(find(output_domain.begin(), output_domain.end(), output) != output_domain.end())) {
                //     output_domain.push_back(output);
                // }
            }
        }
    }
    // sorting output domain
    sort(output_domain.begin(), output_domain.end());
}

// outside conditional is either p(O = o | X_A = x_a) (for awae) or p(O = o | X_A = x_a, X_T = x_t) (for twae)
// we first check if the outside conditional is zero
//      if yes, then the whole term will be zero, and we don't need to calculate it
//      otherwise, continue the calculation as normal
template <typename T>
double functionEvaluation<T>::ent(vector<int> &x_A, const T output, const double outside_conditional) {
    // int tracker = 0;
    double result = 0.0, temp = 0.0;

    double a, b; // used so we don't need need calculate this twice to protect against division by zero
    if (outside_conditional < EPSILON) {
        return 0.0;
    } else {

        for (int i = 0; i < (int)x_T_combinations.size(); i++) {
            a = dualConditional(output, x_T_combinations.at(i), x_A);
            // THIS SHOULD BE dualConditional -- ERROR IN PAPER
            b = singleConditional(output, x_A, "X_A");
            temp = (a * getTargetProbabilitiy(x_T_combinations.at(i))) / b;

            // protection against division by zero, avoiding negative infinities as well
            if (b < EPSILON && a < EPSILON) {
                temp = 0.0;
            }

            result += gFunc(temp);
        }
        return result;
    }
}

// p(O = o | Flag = x)
template <typename T>
double functionEvaluation<T>::singleConditional(const T output, vector<int> &x, const string &flag) {
    double result = 0;
    if (flag == "X_A") {
        for (int i = 0; i < (int)x_T_combinations.size(); i++) {
            for (int j = 0; j < (int)x_S_combinations.size(); j++) {
                if (equal(funct(x_T_combinations.at(i), x, x_S_combinations.at(j)), output)) {
                    result += getTargetProbabilitiy(x_T_combinations.at(i)) * getSpectatorProbabilitiy(x_S_combinations.at(j));
                }
            }
        }
    } else if (flag == "X_T") {
        for (int i = 0; i < (int)x_A_combinations.size(); i++) {
            for (int j = 0; j < (int)x_S_combinations.size(); j++) {
                if (equal(funct(x, x_A_combinations.at(i), x_S_combinations.at(j)), output)) {
                    result += getAttackerProbabilitiy(x_A_combinations.at(i)) * getSpectatorProbabilitiy(x_S_combinations.at(j));
                }
            }
        }
    } else {
        cout << "ERROR: Invalid argument passed, exiting...." << endl;
        return -1;
    }
    return result;
}

// p(O = o | X_a = x_a, X_T = x_T)
template <typename T>
double functionEvaluation<T>::dualConditional(const T output, vector<int> &x_T, vector<int> &x_A) {
    double result = 0;
    for (int i = 0; i < (int)x_S_combinations.size(); i++) {
        if (equal(funct(x_T, x_A, x_S_combinations.at(i)), output)) {
            result += getSpectatorProbabilitiy(x_S_combinations.at(i));
        }
    }
    return result;
}

// calculates P(X_T = \vec{x_T})
template <typename T>
double functionEvaluation<T>::getTargetProbabilitiy(vector<int> &x_T) {
    double result = 1.0;
    for (int i = 0; i < (int)x_T.size(); i++) {
        result *= p_T.at(x_T.at(i));
    }
    return result;
}

// calculates P(X_A = \vec{x_A})
template <typename T>
double functionEvaluation<T>::getAttackerProbabilitiy(vector<int> &x_A) {
    double result = 1.0;
    for (int i = 0; i < (int)x_A.size(); i++) {
        result *= p_A.at(x_A.at(i));
    }
    return result;
}

// calculates P(X_S = \vec{x_S})
template <typename T>
double functionEvaluation<T>::getSpectatorProbabilitiy(vector<int> &x_S) {
    double result = 1.0;
    for (int i = 0; i < (int)x_S.size(); i++) {
        result *= p_S.at(x_S.at(i));
    }
    return result;
}

template <typename T>
double functionEvaluation<T>::evaluateAWAE(vector<int> &x_A) {
    double result = 0.0;
    double temp;
    // loop through all possible outputs
    // faster option
    for (int i = 0; i < (int)output_domain.size(); i++) {
        temp = singleConditional(output_domain.at(i), x_A, "X_A");
        result += temp * ent(x_A, output_domain.at(i), temp);
    }

    // alternative awae formulation
    // for (int i = 0; i < (int)x_T_combinations.size(); i++) {
    //     result += getTargetProbabilitiy(x_T_combinations.at(i)) * evaluateJWAE(x_A, x_T_combinations.at(i));
    // }

    return result;
}

template <typename T>
double functionEvaluation<T>::evaluateAWAE_new() {
    return H_T() + H_S() - H_T_S();
    // double temp = H_S()
    // return H_S() ;
}

template <typename T>
double functionEvaluation<T>::H_T() {
    double sum = 0, temp = 0;
    for (int i = 0; i < N; i++) {
        // temp = poissonPMF(i, numTargets, distProbabilities[0]);
        temp = pow(M_E, i * log(distProbabilities[2] * numTargets) - (distProbabilities[2] * numTargets) - lgamma(i + 1.0));

        sum += gFunc(temp);
    }
    // cout<<"target inital entropy = "<<(-1.0) * sum<<endl;
    return (1.0) * sum;
}

template <typename T>
double functionEvaluation<T>::H_S() {
    double sum = 0, temp = 0;
    // cout<<"N * numSpectators = "<<N * numSpectators<<endl;
    for (int i = 0; i < N * numSpectators; i++) {
        // temp = poissonPMF(i, numSpectators, distProbabilities[2]);
        temp = pow(M_E, i * log(distProbabilities[2] * numSpectators) - (distProbabilities[2] * numSpectators) - lgamma(i + 1.0));
        sum += gFunc(temp);
    }
    return (1.0) * sum;
}

template <typename T>
double functionEvaluation<T>::H_T_S() {

    double sum = 0, temp = 0;
    // cout<<" N * (numSpectators + numTargets) = "<< N * (numSpectators + numTargets)<<endl;
    for (int i = 0; i < N * (numSpectators + numTargets); i++) {
        // temp = poissonPMF(i, numSpectators + numTargets, distProbabilities[2]);

        temp = pow(M_E, i * log(distProbabilities[2] * numSpectators + numTargets) - (distProbabilities[2] * numSpectators + numTargets) - lgamma(i + 1.0));

        sum += gFunc(temp);
    }
    return (1.0) * sum;
}

template <typename T>
double functionEvaluation<T>::evaluateTWAE(vector<int> &x_T) {
    double result = 0.0;
    // loop through all possible outputs
    for (int i = 0; i < (int)x_A_combinations.size(); i++) {
        result += getAttackerProbabilitiy(x_A_combinations.at(i)) * evaluateJWAE(x_A_combinations.at(i), x_T);
        // cout<<result<<endl;
    }
    return result;
}

template <typename T>
double functionEvaluation<T>::evaluateJWAE(vector<int> &x_A, vector<int> &x_T) {
    double result = 0.0;
    double temp;
    // loop through all possible outputs
    for (int i = 0; i < (int)output_domain.size(); i++) {
        temp = dualConditional(output_domain.at(i), x_T, x_A);
        // cout<<temp<<endl;
        result += temp * ent(x_A, output_domain.at(i), temp);
    }

    return result;
}
template <typename T>
void functionEvaluation<T>::generateAWAEs() {
    for (int i = 0; i < (int)x_A_combinations.size(); i++) {
        double tmp = evaluateAWAE(x_A_combinations.at(i));
        // std::cout << i << std::endl;
        awae_values.push_back(tmp);
        awae_map.insert({x_A_combinations.at(i).at(0), tmp});
    }
}

template <typename T>
void functionEvaluation<T>::generateAWAEs_new() {
    // this is ONLY FOR AVERGAE FUNC
    // for summation, awae is independent of all possible attackers inputs
    // awae_values_new.push_back(evaluateAWAE_new());
}

template <typename T>
void functionEvaluation<T>::generateTWAEs() {
    for (int i = 0; i < (int)x_T_combinations.size(); i++) {
        double tmp = evaluateTWAE(x_T_combinations.at(i));

        // std::cout << i << std::endl;
        twae_values.push_back(evaluateTWAE(x_T_combinations.at(i)));
        twae_map.insert({x_T_combinations.at(i).at(0), tmp});
    }
}

#endif