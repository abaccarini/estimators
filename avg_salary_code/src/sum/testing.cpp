#include "testing.hpp"

// decisionTree* testTree;

void test_main() {
    // test_misc();
    // test_misc_2();
    // test_comparison();
    test_median();
    test_max();
}

// returns the largest value among all the inputs
int max_func(vector<int> target_inputs, vector<int> attacker_inputs, vector<int> spectator_inputs) {
    int mx = max(
        max(
            *max_element(target_inputs.begin(), target_inputs.end()),
            *max_element(attacker_inputs.begin(), attacker_inputs.end())),
        *max_element(spectator_inputs.begin(), spectator_inputs.end()));
    //    cout<<mx<<endl;
    return mx;
}

int max_func_m_1(vector<int> target_inputs, vector<int> attacker_inputs, vector<int> spectator_inputs) {
    std::vector<int> combined;

    combined.reserve(target_inputs.size() + attacker_inputs.size() + spectator_inputs.size()); // preallocate memory
    combined.insert(combined.end(), target_inputs.begin(), target_inputs.end());
    combined.insert(combined.end(), attacker_inputs.begin(), attacker_inputs.end());
    combined.insert(combined.end(), spectator_inputs.begin(), spectator_inputs.end());
    nth_element(combined.begin(), combined.begin() + 1, combined.end(), std::greater{});
    // int mx = combined[1];
    return combined[1];
}

double median(vector<int> target_inputs, vector<int> attacker_inputs, vector<int> spectator_inputs) {
    std::vector<int> combined;

    combined.reserve(target_inputs.size() + attacker_inputs.size() + spectator_inputs.size()); // preallocate memory
    combined.insert(combined.end(), target_inputs.begin(), target_inputs.end());
    combined.insert(combined.end(), attacker_inputs.begin(), attacker_inputs.end());
    combined.insert(combined.end(), spectator_inputs.begin(), spectator_inputs.end());
    auto n = combined.size() / 2;
    nth_element(combined.begin(), combined.begin() + n, combined.end());
    // cout << "combined: ";
    // print_vec(combined);
    // return combined[n];
    // auto n = v.size() / 2;
    // nth_element(v.begin(), v.begin()+n, v.end());
    double med = combined[n];
    if (!(combined.size() & 1)) { // If the set size is even
        auto max_it = max_element(combined.begin(), combined.begin() + n);
        med = (*max_it + med) / 2.0;
    }
    return med;
}

double median_round_down(vector<int> target_inputs, vector<int> attacker_inputs, vector<int> spectator_inputs) {
    std::vector<int> combined;

    combined.reserve(target_inputs.size() + attacker_inputs.size() + spectator_inputs.size()); // preallocate memory
    combined.insert(combined.end(), target_inputs.begin(), target_inputs.end());
    combined.insert(combined.end(), attacker_inputs.begin(), attacker_inputs.end());
    combined.insert(combined.end(), spectator_inputs.begin(), spectator_inputs.end());
    auto n = combined.size() / 2;
    nth_element(combined.begin(), combined.begin() + n, combined.end());
    // cout << "combined: ";
    // print_vec(combined);
    // return combined[n];
    // auto n = v.size() / 2;
    // nth_element(v.begin(), v.begin()+n, v.end());
    double med = combined[n];
    if (!(combined.size() & 1)) { // If the set size is even
        auto max_it = max_element(combined.begin(), combined.begin() + n);
        med = floor((*max_it + med) / 2.0);
    }
    // cout<<med<<endl;
    return med;
}

// computes the median of all the inputs
// if the number of inputs is EVEN, it will return the smaller of the two middle values
double median_min(vector<int> target_inputs, vector<int> attacker_inputs, vector<int> spectator_inputs) {
    std::vector<int> combined;

    combined.reserve(target_inputs.size() + attacker_inputs.size() + spectator_inputs.size()); // preallocate memory
    combined.insert(combined.end(), target_inputs.begin(), target_inputs.end());
    combined.insert(combined.end(), attacker_inputs.begin(), attacker_inputs.end());
    combined.insert(combined.end(), spectator_inputs.begin(), spectator_inputs.end());
    auto n = combined.size() / 2;
    if (!(combined.size() & 1)) { // If the set size is even, this gets the
        nth_element(combined.begin(), combined.begin() + n - 1, combined.end());
        return combined[n - 1];
    }
    nth_element(combined.begin(), combined.begin() + n, combined.end());
    return combined[n];
}

int comparison(vector<int> target_inputs, vector<int> attacker_inputs, vector<int> spectator_inputs) {
    // notation:
    // target_inputs[0] == x
    // attacker_inputs[0] == y
    return (target_inputs[0] > attacker_inputs[0]);

    // if we want to test a function that sums all the inputs
    // return (accumulate(inputs.begin(), inputs.end(),0));
}
int summation(vector<int> target_inputs, vector<int> attacker_inputs, vector<int> spectator_inputs) {
    // equivalent to deterimining the average
    // notation:
    // target_inputs == x
    // attacker_inputs == y
    // spectator_inputs == z
    return accumulate(target_inputs.begin(), target_inputs.end(), accumulate(attacker_inputs.begin(), attacker_inputs.end(), accumulate(spectator_inputs.begin(), spectator_inputs.end(), 0)));
}

void test_max() {

    string experiment = "max_m_1";

    int n = 16;
    vector<int> a_range{0, n};
    std::vector<int> D_o(n);
    std::iota(D_o.begin(), D_o.end(), 0); // ivec will become: [0..99]
    int numSpec = 6;
    vector<output_max> output;
    for (int i = 1; i < numSpec; i++) {
        output.push_back(max_exp(experiment, max_func_m_1, n, a_range, {0.5, 0.5, 0.5}, {UNIFORM, UNIFORM, UNIFORM}, {1, 1, i}, D_o));
    }

    writeData_max(output, experiment, "uniform", to_string(n));

    string experiment2 = "max";

    vector<output_max> output_2;
    for (int i = 1; i < numSpec; i++) {
        output_2.push_back(max_exp(experiment2, max_func, n, a_range, {0.5, 0.5, 0.5}, {UNIFORM, UNIFORM, UNIFORM}, {1, 1, i}, D_o));
    }

    writeData_max(output_2, experiment2, "uniform", to_string(n));
}

void test_median() {

    string experiment = "median_min";

    int n = 16;
    vector<int> a_range{0, n};
    int numSpec = 6;
    // std::vector<double> D_o = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    vector<output_max> output;
    for (int i = 1; i < numSpec; i++) {
        std::cout << "numSpec = " << i << endl;
        output.push_back(median_exp(experiment, median_min, n, a_range, {0.5, 0.5, 0.5}, {UNIFORM, UNIFORM, UNIFORM}, {1, 1, i}));
    }

    writeData_max(output, experiment, "uniform", to_string(n));

    string experiment2 = "median";

    vector<output_max> output_2;
    for (int i = 1; i < numSpec; i++) {
        std::cout << "numSpec = " << i << endl;
        output_2.push_back(median_exp(experiment2, median, n, a_range, {0.5, 0.5, 0.5}, {UNIFORM, UNIFORM, UNIFORM}, {1, 1, i}));
    }

    writeData_max(output_2, experiment2, "uniform", to_string(n));
}

output_max median_exp(string experimentName, function<double(vector<int>, vector<int>, vector<int>)> funct, int N, vector<int> a_range, vector<double> probabilities, vector<uint> distFlags, vector<int> participants) {

    std::vector<double> D_o = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    // std::vector<double> D_o(N);
    // std::iota(D_o.begin(), D_o.end(), 0); // ivec will become: [0..99]

    // printf("no output_domain provided\n");
    functionEvaluation<double> func_config(N, participants.at(0), participants.at(1), participants.at(2), distFlags, probabilities, a_range, funct, D_o);

    func_config.generateTWAEs();
    func_config.generateAWAEs();
    cout << "awaes : ";
    func_config.print(func_config.awae_values);

    cout << "twaes : ";
    func_config.print(func_config.twae_values);

    output_max output = {
        func_config.numTargets,
        func_config.numAttackers,
        func_config.numSpectators,
        N,
        func_config.awae_map,
        func_config.twae_map};
    return output;
}

output_max max_exp(string experimentName, function<int(vector<int>, vector<int>, vector<int>)> funct, int N, vector<int> a_range, vector<double> probabilities, vector<uint> distFlags, vector<int> participants, vector<int> output_domain) {

    functionEvaluation<int> func_config(N, participants.at(0), participants.at(1), participants.at(2), distFlags, probabilities, a_range, funct, output_domain);

    func_config.generateTWAEs();
    func_config.generateAWAEs();
    cout << "awaes : ";
    func_config.print(func_config.awae_values);

    cout << "twaes : ";
    func_config.print(func_config.twae_values);

    output_max output = {
        func_config.numTargets,
        func_config.numAttackers,
        func_config.numSpectators,
        N,
        func_config.awae_map,
        func_config.twae_map};
    return output;
}

void writeData_max(vector<output_max> out_str, string experiment_name, string distribution, string param_string) {
    string extension = ".csv";
    string dir_path = "../output/" + experiment_name + "/" + distribution + "/" + param_string;
    // making parent directory for all experiments to go into, if it doesnt exist
    std::system(("mkdir -p " + dir_path).c_str());

    string path = dir_path + "/results" + extension;

    ofstream outFile(path);
    outFile << "num_T,num_A,num_S,N,x_a_awae,x_T_twae\n";

    for (auto &outpt : out_str) {
        outFile << outpt.num_T
                << "," << outpt.num_A
                << "," << outpt.num_S
                << "," << outpt.N << ",";
        for (auto const &[key, val] : outpt.awae_map) {
            outFile << key // string (key)
                    << ':'
                    << val // string's value
                    << ';';
        }
        outFile << ",";
        for (auto const &[key, val] : outpt.twae_map) {
            outFile << key // string (key)
                    << ':'
                    << val // string's value
                    << ';';
        }
        outFile << "\n";
    }

    outFile.close();
    outFile.clear();
}

void singleExperiment(string experimentName, function<int(vector<int>, vector<int>, vector<int>)> funct, int N, vector<int> a_range, vector<double> probabilities, vector<uint> distFlags, vector<int> participants, vector<int> output_domain) {

    functionEvaluation<int> func_config(N, participants.at(0), participants.at(1), participants.at(2), distFlags, probabilities, a_range, funct, output_domain);

    // cout << "p_S : ";
    // func_config.print(func_config.p_S);

    func_config.generateTWAEs();
    func_config.generateAWAEs();
    // cout << "awaes : ";
    // func_config.print(func_config.awae_values);

    // cout << "twaes : ";
    // func_config.print(func_config.twae_values);

    writeData(experimentName, func_config);
}

void singleExperiment(string experimentName, function<int(vector<int>, vector<int>, vector<int>)> funct, int N, vector<int> a_range, vector<double> probabilities, vector<uint> distFlags, vector<int> participants) {

    functionEvaluation<int> func_config(N, participants.at(0), participants.at(1), participants.at(2), distFlags, probabilities, a_range, funct);

    func_config.generateTWAEs();
    func_config.generateAWAEs();

    // writeData(experimentName, func_config);
}
