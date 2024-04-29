#ifndef _DATA_IO_HPP_
#define _DATA_IO_HPP_

#include "plug-in_est.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

using json = nlohmann::json;

using std::string;
using std::vector;

template <typename...>
std::string datetime() {
    time_t rawtime;
    struct tm *timeinfo;
    char buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer, 80, "%m%d%Y%H%M", timeinfo); // fixed format for timestamp
    return std::string(buffer);
}

static const std::string timestamp = datetime(); // generating a single timestamp for an execution

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
struct discrete_data {
    std::string name;
    DIST_T<IN_T> dist;
    const size_t num_samples;
    const size_t num_T;
    const size_t num_A;
    const size_t num_iterations;                             // how many times we repeat the computation to elimiate any random noise, and get closer to convergence upon the "true" entropy
    entType target_init_entropy;
    std::map<size_t, std::map<IN_T, entType>> awae_data; // maps number of spectators to all of the awae data
    // if we start varying other parameters (e.g. numT, numA), we may need to modify this
};

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
struct continuous_data {
    std::string name;
    DIST_T<IN_T> dist;
    const size_t numOutputSamples;
    const size_t numInputSamples;
    const size_t num_T;
    const size_t num_A;
    const size_t num_iterations;                             // how many times we repeat the computation to elimiate any random noise, and get closer to convergence upon the "true" entropy
    entType target_init_entropy;
    std::map<size_t, std::map<IN_T, entType>> awae_data; // see above
    const uint k;
};

template <typename IN_T, template <typename> typename DIST_T>
json getDistribution(DIST_T<IN_T> dist) {
    try {
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::normal_distribution<IN_T>>)
            return json{
                {"dist_name", "normal"},
                {"mu", dist.mean()},
                {"sigma", dist.stddev()},
                {"param_str", "("+std::to_string(dist.mean()) + "," + std::to_string(dist.stddev()) + ")"},

            };
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::uniform_real_distribution<IN_T>>)
            return json{
                {"dist_name", "uniform_real"},
                {"a", dist.a()},
                {"b", dist.b()},
                {"param_str", "("+std::to_string(dist.a()) + "," + std::to_string(dist.b()) + ")"},

            };
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::uniform_int_distribution<IN_T>>)
            return json{
                {"dist_name", "uniform_int"},
                {"a", dist.a()},
                {"b", dist.b()},
                {"param_str", "("+std::to_string(dist.a()) + "," + std::to_string(dist.b()) + ")"},

            };
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::poisson_distribution<IN_T>>)
            return json{
                {"dist_name", "poisson"},
                {"lam", dist.mean()},
                {"param_str", "("+std::to_string(int(dist.mean()))+ ")"},

            };
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::lognormal_distribution<IN_T>>)
            return json{
                {"dist_name", "lognormal"},
                {"m", dist.m()},
                {"s", dist.s()},
                {"param_str", "("+std::to_string(dist.m()) + "," + std::to_string(dist.s()) + ")"},

            };
        else {
            throw std::runtime_error("unsupported distribution encountered, please add the distribution you want in an additional \"if\" block here.");
        }
    } catch (const std::exception &ex) {
        std::cerr << "(getDistribution): " << ex.what() << std::endl;
        exit(1);
    }
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void to_json(json &j, const discrete_data<IN_T, OUT_T, DIST_T> &p) {

    json dist_info = getDistribution<IN_T, DIST_T>(p.dist);

    j = json{
         {"name", p.name},
         {"dist", dist_info},
         {"num_samples", p.num_samples},
         {"num_T", p.num_T},
         {"num_A", p.num_A},
         {"num_iterations", p.num_iterations},
         {"target_init_entropy", p.target_init_entropy},
         {"awae_data", p.awae_data},
         {"timestamp", timestamp}
        };
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void to_json(json &j, const continuous_data<IN_T, OUT_T, DIST_T> &p) {

    json dist_info = getDistribution<IN_T, DIST_T>(p.dist);

    j = json{
         {"name", p.name},
         {"dist", dist_info},
         {"num_output_samples", p.numOutputSamples},
         {"num_input_samples", p.numOutputSamples},
         {"num_T", p.num_T},
         {"num_A", p.num_A},
         {"num_iterations", p.num_iterations},
         {"k", p.k},
         {"target_init_entropy", p.target_init_entropy},
         {"awae_data", p.awae_data},
         {"timestamp", timestamp}
        };
}
template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void writeJSON_discrete(discrete_data<IN_T, OUT_T, DIST_T> data) {
    static string extension = ".json";

    json js;
    to_json(js, data);

    string dir_path = "../output_cpp/" + string(data.name) + "/" + string(js.at("dist").at("dist_name"))+"/";
    std::system(("mkdir -p " + dir_path).c_str());
    // std::cout << js.dump(1) << std::endl;

    std::string fpath = dir_path + string(js.at("dist").at("param_str"))
    // +":" + string(js.at("timestamp")) 
    + extension;
    std::cout << dir_path << std::endl;
    std::cout << fpath << std::endl;

    std::ofstream file(fpath);
    file << std::setw(2) << js << std::endl;
    file.flush();
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void writeJSON_continuous(continuous_data<IN_T, OUT_T, DIST_T> data) {
    static string extension = ".json";

    json js;
    to_json(js, data);

    string dir_path = "../output_cpp/" + string(data.name) + "/" + string(js.at("dist").at("dist_name"))+"/";
    std::system(("mkdir -p " + dir_path).c_str());
    // std::cout << js.dump(1) << std::endl;

    std::string fpath = dir_path + string(js.at("dist").at("param_str"))
    // +":" + string(js.at("timestamp")) 
    + extension;
    std::cout << dir_path << std::endl;
    std::cout << fpath << std::endl;

    std::ofstream file(fpath);
    file << std::setw(2) << js << std::endl;
    file.flush();
}

#endif // _DATA_IO_HPP_