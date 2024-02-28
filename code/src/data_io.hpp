#ifndef _DATA_IO_HPP_
#define _DATA_IO_HPP_

#include "utilities.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
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
    strftime(buffer, 80, "%m-%d-%Y_%H%M", timeinfo);
    return std::string(buffer);
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
struct discrete_data {
    std::string name;
    DIST_T<IN_T> dist;
    const size_t num_samples;
    const size_t num_T;
    const size_t num_A;
    const size_t num_iterations;                             // how many times we repeat the computation to elimiate any random noise, and get closer to convergence upon the "true" entropy
    std::map<size_t, std::map<IN_T, long double>> awae_data; // maps number of spectators to all of the awae data
    // if we start varying other parameters (e.g. numT, numA), we may need to modify this
};

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
struct continuous_data {
    std::string name;
    DIST_T<IN_T> dist;
    const size_t num_samples;
    const size_t num_T;
    const size_t num_A;
    const size_t num_iterations;                             // how many times we repeat the computation to elimiate any random noise, and get closer to convergence upon the "true" entropy
    std::map<size_t, std::map<IN_T, long double>> awae_data; // see above
};

template <typename IN_T, template <typename> typename DIST_T>
json getDistribution(DIST_T<IN_T> dist) {
    try {
        // if (typeid(DIST_T<IN_T>) == typeid(std::normal_distribution<IN_T>)) {
        //     return "normal_pdf";
        // }
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::normal_distribution<IN_T>>)
            return json {
                {"dist_name", "normal"},
                    {"mu", dist.mean()},
                    {"sigma", dist.stddev()},
            };
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::uniform_real_distribution<IN_T>>)
            return json {
                {"dist_name", "uniform_real"},
                    {"a", dist.a()},
                    {"b", dist.b()},
            };
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::uniform_int_distribution<IN_T>>)
            return json {
                {"dist_name", "uniform_int"},
                    {"a", dist.a()},
                    {"b", dist.b()},
            };     
        if constexpr (std::is_same_v<DIST_T<IN_T>, std::lognormal_distribution<IN_T>>)
            return json {
                {"dist_name", "lognormal"},
                    {"m", dist.m()},
                    {"s", dist.s()},
            };   
        else {
            throw std::runtime_error("unsupported distribution encountered, please add the distribution you want in an additional \"if\" block here.");
        }
    } catch (const std::exception &ex) {
        std::cerr << "(getDistribution): " << ex.what() << std::endl;
        exit(1);
    }
}


// template <typename IN_T, typename OUT_T>
// json getFunction(std::function<OUT_T(std::map<IN_T,size_t>, const size_t&)>) {
//     try {
       
//         if constexpr ()
//             // return json {
//             //     {"dist_name", "normal"},
//             //         {"mu", dist.mean()},
//             //         {"sigma", dist.stddev()},
//             // };
//         else {
//             throw std::runtime_error("unsupported distribution encountered, please add the pdf of the distribution you want in pdfs.hpp, as well as additional \"if\" block here.");
//         }
//     } catch (const std::exception &ex) {
//         std::cerr << "(evaluate_pdf): " << ex.what() << std::endl;
//         exit(1);
//     }
// }

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void to_json(json &j, const discrete_data<IN_T, OUT_T, DIST_T> &p) {

    j = json{
        {"name", p.name},
        {"dist", getDistribution<IN_T, DIST_T>(p.dist)},
        {"num_samples", p.num_samples},
        {"num_T", p.num_T},
        {"num_A", p.num_A},
        {"num_iterations", p.num_iterations},
        {"awae_data", p.awae_data}};
}

template <typename IN_T, typename OUT_T, template <typename> typename DIST_T>
void writeJSON_discrete(discrete_data<IN_T, OUT_T, DIST_T> data) {
    json json;
    to_json(json, data);
    std::cout << json.dump(1) << std::endl;
}

#endif // _DATA_IO_HPP_