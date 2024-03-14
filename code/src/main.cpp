#include "testing.hpp"
#include <chrono>
#include <iostream>

int main() {

    // if (__cplusplus == 202101L)
    //     std::cout << "C++23";
    // else if (__cplusplus == 202002L)
    //     std::cout << "C++20";
    // else if (__cplusplus == 201703L)
    //     std::cout << "C++17";
    // else if (__cplusplus == 201402L)
    //     std::cout << "C++14";
    // else if (__cplusplus == 201103L)
    //     std::cout << "C++11";
    // else if (__cplusplus == 199711L)
    //     std::cout << "C++98";
    // else
    //     std::cout << "pre-standard C++." << __cplusplus;
    // std::cout << "\n";

    auto start = std::chrono::system_clock::now();
    testing_main();

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return 0;
}