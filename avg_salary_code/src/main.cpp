#include "main.hpp"

int main() {
    // test_main();
    auto start = std::chrono::system_clock::now();

//    glpk_test(); 
    simple_main();
    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    return 0;
}