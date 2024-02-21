#ifndef _RAND_GEN_HPP_
#define _RAND_GEN_HPP_

#include <random>
#include <limits>

// generates uniformly random data over the domain [range_from, range_to] INCLUSIVE
std::random_device rand_dev;
std::mt19937 generator(rand_dev());
template <typename T>
T random(T range_from, T range_to) {
    std::uniform_int_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

template <typename T>
class Generator {
public:
    Generator();
    virtual ~Generator();
    double rand01(); //random number [0,1)
    T rand_range(T range_from, T range_to); //random number [0,1)
private:
    std::random_device randev; //seed
    /*Engines*/
    std::mt19937_64 gen;
    std::uniform_real_distribution<double> uniform01;
    std::uniform_int_distribution<T> uniform_disc;


    
};

template <typename T>
Generator<T>::Generator() : randev(), gen(randev()), uniform01(0.,1.) {
}

template <typename T>
Generator<T>::~Generator() { }

template <typename T>
double Generator<T>::rand01() {
    return generate_canonical<double,std::numeric_limits<double>::digits>(gen);
}
template <typename T>
T Generator<T>::rand_range(T range_from, T range_to) {
    return generate_canonical<double,std::numeric_limits<double>::digits>(gen);
}




#endif // _RAND_GEN_HPP_