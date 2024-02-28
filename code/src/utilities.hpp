
#ifndef _UTILITIES_HPP_
#define _UTILITIES_HPP_

#include <random>
#include <iostream>
#include <vector>
using std::ostream;
using std::vector;

template <typename TType>
void printVector(const std::vector<TType> &vec) {
    typename std::vector<TType>::const_iterator it;
    std::cout << "(";
    for (it = vec.begin(); it != vec.end(); it++) {
        if (it != vec.begin())
            std::cout << ", ";
        std::cout << (*it);
    }
    std::cout << ")\n";
}

template <typename T2>
void printVector(const std::vector<std::vector<T2>> &vec) {
    for (auto it = vec.begin(); it != vec.end(); it++) {
        print_vector(*it);
    }
}


template<typename T>
ostream& operator<< (ostream& out, const vector<T>& v) {
    out << "[";
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last)
            out << ", ";
    }
    out << "]";
    return out;
}

// template<typename T>
// void printVector(const T& t) {
//     std::copy(t.cbegin(), t.cend(), std::ostream_iterator<typename T::value_type>(std::cout, ", "));
// }

// template<typename T>
// void printVectorInVector(const T& t) {
//     std::for_each(t.cbegin(), t.cend(), printVector<typename T::value_type>);
// }
#endif // _UTILITIES_HPP_