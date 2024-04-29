#ifndef _TESTING_HPP_
#define _TESTING_HPP_

#include <functional>

void testing_main();
void test_est();
void compute_awae();
void test_knn_est();

double test_fn(double x, void *p);
double my_ln(double x);
void test_gsl();

template <typename IN_T, typename OUT_T>
class test_obj {
public:
    test_obj(const size_t _numSamples, const std::function<OUT_T(IN_T)> _func) : numSamples(_numSamples), func(_func){

                                                                                                          };
    ~test_obj(){};

    void set_x_A(IN_T x) {
        x_A = x;
    };
    IN_T get_x_A() {
        return x_A;
    };
    OUT_T evalute_fn(IN_T x) {
        return func(x);
    };

private:
    const size_t numSamples;
    const std::function<OUT_T(IN_T)> func;
    IN_T x_A;
};

#endif // _TESTING_HPP_
