#include "trapezoid.hpp"

void integrate_trap_2d(const my_gsl_function_vec *f, double a, double b, double c, double d, size_t N, size_t M, double *result) {
    // double y[0];
    // double y = GSL_FN_EVAL(f, x); // evaluates a GSL function f(x)
    // auto z = MY_GSL_FN_VEC_EVAL(f, x); // evaluates a GSL function of arbitrary dimension f(x)

    double dx = (b - a) / (double) N;
    double dy = (d - c) / (double) M;

    *result = 0.0;
    std::vector<double> x = {a, b};
    *result += MY_GSL_FN_VEC_EVAL(f, x);
    x = {a, d};
    *result += MY_GSL_FN_VEC_EVAL(f, x);
    x = {b, c};
    *result += MY_GSL_FN_VEC_EVAL(f, x);
    x = {b, d};
    *result += MY_GSL_FN_VEC_EVAL(f, x);

    *result *= (dx * dy) * (0.25);

    double sum = 0.0;

    for (size_t i = 0; i < N; i++) {
        x = {a + i * dx, c};
        sum += MY_GSL_FN_VEC_EVAL(f, x);
        x[1] = d;
        sum += MY_GSL_FN_VEC_EVAL(f, x);
    }
    for (size_t i = 0; i < M; i++) {
        x = {a, c + i * dy};
        sum += MY_GSL_FN_VEC_EVAL(f, x);
        x[0] = b;
        sum += MY_GSL_FN_VEC_EVAL(f, x);
    }
    *result += (dx * dy) * (0.5) * sum;

    sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            x = {a + i*dx, c + j * dy};
            sum += MY_GSL_FN_VEC_EVAL(f, x);
        }
    }
    *result += sum;
}