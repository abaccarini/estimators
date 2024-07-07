#ifndef MIN_ENTROPY_H
#define MIN_ENTROPY_H

#include "poisson.hpp"
#include "uniform.hpp"
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

typedef struct output_str_min_ent_poisson_struct {
    double lambda = 0.0;
    uint num_spec;
    long double H_T = 0.0;
    long double awae = 0.0;
} output_str_min_ent_poisson;

long double min_entropy(uint x_A, uint numSpectators, double pois_lambda);

void min_ent_driver_poisson(double lambda);
void min_ent_driver_uniform();

#endif
