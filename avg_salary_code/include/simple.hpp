#ifndef SIMPLE_H_
#define SIMPLE_H_

#include "constants.hpp"
#include "ent_est.hpp"
#include "gamma.hpp"
#include "gaussian_mixed.hpp"
#include "joint.hpp"
#include "joint_gaussian_v2.hpp"
#include "joint_poisson.hpp"
#include "lognormal_v2.hpp"
#include "min_entropy.hpp"
#include "normal.hpp"
#include "normal_v2.hpp"
#include "plot.hpp"
#include "poisson.hpp"
#include "sln.hpp"
#include "test_gsl.hpp"
#include "test_multiple_gaussian.hpp"
#include "types.hpp"
#include "uniform.hpp"
#include "testing.hpp"
#include "utilities.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/types.h>
#include <vector>

using namespace std;

void simple_main();
#endif