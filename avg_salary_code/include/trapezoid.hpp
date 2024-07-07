#ifndef TRAPEZOID_H_
#define TRAPEZOID_H_

#include "constants.hpp"
#include "plot.hpp"
#include "poisson.hpp"
#include "normal.hpp"
#include "sln.hpp"
#include "utilities.hpp"
#include <cmath>
#include <gmp.h>
#include <gmpxx.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <iostream>
#include <mpfr.h>
#include <mpir.h>
#include <string>
#include <vector>

struct my_gsl_function_vec_struct 
{
  int (* function) (std::vector<double> const &x, void * params);
  void * params;
};

typedef struct my_gsl_function_vec_struct my_gsl_function_vec;

#define MY_GSL_FN_VEC_EVAL(F,x) (*((F)->function))(x,(F)->params)



#endif // TRAPEZOID_H_