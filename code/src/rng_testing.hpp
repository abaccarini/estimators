#ifndef _RNG_TESTING_HPP_
#define _RNG_TESTING_HPP_

void rng_testing_main();

void test_gsl(const unsigned long qty, const unsigned long Nm1);
void test_xos(const unsigned long qty, const unsigned long Nm1);
void test_pcg(const unsigned long qty, const unsigned long Nm1);
void test_random(const unsigned long qty, const unsigned long Nm1);

#endif // _RNG_TESTING_HPP_