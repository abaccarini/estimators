#include "test_gsl.hpp"

void test_gsl_rng() {
    const gsl_rng_type *T;
    gsl_rng *r;

    int i, n = 20;
    double mu = 3.0;

    gsl_rng_env_setup();

    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    /* print n random variates chosen from
       the poisson distribution with mean
       parameter mu */
    // THE INPUT IS AUTOMATICALLY SQUARED IN THE FUNCTON SO WE NEED TO TAKE THE SQUARE ROOT OF SIGMA AS A INPUT
    // printf (" %f\n\n",gsl_ran_lognormal_pdf (1.0, 0, sqrt(0.25)));
    for (i = 0; i < n; i++) {
        // unsigned int k = gsl_ran_poisson (r, mu);
        double k = gsl_ran_lognormal(r, 0, 2.0);
        // printf (" %d", k);
        printf(" %f", k);
    }

    printf("\n");
    gsl_rng_free(r);
}