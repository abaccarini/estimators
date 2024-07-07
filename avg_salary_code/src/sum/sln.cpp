#include "sln.hpp"

double ln_pdf(double x, void *p) {
    struct ln_params *params = (struct ln_params *)p;
    return (1 / (x * sqrt((params->sigma) * 2 * M_PI))) * exp((-1) * (log(x) - (params->mu)) * (log(x) - (params->mu)) / (2 * (params->sigma)));
}

double uniform_pdf(double x, void *p) {
    struct uniform_params *params = (struct uniform_params *)p;

    if (x > (params->a) and x < (params->b)) {
        return 1.0 / ((params->b) - (params->a));

    } else {
        return 0.0;
    }
}

// computes f(x) log f(X)
double log_ln_pdf(double x, void *p) {
    // struct ln_params *params = (struct ln_params *)p;
    // by definition (according to us)
    if (ln_pdf(x, p) < EPSILON) {
        return 0.0;
    } else {
        return ln_pdf(x, p) * log2(ln_pdf(x, p));
    }
}

// computes f(x) log f(X)
double log_uniform_pdf(double x, void *p) {
    // struct ln_params *params = (struct ln_params *)p;
    // by definition (according to us)
    if (uniform_pdf(x, p) < EPSILON) {
        return 0.0;
    } else {
        return uniform_pdf(x, p) * log2(uniform_pdf(x, p));
    }
}

double ln_pdf_eps(double x, void *p) {
    struct ln_params *params = (struct ln_params *)p;
    return (1 / (x * sqrt((params->sigma) * 2 * M_PI))) * exp((-1) * (log(x) - (params->mu)) * (log(x) - (params->mu)) / (2 * (params->sigma))) - (params->eps);
}

double ln_pdf_deriv(double x, void *p) {
    struct ln_params *params = (struct ln_params *)p;

    return (-1.0) * (log(x) - (params->mu) + (params->sigma)) / (x * x * (params->sigma) * sqrt(2 * M_PI * (params->sigma))) * exp((-1) * (log(x) - (params->mu)) * (log(x) - (params->mu)) / (2 * (params->sigma)));
}

void ln_pdf_deriv_fdf(double x, void *p, double *f, double *df) {
    struct ln_params *params = (struct ln_params *)p;

    *f = (1 / (x * (params->sigma) * sqrt(2 * M_PI))) * exp((-1) * (log(x) - (params->mu)) * (log(x) - (params->mu)) / (2 * (params->sigma))) - (params->eps);

    *df = (-1.0) * (log(x) - (params->mu) + (params->sigma)) / (x * x * (params->sigma) * sqrt(2 * M_PI * (params->sigma))) * exp((-1) * (log(x) - (params->mu)) * (log(x) - (params->mu)) / (2 * (params->sigma)));
}

void update_fw_params(double &mu_out, double &sigma_out, double mu, double sigma, uint16_t L) {
    sigma_out = (log((exp(sigma ) - 1) / L + 1));
    // printf("sigma : %f\n", sigma);
    // printf("sigma_out : %f\n", sigma_out);

    mu_out = log(L * exp(mu)) + 0.5 * ( sigma - sigma_out);
}

double evaluate_ln_pmf(gsl_integration_workspace *w, uint w_size, gsl_function *F, uint i, double delta) {

    double result, error;
    gsl_integration_qags(F, delta * i, delta * (i + 1), 0, 1e-4, w_size, w, &result, &error);
    return result;
}

// requries F to be initialized using log_ln_pdf (as defined by diff entropy)
double differential_entropy_ln(gsl_integration_workspace *w, uint w_size, struct ln_params params) {

    gsl_function F;
    F.function = &log_ln_pdf;
    F.params = &params;

    double result, error;

    //  integral from (0, +infty)
    gsl_integration_qagiu(&F, 0, 0, 1e-7, w_size, w, &result, &error);
    return (-1.0) * result;
}

double differential_entropy_ln_new(struct ln_params params) {

    return params.mu + (0.5) * log(2.0 * M_E * M_PI *  params.sigma);
}

double differential_entropy_uniform(gsl_integration_workspace *w, uint w_size, struct uniform_params params) {

    gsl_function F;
    F.function = &log_uniform_pdf;
    F.params = &params;
    double result, error;

    //  integral from (a, b)
    gsl_integration_qags(&F, params.a, params.b, 0, 1e-7, w_size, w, &result, &error);
    return (-1.0) * result;
}

void test_gsl() {
    // int w_size = 1000;
    // gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);

    // struct ln_params params = {0.0, 0.25, 0.0};

    // gsl_function F;
    // F.function = &ln_pdf;
    // F.params = &params;

    // for (int i = 3; i < 11; i++) {
    //     test_sln(1 << i);
    // }

    // test_root_solver();
}

void test_root_solver() {
    int max_iter = 100;
    // double x_lo = 1.0;
    double rel_error = 0.000001;
    double epsilon = 0.0001; // where prob is functinoally zero
    double mu = 0.0;
    double sigma = 0.5;

    double x_hi = 1000.0;

    double r = ln_root_solver(mu, sigma, epsilon, rel_error, x_hi, max_iter);
    // r = ln_root_solver_derivative(mu, sigma, epsilon, rel_error, x_hi, max_iter);
    printf("r = %f\n", r);
    // return status;
}

// returns the root where the pdf(root) < epsilon (functionally zero probability)
// only needs to done once in the beginning of a computation
// upper bound represents "infinity"
double ln_root_solver(double mu, double sigma, double epsilon, double rel_error, double upper_bound, int max_iter) {
    int status;
    int iter = 0;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 0;
    double x_lo = exp(mu - sigma)+0.1;
    double x_hi = upper_bound;
    gsl_function F;
    struct ln_params params = {mu, sigma, epsilon};

    F.function = &ln_pdf_eps;
    F.params = &params;
    // algorithms
    // T = gsl_root_fsolver_brent;
    T = gsl_root_fsolver_falsepos;
    // T = gsl_root_fsolver_bisection;
    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    // printf("using %s method\n", gsl_root_fsolver_name(s));

    // printf("%5s [%9s, %9s] %9s %9s\n",
    //        "iter", "lower", "upper", "root", "err(est)");

    do {
        iter++;
        status = gsl_root_fsolver_iterate(s);
        r = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi, 0, rel_error);

        // printf("%5d [%.7f, %.7f] %.7f %.7f ",
        //        iter, x_lo, x_hi,
        //        r,
        //        x_hi - x_lo);
        // if (status == GSL_SUCCESS)
        //     printf(" <-- Converged\n");
        // else
        //     printf("\n");

    } while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free(s);

    // printf("(zero) f_of_x = %.20f \n", ln_pdf_eps(r, &params));

    return r;
}

// not good for this function since the derivative of pdf can be zero
double ln_root_solver_derivative(double mu, double sigma, double epsilon, double rel_error, double upper_bound, int max_iter) {
    int status;
    int iter = 0;
    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    double r = 0;
    struct ln_params params = {mu, sigma, epsilon};
    double x = exp(mu - sigma) + 1000 * epsilon;
    // double x = 1.0;
    printf("starting x = %.7f\n", x);
    printf("(gsl) f_prime_of_x = %.20f \n", ln_pdf_deriv(r, &params));
    double x0;
    // double x_hi = upper_bound;

    gsl_function_fdf F;
    F.f = &ln_pdf_eps;
    F.df = &ln_pdf_deriv;
    F.fdf = &ln_pdf_deriv_fdf;
    F.params = &params;
    // T = gsl_root_fdfsolver_newton;
    T = gsl_root_fdfsolver_secant;
    // T = gsl_root_fdfsolver_steffenson;
    s = gsl_root_fdfsolver_alloc(T);
    gsl_root_fdfsolver_set(s, &F, x);

    printf("using %s method\n",
           gsl_root_fdfsolver_name(s));

    printf("%-5s %10s %10s %10s\n",
           "iter", "root", "err", "err(est)");
    do {
        iter++;
        status = gsl_root_fdfsolver_iterate(s);
        x0 = x;
        x = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, 0, rel_error);

        if (status == GSL_SUCCESS)
            printf("Converged:\n");

        printf("%5d %10.7f %+10.7f %10.7f\n",
               iter, x, x - 0, x - x0);
    } while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fdfsolver_free(s);
    return r;
}

void test_sln(int N, double mu, double sigma) {
    string experiment = "sum_lognorm";

    int max_iter = 100;
    // double r;
    // double x_lo = 1.0;
    const double rel_error = 0.000001;
    const double epsilon = 0.0001; // where prob is functinoally zero

    double mu_round, sigma_round, delta;
    double x_hi = 100000.0;

    uint16_t numTargets = 1;
    uint16_t numSpectators = 0;

    double H_T, H_S, H_T_S, awae; // shannon entropies
    double h_T, h_S, h_T_S;       // differential entropies
    vector<double> awae_results;
    vector<double> differential_awae_results;
    vector<double> H_T_results;
    vector<double> H_S_results;
    vector<double> H_T_S_results;
    vector<double> h_T_results;
    vector<double> h_S_results;
    vector<double> h_T_S_results;
    vector<double> target_init_entropy;
    vector<int> spectators;

    int w_size = 2000;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);

    // rightmost point of pdf where it is essentially zero
    double right_bound = 0.0;
    struct ln_params params = {mu, sigma, 0.0};
    struct ln_params round_params = {mu, sigma, 0.0};

    gsl_function F;
    F.function = &ln_pdf;

    int cutoff = 100;

    printf("%5s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s\n",
           "nSpec", "H_T", "H_S", "H_T_S", "awae", "h_T", "h_S", "h_T_S", "time(s)");
    for (int i = 1; i < cutoff; i++) {

        numSpectators = i;
        // cout << numSpectators << endl;
        auto start = std::chrono::system_clock::now();

        H_T = 0;
        H_S = 0;
        H_T_S = 0;
        awae = 0;
        F.params = &params;
        // printf("H_T params : %f, %f\n", params.mu, params.sigma);

        right_bound = ln_root_solver(mu, sigma, epsilon, rel_error, x_hi*params.sigma, max_iter);
        delta = right_bound / ((N - 1) + 1);
        // cout<<"delta = "<<delta<<", log2(delta) = "<<log2(delta)<<endl;
        for (int i = 0; i <= (N - 1); i++) {
            H_T += gFunc(evaluate_ln_pmf(w, w_size, &F, i, delta));
        }
        H_T = (-1.0) * H_T;
        h_T = differential_entropy_ln_new(params);
        H_T_results.push_back(H_T);
        h_T_results.push_back(h_T);

        update_fw_params(mu_round, sigma_round, mu, sigma, numSpectators);
        round_params.mu = mu_round;
        round_params.sigma = sigma_round;
        F.params = &round_params;

        right_bound = ln_root_solver(round_params.mu, round_params.sigma, epsilon, rel_error, x_hi*round_params.sigma, max_iter);
        delta = right_bound / (numSpectators * (N - 1) + 1);
        // cout<<"delta = "<<delta<<", log2(delta) = "<<log2(delta)<<endl;

        for (int i = 0; i <= (N - 1) * numSpectators; i++) {
            H_S += gFunc(evaluate_ln_pmf(w, w_size, &F, i, delta));
        }
        H_S = (-1.0) * H_S;
        h_S = differential_entropy_ln_new(round_params);
        H_S_results.push_back(H_S);
        h_S_results.push_back(h_S);

        update_fw_params(mu_round, sigma_round, mu, sigma, numSpectators + numTargets);
        round_params.mu = mu_round;
        round_params.sigma = sigma_round;
        F.params = &round_params;

        right_bound = ln_root_solver(round_params.mu, round_params.sigma, epsilon, rel_error, x_hi*round_params.sigma, max_iter);
        delta = right_bound / ((numSpectators + numTargets) * (N - 1) + 1);
        // cout<<"delta = "<<delta<<", log2(delta) = "<<log2(delta)<<endl;

        for (int i = 0; i <= (N - 1) * (numSpectators + numTargets); i++) {
            H_T_S += gFunc(evaluate_ln_pmf(w, w_size, &F, i, delta));
        }
        H_T_S = (-1.0) * H_T_S;
        h_T_S = differential_entropy_ln_new(round_params);
        H_T_S_results.push_back(H_T_S);
        h_T_S_results.push_back(h_T_S);

        awae = H_T + H_S - H_T_S;
        long double differential_awae = h_T + h_S - h_T_S;

        awae_results.push_back(awae);
        differential_awae_results.push_back(differential_awae);
        spectators.push_back(numSpectators);
        target_init_entropy.push_back(H_T);

        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        printf("%5d, %.7f, %.7f, %.7f, %.7f, %.7f  %.7f, %.7f, %.7f \n",
               i, H_T, H_S, H_T_S, awae,
               h_T, h_S, h_T_S,
               elapsed_seconds.count());
        // std::cout << i << ": "
        //           << "H_T = " << H_T << ", "
        //           << "H_S = " << H_S << ", "
        //           << "H_T_S = " << H_T_S << ", ";
        // cout  << "awae = " << awae << ", " << elapsed_seconds.count() << "s" << endl;
        // cout<<"-------------\n";
        // writeData_lnorm_differential(experiment,  spectators, h_T_results, h_S_results, h_T_S_results, differential_awae_results, mu, sigma, N);
        writeData_lnorm(experiment, awae_results, target_init_entropy, spectators, h_T_results, h_S_results, h_T_S_results, differential_awae_results, mu, sigma, N);
    }
}