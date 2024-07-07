#include "normal.hpp"

double normal_pdf(double x, void *p) {
    normal_params *params = (normal_params *)p;
    return (1 / ( sqrt((params->sigma) *2 * M_PI))) * exp((-1) * (x - (params->mu)) * (x - (params->mu)) / (2 *  (params->sigma)));
}

double normal_pdf_eps(double x, void *p) {
    normal_params *params = (normal_params *)p;
    return (1 / (sqrt((params->sigma) *2 * M_PI))) * exp((-1) * (x - (params->mu)) * (x - (params->mu)) / (2 *  (params->sigma))) - (params->eps);
}

// computes f(x) log f(X)
double log_normal_pdf(double x, void *p) {
    // struct ln_params *params = (struct ln_params *)p;
    // by definition (according to us)
    if (normal_pdf(x, p) < EPSILON) {
        return 0.0;
    } else {
        return normal_pdf(x, p) * log2(normal_pdf(x, p));
    }
}

double evaluate_normal_pmf(gsl_integration_workspace *w, uint w_size, gsl_function *F, uint i, double delta, double left_bound) {

    double result, error;
    // cout<<"["<<left_bound + delta * i<<", "<<left_bound + delta * (i+1)<<"] = ";

    double a = left_bound + delta * i;
    double b = left_bound + delta * (i + 1);

    gsl_integration_qags(F, a, b, 0, 1e-7, w_size, w, &result, &error);
    // gsl_integration_qag(F, left_bound + delta * i, left_bound + delta * (i + 1), 0, 1e-7, 6, w_size, w, &result, &error);
    // DO NOT USE NON SINGULATIY VERSION
    // gsl_integration_qag(F, left_bound + delta * i, left_bound + delta * (i + 1), 0, 1e-7, 1, w_size, w, &result, &error);

    // printf("[%.7f, %.7f] = %.7f ", a,b, result);
    return result;
}

double normal_root_solver(double mu, double sigma, double epsilon, double rel_error, double upper_bound, int max_iter) {
    int status;
    int iter = 0;
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    double r = 0;
    double x_lo = 0.1;
    double x_hi = upper_bound;
    gsl_function F;
    normal_params params = {mu, sigma, epsilon};

    F.function = &normal_pdf_eps;
    F.params = &params;
    // algorithms
    // T = gsl_root_fsolver_brent;
    T = gsl_root_fsolver_falsepos;
    // T = gsl_root_fsolver_bisection;
    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    do {
        iter++;
        status = gsl_root_fsolver_iterate(s);
        r = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi, 0, rel_error);

    } while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free(s);

    return r;
}

// requries F to be initialized using log_ln_pdf (as defined by diff entropy)
double differential_entropy_normal(gsl_integration_workspace *w, uint w_size, normal_params params) {

    gsl_function F;
    F.function = &log_normal_pdf;
    F.params = &params;

    double result, error;
    //  integral from (0, +infty)
    gsl_integration_qagi(&F,  0, 1e-7, w_size, w, &result, &error);
    return (-1.0) * result;
}

double differential_entropy_normal_new(normal_params params) {
    // return log(sqrt(params.sigma * 2.0 * M_PI * M_E)) ;
    return 0.5*log2(params.sigma * 2.0 * M_PI * M_E) ;
}


void test_normal(int N, double mu, double sigma) {
    string experiment = "sum_normal";

    // const int num_stdevs = 5; // number of standard deviations on each side of f(x) that are considered nonzero
    double x_hi = 1000.0;
    const double rel_error = 1e-6;
    const double epsilon = 1e-25; // where prob is functinoally zero
    int max_iter = 500;

    double delta;

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

    int w_size = 5000;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(w_size);

    // rightmost point of pdf where it is essentially zero
    double right_bound = 0.0;
    // double left_bound = 0.0;
    normal_params params = {mu, sigma, 0.0};
    normal_params round_params = {mu, sigma, 0.0};

    gsl_function F;
    F.function = &normal_pdf;

    // int cutoff = 5;
    int cutoff = 100;

    printf("%5s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s, %9s\n",
           "nSpec", "H_T", "H_S", "H_T_S", "awae", "delta", "h_T", "h_S", "h_T_S", "time(s)");
    for (int i = 1; i < cutoff; i++) {

        numSpectators = i;
        // cout << numSpectators << endl;
        auto start = std::chrono::system_clock::now();

        H_T = 0;
        H_S = 0;
        H_T_S = 0;
        awae = 0;
        F.params = &params;

        right_bound = normal_root_solver(mu, sigma, epsilon, rel_error, 100*params.sigma, max_iter);
        delta = (2 * right_bound) / ((N - 1) + 1);

        // right_bound = num_stdevs * params.sigma;
        // left_bound = (-1.0) * num_stdevs * params.sigma;
        // delta = (right_bound - left_bound) / ((N - 1) + 1);

        //  cout <<  "f("<<right_bound<<") = " << normal_pdf(right_bound, &params)<<", delta = " << delta <<  ", log(delta) = " << log(delta) << ", mu =  " << params.mu << ", sigma =  " << params.sigma << endl;
        for (int i = 0; i <= (N - 1); i++) {
            H_T += gFunc(evaluate_normal_pmf(w, w_size, &F, i, delta, right_bound * (-1.0)));
        }
        H_T = (-1.0) * H_T;
        h_T = differential_entropy_normal_new(params);
        H_T_results.push_back(H_T);
        h_T_results.push_back(h_T);

        round_params.mu = numSpectators * mu;
        round_params.sigma = (numSpectators * sigma);

        F.params = &round_params;

        right_bound = normal_root_solver(round_params.mu, round_params.sigma, epsilon, rel_error, 100*round_params.sigma, max_iter);
        delta = (2 * right_bound) / ((numSpectators) * (N - 1) + 1);

        // right_bound = num_stdevs * round_params.sigma;
        // left_bound = (-1.0) * num_stdevs * round_params.sigma;
        // delta = (right_bound - left_bound) / ((N - 1) + 1);

        // cout << "delta = " << delta << ", mu =  " << round_params.mu << ", sigma =  " << round_params.sigma << endl;
        // cout <<  "f("<<right_bound<<") = " << normal_pdf(right_bound, &round_params)<< ", delta = " << delta <<  ", log(delta) = " << log(delta) << ", mu =  " << round_params.mu << ", sigma =  " << round_params.sigma << endl;

        for (int i = 0; i <= (N - 1) * numSpectators; i++) {
            H_S += gFunc(evaluate_normal_pmf(w, w_size, &F, i, delta, right_bound * (-1.0)));
        }
        H_S = (-1.0) * H_S;
        h_S = differential_entropy_normal_new(round_params);
        H_S_results.push_back(H_S);
        h_S_results.push_back(h_S);

        round_params.mu = mu * (numSpectators + numTargets);
        round_params.sigma = ((numSpectators + numTargets) * sigma);

        F.params = &round_params;

        right_bound = normal_root_solver(round_params.mu, round_params.sigma, epsilon, rel_error, 100*round_params.sigma, max_iter);
        delta = (2 * right_bound) / ((numSpectators + numTargets) * (N - 1) + 1);

        // right_bound = num_stdevs * round_params.sigma;
        // left_bound = (-1.0) * num_stdevs * round_params.sigma;
        // delta = (right_bound - left_bound) / (N - 1);

        // cout << "delta = " << delta << ", mu =  " << round_params.mu << ", sigma =  " << round_params.sigma << endl;
         cout <<  "f("<<right_bound<<") = " << normal_pdf(right_bound, &round_params)<<", delta = " << delta <<  ", log(delta) = " << log(delta) << ", mu =  " << round_params.mu << ", sigma =  " << round_params.sigma << endl;

        for (int i = 0; i <= (N - 1) * (numSpectators + numTargets); i++) {
            H_T_S += gFunc(evaluate_normal_pmf(w, w_size, &F, i, delta, right_bound * (-1.0)));
        }
        H_T_S = (-1.0) * H_T_S;
        h_T_S = differential_entropy_normal_new(round_params);
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
        printf("%5d, %.7f, %.7f, %.7f, %.7f, %.7f,  %.7f, %.7f, %.7f, %.7f \n",
               i, H_T, H_S, H_T_S, awae, H_T - awae,
               h_T, h_S, h_T_S,
               elapsed_seconds.count());

        // writeData_lnorm_differential(experiment,  spectators, h_T_results, h_S_results, h_T_S_results, differential_awae_results, mu, sigma, N);
        writeData_lnorm(experiment, awae_results, target_init_entropy, spectators, h_T_results, h_S_results, h_T_S_results, differential_awae_results,mu, sigma, N);
    }
}