# Notes

This contains all my thoughts about the implementation of the estimation technique itself

<!-- alt+c completes a task -->
To-do list:
- [ ] add JSON writing functionality for data output
- [ ] re-work the plotting library from `func_eval`
- [ ] finish the 1D kNN estimator
- [ ] look into the k-d tree implementation (higher dimensionality)


## GSL compatibility with k-NN (continuous)

- the only reason this is (potentially) needed is due to the middle term in the leakage computation for continuous:
$$
\begin{aligned}
h(X_T \mid X_A = x_A, O)  
         &= h(\vec{X}_T,O\mid \vec{X}_A=\vec{x}_A)-h(O\mid \vec{X}_A=\vec{x}_A)\\
         &=h(\vec{X}_T\mid \vec{X}_A=\vec{x}_A) + \underbrace{h(O\mid \vec{X}_T,\vec{X}_A=\vec{x}_A)} - h(O\mid \vec{X}_A=\vec{x}_A)
\end{aligned}
$$
- in the function `gsl_func`, the args are `x` and `params`
- propose using the estimator object *as* the params variable
- several things need to be added to the `knn_est` class in order for this to be feasible
  - first check that GSL will actually work as expected if using a `class` instead of a `struct`.
  - the function $f$ itself that we are evaluating
    - the "params" object (which is passed into the integrator) knows what function to evaluate
  - the number of iterations for the kNN estimator 
  - when calling `estimate()`, the value $x_A$ will be fixed (iterated through in a higher-up function which is performing the experiment)
    - therefore, it needs to be a member of the object as well
    - in subsequent experimental evaluations (different $x_A$), we need to update the member variable accordingly