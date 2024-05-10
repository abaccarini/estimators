from estimator_exp import calculateTargetInitEntropy
from dist_params import *

lam = 8

params = poisson_params(lam)
print(params.lam)


entropy = calculateTargetInitEntropy(params)
print(entropy)
