import math
import numpy as np
from estimator_exp import Mixed_KSG
from dist_params import *


numIterations = 1
maxNumSpecs = 11
minNumSpecs = 1
numT = 1
numA = 1
N = 10

arr = [0, 1, 4, 3, 6, 1]
print(np.var(arr))