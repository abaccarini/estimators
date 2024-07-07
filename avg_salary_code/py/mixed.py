import math
import os 

groups = {
    "B": (0.25, 0, 3),
    "C": (0.5, 2, 6),
    "D": (0.25, 4, 8)
}

def compute_pmf(x, groups):
    accum = 0
    for g,params in groups.items():
        n = params[2] - params[1] + 1
        if x in range(params[1],params[2]+1):
            accum += params[0]*(1/n)
    return accum



print(sum([compute_pmf(x, groups) for x in range(0,9)]))