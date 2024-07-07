import numpy as np
import math
import scipy

# import matplotlib as mpl
# import matplotlib.pyplot as plt

from scipy.stats import binom, poisson

from numpy import random


x_t = 7
x_a = 6

# print(str(x_t) + " > " + str(x_a) + " --> " + str(int(x_t > x_a)))
# print(int(x_t > x_a))

# for x_t in range(0,8):
#     for x_a in range(0,8):
#         print(int(x_t > x_a), end =" & ")
#     print()


# def twae(k, n):
#     sum1 = 0.0
#     sum2 = 0.0
#     for i in range(9-k,8):
#         sum1+=math.log2(1/i)
#     for i in range(k, 9):
#         sum2+=math.log2(1/i)
#     return (1.0/n) * ( (-1)*sum1 - sum2)

# print(twae(1,8))
# print(twae(2,8))


# x = random.binomial(n=10, p=0.5, size=10)
# # print(x)
# n = 8
p = 0.5

# lam = 4
# # r_values = list(range(n))
# print(r_values)
# # poisson.pmf(x, mu)
# # dist = [binom.pmf(r, n-1, p) for r in r_values ]
# dist = [poisson.pmf(r, lam) for r in r_values]

# print(dist)

# print(len(dist))


numSpec = 1
numTargets = 1
numAttackers = 1


def H_T(nT, n, lam):
    xx = 0
    for i in range(0, n):
        temp = my_poisson(i, lam, nT)
        xx += temp * math.log2(temp)
    return (-1.0) * xx


def H_S(nS, n, lam):
    xx = 0
    for i in range(0, n * nS):
        temp = my_poisson(i, lam, nS)
        xx += temp * math.log2(temp)
    return (-1.0) * xx


def H_T_S(nT_S, n, lam):
    xx = 0
    for i in range(0, n * nT_S):
        temp = my_poisson(i, lam, nT_S)
        xx += temp * math.log2(temp)
    return (-1.0) * xx


def my_poisson(x, lam, n):
    return poisson.pmf(x, lam) * np.exp(scipy.special.xlogy(x, n) - (n * lam) + lam)


# new_awae = H_T(numTargets) + H_S(numSpec) - H_T_S(numTargets + numSpec)
# print("new_awae = ", new_awae)
