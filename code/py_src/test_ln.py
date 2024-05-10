from pathlib import Path
from scipy.special import xlogy
from scipy.stats import poisson
import sys
import json
import time
import numpy as np
from mixed_ksg import Mixed_KSG
from dist_params import *
from datetime import datetime
from multiprocessing import Pool, cpu_count
from matplotlib import pyplot as plt

mu = 1.6702
sigma = 0.3815


# np.pi * np.e
def pdf(x):
    return (1.0 / (sigma * x * np.sqrt(2.0 * np.pi))) * np.exp(-1.0*(np.log(x) - mu) *(np.log(x) - mu) / (2.0 * sigma * sigma) )

print(pdf(5.0))
xs = np.linspace(0.0001, 15, 10000)
ys = list(map(pdf, xs))
print(max(ys))

plt.plot(xs, ys)

s = np.random.lognormal(mu, sigma, 10000000)

count, bins, ignored = plt.hist(s, 10000, density=True, align='mid')
plt.savefig(
"test.pdf", 
bbox_inches="tight",
)