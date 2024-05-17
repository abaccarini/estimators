import math
import numpy as np
import os
import sys
import matplotlib as mpl
from itertools import (
    combinations,
    combinations_with_replacement,
    product,
    permutations,
    chain,
)
from math import comb, log2
from NPEET.npeet import entropy_estimators as ee


EPSILON = 1e-6


def input_comb(input_domain, numParticipants):
    for c in product(input_domain, repeat=numParticipants):
        yield c


def max_func(x_T, x_A, x_S):
    return max(x_T + x_A + x_S)

def g_fn(x):
    return (-1.0) * x * log2(x) if x > EPSILON else 0.0


class func_eval:
    def __init__(self, fn, num_T, num_A, num_S, N):
        self.fn = fn
        self.num_T = num_T
        self.num_A = num_A
        self.num_S = num_S
        self.N = N
        self.input_domain = range(N)
        self.output_domain = range(N) # only applies to MAX

    def singleCond_target(self, o, x_T):
        result = 0
        for x_A in input_comb(self.input_domain, self.num_A):
            for x_S in input_comb(self.input_domain, self.num_S):
                if self.fn(x_T, x_A, x_S) == o:
                    # only applicable for uniform inputs
                    result += pow(1.0 / self.N, self.num_A) * pow(1.0 / self.N, self.num_S)
        return result

    def singleCond_attacker(self, o, x_A):
        result = 0
        for x_T in input_comb(self.input_domain, self.num_T):
            for x_S in input_comb(self.input_domain, self.num_S):
                if self.fn(x_T, x_A, x_S) == o:
                    # only applicable for uniform inputs
                    result += pow(1.0 / self.N, self.num_T) * pow(1.0 / self.N, self.num_S)
        return result

    def dualCond(self, o, x_A, x_T):
        result = 0
        for x_S in input_comb(self.input_domain, self.num_S):
            if self.fn(x_T, x_A, x_S) == o:
                # only applicable for uniform inputs
                result += pow(1.0 / self.N, self.num_S)
        return result

    def ent(self, o, x_A, outside_term):
        if outside_term < EPSILON:
            return 0.0
        else:
            result = 0.0
            for x_T in input_comb(self.input_domain, self.num_T):
                a = self.dualCond(o, x_A, x_T)
                b = self.singleCond_attacker(o, x_A)
                tmp = a * pow(1.0 / self.N, self.num_T) / b
                if b < EPSILON and a < EPSILON:
                    tmp = 0.0
                result += g_fn(tmp)
            return result

    def awae(self, x_A):
        tmp = 0
        result = 0
        for o in input_comb(self.output_domain, 1):
            # print(o)
            tmp = self.singleCond_attacker(o[0], x_A)
            result += tmp*self.ent(o[0], x_A, tmp)
        return result
    def awae_all(self):
        res = []
        for x_A in input_comb(self.input_domain, self.num_A):
            print(x_A)
            res.append(self.awae(x_A))
        return res
            
def main_fn():
    num_T = 1
    num_A = 1
    num_S = 6

    N = 8
    input_domain = range(N)
    print(input_domain)

    eval_max = func_eval(max_func,num_T, num_A, num_S, N)
    print(eval_max.awae_all())
    
    

if __name__ == "__main__":
    if len(sys.argv) == 1:      
        main_fn()