import math
import numpy as np
from multiprocessing import Pool, cpu_count
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

thread_count = 14

EPSILON = 1e-6


def input_comb(input_domain, numParticipants):
    for c in product(input_domain, repeat=numParticipants):
        yield c


def max_func(x_T, x_A, x_S):
    return max(x_T + x_A + x_S)


def var_func(x_T, x_A, x_S):
    # print(x_T + x_A + x_S, np.var(x_T + x_A + x_S))
    # print(x_T )
    # print( x_A )
    # print(x_S)
    return np.var(x_T + x_A + x_S)


def std_func(x_T, x_A, x_S):
    return np.std(x_T + x_A + x_S)


def g_fn(x):
    return (-1.0) * x * log2(x) if x > EPSILON else 0.0


class func_eval_no_A:
    def __init__(self, fn, num_T, num_S, N):
        self.fn = fn
        self.num_T = num_T
        self.num_S = num_S
        self.N = N
        self.input_domain = range(N)
        # self.output_domain = range(N)  # only applies to MAX
        self.output_domain = self.generate_d_o()
        # print(self.output_domain)

    def generate_d_o(self):
        d_o = []
        # for x_A in input_comb(self.input_domain, self.num_A):
        for x_T in input_comb(self.input_domain, self.num_T):
            for x_S in input_comb(self.input_domain, self.num_S):
                d_o.append(self.fn(x_T, (), x_S))
        d_o_np = np.array(d_o)
        # return np.unique(d_o_np, axis=0)
        _, unique = np.unique(d_o_np.round(decimals=10), return_index=True)
        return d_o_np[unique]
        # return sorted(set(d_o))

    def singleCond_target(self, o, x_T):
        result = 0
        # for x_A in input_comb(self.input_domain, self.num_A):
        for x_S in input_comb(self.input_domain, self.num_S):
            if np.isclose(self.fn(x_T, (), x_S), o):
                # only applicable for uniform inputs
                result += pow(1.0 / self.N, self.num_S)
        return result

    def singleCond_of_o(self, o):
        result = 0
        for x_T in input_comb(self.input_domain, self.num_T):
            for x_S in input_comb(self.input_domain, self.num_S):
                if np.isclose(self.fn(x_T, (), x_S), o):
                    # only applicable for uniform inputs
                    result += pow(1.0 / self.N, self.num_T) * pow(
                        1.0 / self.N, self.num_S
                    )
        return result

    def dualCond(self, o, x_T):
        result = 0
        for x_S in input_comb(self.input_domain, self.num_S):
            if np.isclose(self.fn(x_T, (), x_S), o):
                # only applicable for uniform inputs
                result += pow(1.0 / self.N, self.num_S)
        return result

    def ent(self, o, outside_term):
        if outside_term < EPSILON:
            return 0.0
        else:
            result = 0.0
            for x_T in input_comb(self.input_domain, self.num_T):
                a = self.dualCond(o, x_T)
                b = self.singleCond_of_o(o)
                tmp = a * pow(1.0 / self.N, self.num_T) / b
                if b < EPSILON and a < EPSILON:
                    tmp = 0.0
                result += g_fn(tmp)
            return result

    def awae_no_A(self):
        tmp = 0
        result = 0
        for o in input_comb(self.output_domain, 1):
            # print(o)
            tmp = self.singleCond_of_o(o[0])
            result += tmp * self.ent(o[0], tmp)
        return result


class func_eval:
    def __init__(self, fn, num_T, num_A, num_S, N):
        self.fn = fn
        self.num_T = num_T
        self.num_A = num_A
        self.num_S = num_S
        self.N = N
        self.input_domain = range(N)
        # self.output_domain = range(N)  # only applies to MAX
        self.output_domain = self.generate_d_o()
        # print(self.output_domain)

    def generate_d_o(self):
        d_o = []
        for x_A in input_comb(self.input_domain, self.num_A):
            for x_T in input_comb(self.input_domain, self.num_T):
                for x_S in input_comb(self.input_domain, self.num_S):
                    d_o.append(self.fn(x_T, x_A, x_S))
        d_o_np = np.array(d_o)
        # return np.unique(d_o_np, axis=0)
        _, unique = np.unique(d_o_np.round(decimals=10), return_index=True)
        return d_o_np[unique]
        # return sorted(set(d_o))

    def singleCond_target(self, o, x_T):
        result = 0
        for x_A in input_comb(self.input_domain, self.num_A):
            for x_S in input_comb(self.input_domain, self.num_S):
                if np.isclose(self.fn(x_T, x_A, x_S), o):
                    # only applicable for uniform inputs
                    result += pow(1.0 / self.N, self.num_A) * pow(
                        1.0 / self.N, self.num_S
                    )
        return result

    def singleCond_attacker(self, o, x_A):
        result = 0
        for x_T in input_comb(self.input_domain, self.num_T):
            for x_S in input_comb(self.input_domain, self.num_S):
                if np.isclose(self.fn(x_T, x_A, x_S), o):
                    # only applicable for uniform inputs
                    result += pow(1.0 / self.N, self.num_T) * pow(
                        1.0 / self.N, self.num_S
                    )
        return result

    def dualCond(self, o, x_A, x_T):
        result = 0
        for x_S in input_comb(self.input_domain, self.num_S):
            if np.isclose(self.fn(x_T, x_A, x_S), o):
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
            result += tmp * self.ent(o[0], x_A, tmp)
        return result

    def awae_all(self):
        res = []
        pool = Pool(thread_count)

        all_args = [(xA,) for xA in input_comb(self.input_domain, self.num_A)]
        # print(all_args)
        # print(input_comb(self.input_domain, self.num_A))
        # all_x_A = input_comb(self.input_domain, self.num_A)
        results = pool.starmap(self.awae, all_args)
        # for x_A in input_comb(self.input_domain, self.num_A):
        #     print(x_A)
        #     res.append(self.awae(x_A))
        return results


def main_fn():
    num_T = 1
    num_A = 1
    num_S = 2

    N = 8
    input_domain = range(N)
    # print(input_domain)

    # eval_max = func_eval(max_func, num_T, num_A, num_S, N)
    # print("max : ", eval_max.awae_all())
    target_init = 2.0

    eval_var = func_eval(var_func, num_T, num_A, num_S, N)
    awaes = eval_var.awae_all()
    abs_loss = [target_init - x for x in awaes]
    # print("var : ", abs_loss)
    print("var : ", awaes)
    eval_var = func_eval_no_A(var_func, num_T, num_S, N)
    awae_no_a = eval_var.awae_no_A()
    print("no A : ", awae_no_a)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main_fn()
