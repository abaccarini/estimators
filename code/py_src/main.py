from pathlib import Path
from scipy.special import xlogy
from scipy.stats import poisson
import sys
import json
import os
import time
import numpy as np
from mixed_ksg import Mixed_KSG
from dist_params import *
from datetime import datetime
from multiprocessing import Pool, freeze_support, cpu_count

np.random.seed(0)

numIterations = 10
maxNumSpecs = 11
numT = 1
numA = 1
N = 3000
k = 1

# used for conitnuous input distributions
step_size = 0.05


class sampleData:
    def __init__(self, params, N, numT, numS, x_A, thefunc):
        self.params = params  # the distribution params
        self.N = N
        self.x_A = (
            x_A  # vector of attacker inputs (fixed, but can be changed if needed )
        )
        self.x_T = np.asarray(self.sample(numT))
        self.x_S = np.asarray(self.sample(numS))
        self.thefunc = thefunc
        self.O = self.produceOutputs()

    def sample(self, numP):
        if self.params.t == "uniform_int":
            return np.reshape(
                np.random.randint(self.params.a, self.params.b, self.N * numP),
                (self.N, numP),
            )
        if self.params.t == "uniform_real":
            return np.reshape(
                np.random.uniform(self.params.a, self.params.b, self.N * numP),
                (self.N, numP),
            )
        if self.params.t == "normal":
            return np.reshape(
                np.random.normal(self.params.mu, self.params.sigma, self.N * numP),
                (self.N, numP),
            )
        if self.params.t == "lognormal":
            return np.reshape(
                np.random.lognormal(self.params.mu, self.params.sigma, self.N * numP),
                (self.N, numP),
            )
        if self.params.t == "poisson":
            return np.reshape(
                np.random.poisson(self.params.lam, self.N * numP), (self.N, numP)
            )
        print("unknown distribution encountered: %s" % (self.params.t))
        exit()

    def produceOutputs(self):
        return np.asarray(
            [
                (self.thefunc.fn(np.concatenate((xts, self.x_A))))
                for (xts) in np.column_stack((self.x_T, self.x_S))
            ]
        )

    # updates x_A, then re-generateates O samples automatically
    def update_xA(self, x_A):
        self.x_A = x_A
        self.O = self.produceOutputs()


class func:
    def __init__(self, fn, fn_name):
        self.fn = fn
        self.fn_name = fn_name


def write_json(
    numIterations, params, N, numT, numA, target_init_entropy, MI_data, fn: func
):
    dt = datetime.now()
    data = {
        "name": fn.fn_name,
        "dist": params.getJSON(),
        "num_samples": N,
        "timestamp": str(dt),
        "num_T": numT,
        "num_A": numA,
        "num_iterations": numIterations,
        "target_init_entropy": target_init_entropy,
        "awae_data": MI_data,
    }
    # print(data)
    dir_path = "../output_py/" + str(fn.fn_name) + "/" + str(params.t) + "/"

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    pstr = params.getJSON()["param_str"]
    # print(params.getJSON())
    fname = dir_path + pstr + ".json"
    with open(fname, "w") as json_file:
        json.dump(data, json_file, default=int, indent=2)


def calculateTargetInitEntropy(dist_params):
    if isinstance(dist_params, uniform_real_params):
        return np.log2(float(dist_params.b) - float(dist_params.a - 1))

    if isinstance(dist_params, uniform_int_params):
        return np.log2(float(dist_params.b) - float(dist_params.a))

    if isinstance(dist_params, normal_params):
        return 0.5 * np.log2(2.0 * np.pi * np.e * dist_params.sigma)

    if isinstance(dist_params, poisson_params):
        if dist_params.lam > 10:
            return 0.5 * np.log2(2.0 * np.pi * np.e * dist_params.lam)
        else:
            accum = 0.0
            for i in range(0, 10 * dist_params.lam):
                tmp = poisson.pmf(i, dist_params.lam)
                accum += xlogy(tmp, tmp)
            return (-1.0) * accum

    if isinstance(dist_params, lognormal_params):
        return (
            dist_params.mu + 0.5 * np.log(2.0 * np.pi * np.e * dist_params.sigma)
        ) / np.log(2.0)

    print("unknown distribution encountered: %s" % (dist_params.t))
    exit()


# used for discrete and continuous data
def evaluate_estimator(params, numSpecs, xA, fn):
    MI = 0.0
    for i in range(numIterations):
        s = sampleData(params, N, numT, numSpecs, xA, fn)
        MI += Mixed_KSG(s.x_T, s.O, k)
    return (xA, MI / float(numIterations))


def batch_ex_uniform_int(fn: func):

    # N_vals = np.array([4])
    N_vals = np.array([4, 8, 16])
    x_A_min = 0
    for n in N_vals:
        spec_to_xA_to_MI = {}
        params = uniform_int_params(0, n)  # generates data from 0, 3-1
        target_init_entropy = calculateTargetInitEntropy(params)
        x_A_max = n

        x_A_range = range(x_A_min, x_A_max)

        for numSpecs in range(1, maxNumSpecs):
            print("uniform", fn.fn_name, n, numSpecs)
            pool = Pool(int(cpu_count() / 2))
            all_args = [(params, numSpecs, [xA], fn) for xA in x_A_range]
            results = pool.starmap(evaluate_estimator, all_args)
            xA_to_MI = dict(results)
            spec_to_xA_to_MI[numSpecs] = xA_to_MI

        write_json(
            numIterations,
            params,
            N,
            numT,
            numA,
            target_init_entropy,
            spec_to_xA_to_MI,
            fn,
        )


def batch_ex_poisson(fn: func):

    # N_vals = np.array([4])
    lam_vals = np.array([2, 4, 8])
    for lam in lam_vals:
        spec_to_xA_to_MI = {}
        params = poisson_params(lam)  # generates data from 0, 3-1
        target_init_entropy = calculateTargetInitEntropy(params)
        x_A_min = 0
        x_A_max = lam * 10

        x_A_range = range(x_A_min, x_A_max)

        for numSpecs in range(1, maxNumSpecs):
            print("poission", fn.fn_name, lam, numSpecs)
            pool = Pool(int(cpu_count() / 2))
            all_args = [(params, numSpecs, [xA], fn) for xA in x_A_range]
            results = pool.starmap(evaluate_estimator, all_args)
            xA_to_MI = dict(results)
            spec_to_xA_to_MI[numSpecs] = xA_to_MI

        write_json(
            numIterations,
            params,
            N,
            numT,
            numA,
            target_init_entropy,
            spec_to_xA_to_MI,
            fn,
        )


def batch_ex_normal(fn: func):

    # N_vals = np.array([4])
    mu = 0.0
    sigma_vals = np.array([1.0, 2.0, 4.0])

    for sigma in sigma_vals:
        spec_to_xA_to_MI = {}
        params = normal_params(mu, sigma)  # generates data from 0, 3-1
        target_init_entropy = calculateTargetInitEntropy(params)

        x_A_min = -4.0 * sigma
        x_A_max = 4.0 * sigma

        x_A_range = np.linspace(
            x_A_min,
            x_A_max,
            num=int((x_A_max - x_A_min) / step_size),
            # x_A_min, x_A_max, num=50
        )

        for numSpecs in range(1, maxNumSpecs):
            print("normal", fn.fn_name, sigma, numSpecs)
            pool = Pool(int(cpu_count() / 2))
            # pool = Pool(20)
            all_args = [(params, numSpecs, [xA], fn) for xA in x_A_range]
            results = pool.starmap(evaluate_estimator, all_args)
            xA_to_MI = dict(results)
            spec_to_xA_to_MI[numSpecs] = xA_to_MI

        write_json(
            numIterations,
            params,
            N,
            numT,
            numA,
            target_init_entropy,
            spec_to_xA_to_MI,
            fn,
        )


def batch_ex_lognormal(fn: func):

    # N_vals = np.array([4])
    mu = 0.0
    sigma_vals = np.array([1.0, 2.0, 4.0])
    # sigma_vals = np.array([1.0])

    for sigma in sigma_vals:
        spec_to_xA_to_MI = {}
        params = lognormal_params(mu, sigma)  # generates data from 0, 3-1
        target_init_entropy = calculateTargetInitEntropy(params)

        # 0 is undefined for lognormal
        # therefore we start close to zero and go from there
        x_A_min = 0.00001
        x_A_max = 4.0 * sigma

        x_A_range = np.linspace(
            x_A_min, x_A_max, num=int((x_A_max - x_A_min) / step_size)
        )

        for numSpecs in range(1, maxNumSpecs):
            print("lognormal", fn.fn_name, sigma, numSpecs)
            pool = Pool(int(cpu_count() / 2))
            all_args = [(params, numSpecs, [xA], fn) for xA in x_A_range]
            results = pool.starmap(evaluate_estimator, all_args)
            xA_to_MI = dict(results)
            spec_to_xA_to_MI[numSpecs] = xA_to_MI

        write_json(
            numIterations,
            params,
            N,
            numT,
            numA,
            target_init_entropy,
            spec_to_xA_to_MI,
            fn,
        )


def main():

    # params = uniform_int_params(0, 4)  # generates data from 0, 3-1
    # print("target init entropy : ", calculateTargetInitEntropy(params))

    # def fn(x):
    #     # return np.asarray([sum(x), sum(x)])
    #     return np.asarray([sum(x)])

    # fn = func(np.max, "max")
    # batch_ex_lognormal(fn)
    # batch_ex_normal(fn)
    # # batch_ex_uniform_int(fn)
    # # batch_ex_poisson(fn)

    # fn = func(np.var, "var")
    # batch_ex_lognormal(fn)
    # batch_ex_normal(fn)
    # # batch_ex_uniform_int(fn)
    # # batch_ex_poisson(fn)

    fn = func(np.median, "median")
    batch_ex_lognormal(fn)
    batch_ex_normal(fn)
    # batch_ex_uniform_int(fn)
    # batch_ex_poisson(fn)

    def var_mu(x):
        return np.asarray([np.var(x), np.mean(x)])

    fn = func(var_mu, "var_mu")
    batch_ex_lognormal(fn)
    batch_ex_normal(fn)
    # batch_ex_uniform_int(fn)
    # batch_ex_poisson(fn)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        start_time = time.time()
        main()
        print(
            "finished computation at %s\nelapsed time: %s seconds "
            % (time.ctime(), time.time() - start_time)
        )
