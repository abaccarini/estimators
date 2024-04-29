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

k = 1
output_path = "../output_k" + str(k)+"/"
np.random.seed(0)

numIterations = 100
maxNumSpecs = 11
numT = 1
numA = 1
N = 5000

# used for conitnuous input distributions
step_size = 0.05

# the number of threads we want to use for the experiments
thread_count = 12 

# contains the mapping of the param strs (used as the filenames for the json data), which will be dumped to its own json later (to be used by plotting progrm so it knows all of the data available to plot)
param_str_dict = {}


class func:
    def __init__(self, fn, fn_name):
        self.fn = fn
        self.fn_name = fn_name


def var_mu(x):
    return np.asarray([np.var(x), np.mean(x)])


fn_max = func(np.max, "max")
fn_var = func(np.var, "var")
fn_median = func(np.median, "median")
fn_var_mu = func(var_mu, "var_mu")


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
    dir_path = output_path + str(fn.fn_name) + "/" + str(params.t) + "/"

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
        s = sampleData(params, N, numT, numSpecs, [xA], fn)
        # print("MI ", MI)
        MI += Mixed_KSG(s.x_T, s.O, k)
    return (xA, MI / float(numIterations))


def batch_ex_uniform_int(fn: func):

    dist_name = "uniform_int"
    N_vals = np.array([4, 8, 16])
    # N_vals = np.array([4])
    x_A_min = 0
    p_str_list = []
    for n in N_vals:
        spec_to_xA_to_MI = {}
        params = uniform_int_params(0, n)  # generates data from 0, 3-1
        target_init_entropy = calculateTargetInitEntropy(params)
        x_A_max = n
        p_str_list.append(params.p_str)
        x_A_range = range(x_A_min, x_A_max)

        for numSpecs in range(1, maxNumSpecs):
            print("uniform", fn.fn_name, n, numSpecs)
            pool = Pool(thread_count)
            all_args = [(params, numSpecs, xA, fn) for xA in x_A_range]
            results = pool.starmap(evaluate_estimator, all_args)
            xA_to_MI = dict(results)
            spec_to_xA_to_MI[numSpecs] = xA_to_MI
            # print(xA_to_MI)
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
    param_str_dict[dist_name] = p_str_list


def batch_ex_poisson(fn: func):

    dist_name = "poisson"
    lam_vals = np.array([2, 4, 8])
    x_A_min = 0
    p_str_list = []
    for lam in lam_vals:
        spec_to_xA_to_MI = {}
        params = poisson_params(lam)  # generates data from 0, 3-1
        p_str_list.append(params.p_str)
        target_init_entropy = calculateTargetInitEntropy(params)
        x_A_max = lam * 10

        x_A_range = range(x_A_min, x_A_max)

        for numSpecs in range(1, maxNumSpecs):
            print("poission", fn.fn_name, lam, numSpecs) # just used to track the status of experiments
            pool = Pool(thread_count)
            all_args = [(params, numSpecs, xA, fn) for xA in x_A_range]
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
    param_str_dict[dist_name] = p_str_list


def batch_ex_normal(fn: func):

    dist_name = "normal"
    mu = 0.0
    sigma_vals = np.array([1.0, 2.0, 4.0])

    p_str_list = []
    for sigma in sigma_vals:
        spec_to_xA_to_MI = {}
        params = normal_params(mu, sigma)  # generates data from 0, 3-1
        target_init_entropy = calculateTargetInitEntropy(params)
        p_str_list.append(params.p_str)

        x_A_min = -4.0 * sigma
        x_A_max = 4.0 * sigma

        x_A_range = np.linspace(
            x_A_min,
            x_A_max,
            num=int((x_A_max - x_A_min) / step_size),
        )

        for numSpecs in range(1, maxNumSpecs):
            print("normal", fn.fn_name, sigma, numSpecs)
            pool = Pool(thread_count)
            all_args = [(params, numSpecs, xA, fn) for xA in x_A_range]
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
    param_str_dict[dist_name] = p_str_list


def batch_ex_lognormal(fn: func):
    dist_name = "lognormal"
    sigma_vals = np.array([1.0, 2.0, 4.0])
    mu_vals = np.full((sigma_vals.size), 0.0)
    sigma_vals =  np.append(sigma_vals, 0.145542 )
    mu_vals =  np.append(mu_vals, 1.6702 )

    
    # sigma_vals = np.array([1.0])
    p_str_list = []

    x_A_min = 0.00001
    for sigma, mu in zip(sigma_vals, mu_vals):
        spec_to_xA_to_MI = {}
        params = lognormal_params(mu, sigma)  # generates data from 0, 3-1
        target_init_entropy = calculateTargetInitEntropy(params)
        p_str_list.append(params.p_str)

        # 0 is undefined for lognormal
        # therefore we start close to zero and go from there
        x_A_max = 4.0 * sigma

        x_A_range = np.linspace(
            x_A_min, x_A_max, num=int((x_A_max - x_A_min) / step_size)
        )

        for numSpecs in range(1, maxNumSpecs):
            print("lognormal", fn.fn_name, sigma, numSpecs)
            pool = Pool(thread_count)
            all_args = [(params, numSpecs, xA, fn) for xA in x_A_range]
            results = pool.starmap(evaluate_estimator, all_args)
            # print(results)
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
    param_str_dict[dist_name] = p_str_list


def main():
    pass


def normal_exp():
    batch_ex_normal(fn_max)
    batch_ex_normal(fn_var)
    batch_ex_normal(fn_median)
    batch_ex_normal(fn_var_mu)


def lognormal_exp():
    batch_ex_lognormal(fn_max)
    batch_ex_lognormal(fn_var)
    batch_ex_lognormal(fn_median)
    batch_ex_lognormal(fn_var_mu)


def uniform_exp():
    # batch_ex_uniform_int(fn_max)
    batch_ex_uniform_int(fn_var)
    # batch_ex_uniform_int(fn_median)
    # batch_ex_uniform_int(fn_var_mu)


def poisson_exp():
    batch_ex_poisson(fn_max)
    batch_ex_poisson(fn_var)
    batch_ex_poisson(fn_median)
    batch_ex_poisson(fn_var_mu)


def update_p_str_json():
    json_fname = output_path + "p_strs.json"

    my_file = Path(json_fname)
    if not my_file.is_file():  # the file does not exist, so just create and dump
        with open(json_fname, "w") as json_file:
            json.dump(param_str_dict, json_file, default=int, indent=2)
    else:
        # we need to open the existing version and compare with what was generated in this execution
        fname = open(json_fname)
        old = json.load(fname)
        # print(old)
        # print(param_str_dict)
        for key, value in param_str_dict.items():
            if (key not in old) or (sorted(value) != sorted(old[key])):
                old[key] = value

        with open(json_fname, "w") as json_file:
            json.dump(old, json_file, default=int, indent=2)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("missing distribution name argument, exiting...")
        exit(1)
    else:
        start_time = time.time()

        exp_name = sys.argv[1]
        if exp_name == "poisson":
            poisson_exp()
        elif exp_name == "lognormal":
            lognormal_exp()
        elif exp_name == "normal":
            normal_exp()
        elif exp_name == "uniform":
            uniform_exp()
        elif exp_name == "all":
            poisson_exp()
            uniform_exp()
            normal_exp()
            lognormal_exp()
        else:
            print("unknown distribution name provided (%s), exiting..."% (exp_name))
            exit(1)
        update_p_str_json()

        print(
            "finished computation at %s\nelapsed time: %s seconds "
            % (time.ctime(), time.time() - start_time)
        )
