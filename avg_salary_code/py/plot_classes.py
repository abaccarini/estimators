import logging
import os
import re
import subprocess
import sys
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, figure, rc
from matplotlib.backends.backend_pdf import PdfPages
from natsort import natsorted
from numpy import genfromtxt


joint_col_names = [
    "target.mu",
    "target.sigma",
    "shared_params.mu",
    "shared_params.sigma",
    "ex1_params.mu",
    "ex1_params.sigma",
    "ex2_params.mu",
    "ex2_params.sigma",
    "num_shared",
    "num_spec_1",
    "num_spec_2",
    "diff_ent_joint_T_exps",
    "diff_ent_joint_exps",
    "diff_ent_cond",
    "target_init",
    "awae",
]

joint_col_names = [
    "target.mu",
    "target.sigma",
    "shared_params.mu",
    "shared_params.sigma",
    "ex1_params.mu",
    "ex1_params.sigma",
    "ex2_params.mu",
    "ex2_params.sigma",
    "num_shared",
    "num_spec_1",
    "num_spec_2",
    "diff_ent_joint_T_exps",
    "diff_ent_joint_exps",
    "diff_ent_cond",
    "target_init",
    "awae",
]

mixed_cols = [
    "target_params_mu",
    "target_params_sigma",
    "spec_params_mu_1",
    "spec_params_sigma_1",
    "spec_params_mu_2",
    "spec_params_sigma_2",
    "spec_params_mu_3",
    "spec_params_sigma_3",
    "num_targets",
    "num_spec_1",
    "num_spec_2",
    "num_spec_3",
    "h_T",
    "h_S",
    "h_T_S",
    "awae_differential",
]

case2cols = [
    "num_spec",
    "sigma_1",
    "sigma_2",
    "sigma_2",
    "h_X_T",
    "h_X_S_T",
    "h_X_S",
    "abs_loss",
]


continous_col_names = [
    "target_params_mu",
    "target_params_sigma",
    "spec_params_mu",
    "spec_params_sigma",
    "num_targets",
    "num_spec",
    "H_T",
    "H_S",
    "H_T_S",
    "delta_T",
    "delta_S",
    "delta_T_S",
    "awae_shannon",
    "h_T",
    "h_S",
    "h_T_S",
    "awae_differential",
]

gamma_col_names = [
    "target_params_mu",
    "target_params_sigma",
    "spec_params_mu",
    "spec_params_sigma",
    "num_targets",
    "num_spec",
    "h_T",
    "h_S",
    "h_T_S",
    "awae_differential",
]


continous_col_names_est = [
    "target_params_mu",
    "target_params_sigma",
    "spec_params_mu",
    "spec_params_sigma",
    "num_targets",
    "num_spec",
    "numSamples",
    "numIterations",
    "k",
    "h_T",
    "h_S",
    "h_T_S",
    "awae_differential",
]
discrete_col_names = ["spectators", "awae_results", "target_init_entropy"]


continuous_3xp_cols = [
    "sigma",
    "S_1",
    "S_2",
    "S_3",
    "S_12",
    "S_13",
    "S_23",
    "S_123",
    "T",
    "t_1",
    "t_2",
    "t_3",
    "H_X_T",
    "H_X_T_O_1",
    "H_X_T_O_12",
    "H_X_T_O_123",
]


continuous_3xp_cols_brute_force = [
    "sigma",
    "S_1",
    "S_2",
    "S_3",
    "S_12",
    "S_13",
    "S_23",
    "S_123",
    "T",
    "H_X_T",
    "H_X_T_O_1",
    "H_X_T_O_12_t11",
    "H_X_T_O_12_t10",
    "H_X_T_O_123_t111",
    "H_X_T_O_123_t110",
    "H_X_T_O_123_t101",
    "H_X_T_O_123_t011",
    "H_X_T_O_123_t100",
    "H_X_T_O_123_t010",
    "H_X_T_O_123_t001",
]


def parse_discrete_param_string(param_string):
    param_list = param_string.split("_")
    if len(param_list) == 1:
        # uniform
        return int(param_list[0]), None
    elif len(param_list) == 2:
        # poisson
        return int(param_list[0]), (float(param_list[1]))
    else:
        raise Exception("Input string incorrect format", param_string)


def parse_continuous_param_string(exp_string, param_string):
    param_list = param_string.split("_")
    exp_list = exp_string.split("_")
    # uniform
    return (
        exp_list[0],
        exp_list[1],
        int(exp_list[2][1:]),
        int(param_list[0]),
        float(param_list[1]),
    )


class discrete_params:
    def __init__(self, N=0, lam=None):
        self.N = N
        self.lam = lam

    def __eq__(self, other):
        if self.lam is not None:
            return other.N == self.N and math.isclose(other.lam, self.lam)
        else:
            return other.N == self.N

    def __hash__(self):
        return id(self)


class continuous_params_est:
    def __init__(self, exp_tag, mu, sigma):
        self.exp_tag = exp_tag
        self.mu = mu
        self.sigma = sigma

    def __eq__(self, other):
        return (
            other.exp_tag == self.exp_tag
            and math.isclose(other.mu, self.mu)
            and math.isclose(other.sigma, self.sigma)
        )

    def __hash__(self):
        return id(self)


class continuous_params:
    def __init__(self, exp_tag, integ_type, eps, N, sigma):
        self.exp_tag = exp_tag
        self.integ_type = integ_type
        self.eps = eps
        self.N = N
        self.sigma = sigma

    def __eq__(self, other):
        return (
            other.exp_tag == self.exp_tag
            and other.integ_type == self.integ_type
            and other.eps == self.eps
            and other.N == self.N
            and math.isclose(other.sigma, self.sigma)
        )

    def __hash__(self):
        return id(self)


class single_exp_data:
    def __init__(self, data, col_names):
        self.single_data = {}
        for col in col_names:
            # print(col)
            self.single_data.update({col: data[col]})


class single_computation_discrete:
    def __init__(self, directory, exp_name, _col_names):
        self.class_data = {}
        self.col_names = _col_names
        self.exp_name = exp_name
        self.path = directory + exp_name
        self.dirs = [
            d
            for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        for d in self.dirs:
            N, lam = parse_discrete_param_string(d)
            params = discrete_params(N=N, lam=lam)
            # print(params)
            path = directory + exp_name + "/" + d + "/results.csv"
            data = genfromtxt(path, dtype=float, delimiter=",", names=True)
            self.class_data.update({params: single_exp_data(data, self.col_names)})
            # print(params.N, params.lam)

    def get_all_data(self, N_val, lam):
        input_params = discrete_params(N=N_val, lam=lam)
        key = next((k for k in self.class_data if input_params == k), None)
        return self.class_data[key]


class single_computation_continuous:
    def __init__(self, directory, exp_name, _col_names):
        self.class_data = {}
        self.col_names = _col_names
        self.exp_name = exp_name
        self.path = directory + exp_name
        self.exp_dirs = [
            d
            for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        for exp in self.exp_dirs:
            sub_exp_dirs = [
                d
                for d in os.listdir(self.path + "/" + exp)
                if os.path.isdir(os.path.join(self.path + "/" + exp, d))
            ]
            for sub_exp in sub_exp_dirs:
                # print(exp,sub_exp)
                exp_tag, integ_type, eps, N, sigma = parse_continuous_param_string(
                    exp, sub_exp
                )
                # print(exp_tag, integ_type, eps, N, sigma)
                params = continuous_params(
                    exp_tag=exp_tag, integ_type=integ_type, eps=eps, N=N, sigma=sigma
                )
                path = directory + exp_name + "/" + exp + "/" + sub_exp + "/results.csv"
                data = genfromtxt(path, dtype=float, delimiter=",", names=True)
                self.class_data.update({params: single_exp_data(data, self.col_names)})
                # print(params.N, params.lam)

    def get_all_data(self, exp_tag, integ_type, eps, N, sigma):
        input_params = continuous_params(
            exp_tag=exp_tag, integ_type=integ_type, eps=eps, N=N, sigma=sigma
        )
        key = next((k for k in self.class_data if input_params == k), None)
        return self.class_data[key]


class single_computation_continuous_est:
    def __init__(self, directory, exp_name, _col_names):
        self.class_data = {}
        self.col_names = _col_names
        self.exp_name = exp_name
        self.path = directory + exp_name
        self.exp_dirs = [
            d
            for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        for exp in self.exp_dirs:
            sub_exp_dirs = [
                d
                for d in os.listdir(self.path + "/" + exp)
                if os.path.isdir(os.path.join(self.path + "/" + exp, d))
            ]
            for sub_exp in sub_exp_dirs:
                # print(exp)
                exp_tag = exp
                mu = float(sub_exp.split("_")[0])
                sigma = float(sub_exp.split("_")[1])
                # print(exp,sub_exp)
                # print(exp, mu, sigma)

                # print(exp_tag, integ_type, eps, N, sigma)
                params = continuous_params_est(exp_tag=exp_tag, mu=mu, sigma=sigma)
                # print(params.exp_tag,params.mu,params.sigma)
                path = directory + exp_name + "/" + exp + "/" + sub_exp + "/results.csv"
                data = genfromtxt(path, dtype=float, delimiter=",", names=True)
                self.class_data.update({params: single_exp_data(data, self.col_names)})
                # print(data)
            # print(params.N, params.lam)

    def get_all_data(self, exp_tag, mu, sigma):
        input_params = continuous_params_est(exp_tag=exp_tag, mu=mu, sigma=sigma)
        key = next((k for k in self.class_data if input_params == k), None)
        return self.class_data[key]


# this refers to the spectators
sigma_dict = {
    "0": 0.25,
    "1": 0.5,
    "2": 1.0,
    "3": 2.0,
    "4": 4.0,
    # "5": 8.0,
    # "6": 16.0,
    # "7": 32.0,
    # "8": 64.0,
    # "9": 128.0,
    # "10": 256.0,
    # "11": 512.0,
    # "12": 1024.0,
    # "13": 2048.0,
}


class joint_params:
    def __init__(self, exp_tag, sigma):
        self.exp_tag = exp_tag
        self.sigma = sigma

    def __eq__(self, other):
        return other.exp_tag == self.exp_tag and math.isclose(other.sigma, self.sigma)

    def __hash__(self):
        return id(self)


class joint_computation_continuous:
    def __init__(self, directory, exp_name, sub_exp, _col_names, num_spec=None):
        self.class_data = {}
        self.col_names = _col_names
        self.exp_name = exp_name
        self.sub_exp = sub_exp
        self.num_spec = num_spec
        self.path = directory + exp_name

        if self.num_spec is None:
            self.exp_dirs = [sub_exp + "_" + sigma_ctr for sigma_ctr in sigma_dict]
        else:
            self.exp_dirs = [
                str(num_spec) + "_" + sub_exp + "_" + sigma_ctr
                for sigma_ctr in sigma_dict
            ]

        # self.exp_dirs = [
        #     d
        #     for d in os.listdir(self.path)
        #     if os.path.isdir(os.path.join(self.path, d))
        # ]
        for exp, key in zip(self.exp_dirs, sigma_dict):
            # print(exp,sub_exp)
            # exp_tag, integ_type,  N, sigma = parse_continuous_param_string(
            #     exp, sub_exp
            # )
            # print(exp_tag, integ_type, eps, N, sigma)
            path = self.path + "/" + exp + "/results.csv"
            # print(path)
            data = genfromtxt(
                path,
                dtype=float,
                delimiter=",",
                names=True,
                deletechars="""~!@#$%^&*()=+~\|]}[{';: /?>,<""",
            )
            # print(data)
            self.class_data.update(
                {sigma_dict[key]: single_exp_data(data, self.col_names)}
            )

    def get_all_data(self, sigma):
        key = next((k for k in self.class_data if sigma == k), None)
        return self.class_data[key]


participation_configs = ["111", "110", "100"]
participation_mapping = {
    "111": [r"$h(\vec{X}_{T} \mid O_1, O_2)$", r"$h(\vec{X}_{T} \mid O_1, O_2, O_3)$"],
    "110": [r"$h(\vec{X}_{T} \mid O_1, O_2)$", r"$h(\vec{X}_{T} \mid O_1, O_2, O_3')$"],
    "100": [
        r"$h(\vec{X}_{T} \mid O_1, O_2')$",
        r"$h(\vec{X}_{T} \mid O_1, O_2', O_3')$",
    ],
}


class three_exp_continuous:
    def __init__(self, directory, exp_name, sub_exp, _col_names, num_spec=None):
        self.class_data = {}
        self.col_names = _col_names
        self.exp_name = exp_name
        self.sub_exp = sub_exp
        self.num_spec = num_spec
        self.path = directory + exp_name

        self.exp_dirs = [
            str(num_spec) + "_" + sub_exp + "_" + p_conf
            for p_conf in participation_configs
        ]
        for exp, part in zip(self.exp_dirs, participation_configs):
            path = self.path + "/" + exp + "/results.csv"
            data = genfromtxt(
                path,
                dtype=float,
                delimiter=",",
                names=True,
                deletechars="""~!@#$%^&*()=+~\|]}[{';: /?>,<""",
            )
            # print(data)
            self.class_data.update({part: single_exp_data(data, self.col_names)})

    def get_all_data(self, sigma):
        key = next((k for k in self.class_data if sigma == k), None)
        return self.class_data[key]


class three_exp_continuous_brute_force:
    def __init__(self, directory, exp_name, sub_exp, _col_names, num_spec=None):
        # self.class_data = ()
        self.col_names = _col_names
        self.exp_name = exp_name
        self.sub_exp = sub_exp
        self.num_spec = num_spec
        self.path = directory + exp_name

        self.exp_dir = str(num_spec) + "_" + sub_exp
        path = self.path + "/" + self.exp_dir + "/results.csv"
        data = genfromtxt(
            path,
            dtype=float,
            delimiter=",",
            names=True,
            deletechars="""~!@#$%^&*()=+~\|]}[{';: /?>,<""",
        )
        self.class_data = data


class N_exp_brute_force:
    def __init__(self, directory, exp_name, N, num_spec=None):
        # self.class_data = ()
        # self.col_names = _col_names
        self.exp_name = exp_name
        self.num_spec = num_spec
        self.path = directory + exp_name

        self.exp_dir = str(N) + "_" + str(num_spec) + "_all"
        path = self.path + "/" + self.exp_dir + "/results.csv"
        data = genfromtxt(
            path,
            dtype=float,
            delimiter=",",
            names=True,
            deletechars="""~!@#$%^&*()=+~\|]}[{';: /?>,<""",
        )
        self.class_data = data

    # def get_all_data(self, sigma):
    #     key = next((k for k in self.class_data if sigma == k), None)
    #     return self.class_data[key]


class gamma_datatype:
    def __init__(self, directory, exp_name, _col_names):
        # self.class_data = ()
        self.col_names = _col_names
        self.exp_name = exp_name
        self.path = directory + exp_name

        path = self.path + "/" + "/results.csv"
        data = genfromtxt(
            path,
            dtype=float,
            delimiter=",",
            names=True,
            deletechars="""~!@#$%^&*()=+~\|]}[{';: /?>,<""",
        )
        self.class_data = data


class single_mixed:
    def __init__(self, directory, exp_name, _col_names):
        self.class_data = {}
        self.col_names = _col_names
        self.exp_name = exp_name
        self.path = directory + exp_name
        self.dirs = [
            d
            for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        for d in self.dirs:
            N, sigma = parse_discrete_param_string(d)
            params = self.mixed_params(N=N, sigma=sigma)
            path = directory + exp_name + "/" + d + "/results.csv"
            data = genfromtxt(path, dtype=float, delimiter=",", names=True)
            self.class_data.update({params: single_exp_data(data, self.col_names)})

    def get_all_data(self, N_val, sigma):
        input_params = self.mixed_params(N=N_val, sigma=sigma)
        key = next((k for k in self.class_data if input_params == k), None)
        return self.class_data[key]

    class mixed_params:
        def __init__(self, N, sigma):
            self.N = N
            self.sigma = sigma

        def __eq__(self, other):
            return other.N == self.N and math.isclose(other.sigma, self.sigma)

        def __hash__(self):
            return id(self)
