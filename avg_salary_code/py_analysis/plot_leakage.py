import logging
import os
import re
import subprocess
import sys
import math
import csv
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, figure, rc
from matplotlib.backends.backend_pdf import PdfPages
from natsort import natsorted
from numpy import genfromtxt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager

mpl.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "font.size": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb,sansmath}",
    }
)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)


colors = [
    "red",
    "blue",
    # "green",
    # "magenta",
    "green",
    "magenta",
    "darkorange",
    "black",
    "red",
    "blue",
    "darkgreen",
    "magenta",
    "darkorange",
]


def parse_discrete_param_string(param_string):
    param_list = param_string.split("_")
    if len(param_list) == 1:
        # uniform
        return int(param_list[0])
    else:
        raise Exception("Input string incorrect format", param_string)


class single_exp_data:
    def __init__(self, data, col_names):
        self.single_data = {}
        for col in col_names:
            self.single_data.update({col: data[col]})


class single_max:
    def __init__(self, directory, exp_name, dist):
        self.class_data = {}
        self.exp_name = exp_name
        self.dist = dist
        self.path = directory + exp_name + "/" + dist
        self.col_names = []
        self.dirs = [
            d
            for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d))
        ]
        for d in self.dirs:
            # print("d",d)
            N = parse_discrete_param_string(d)
            # params = self.max_params(N=N)
            path = self.path + "/" + d + "/results.csv"
            data = genfromtxt(
                path, dtype=None, delimiter=",", names=True, encoding=None
            )
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                for row in csv_reader:
                    self.col_names.append(row)
                    break
            print(self.col_names[0])
            # print(data)
            # print()
            # print(data['num_T'])
            # print(data['num_A'])
            # print(data['num_S'])
            # print(data['N'])
            # print(data['x_a_awae'])
            self.class_data.update({int(N): single_exp_data(data, self.col_names[0])})

    class max_params:
        def __init__(self, N):
            self.N = N

        def __eq__(self, other):
            return other.N == self.N

        def __hash__(self):
            return id(self)

    def get_all_data(self, N_val, key_str):
        # input_params = self.max_params(N=N_val)
        # key = next((k for k in self.class_data if N_val == k), None)
        data_string = self.class_data[N_val].single_data[key_str]
        # will be a string of the form x_a:awae(x_A);..
        # print(data_string)
        # split_data = data_string.split(';')
        # x_val = [], computed = []
        # for d in split_data:
        #     dd = d.split(':')
        #     x_val.append(float(dd[0]))
        #     computed.append(float(dd[1]))
        # # return x_val,computed
        return self.class_data[N_val].single_data[key_str]


def parse_data_string(data_str):
    split_data = data_str[:-1].split(";")
    x_val = []
    computed = []
    for d in split_data:
        dd = d.split(":")
        x_val.append(float(dd[0]))
        computed.append(float(dd[1]))
    return x_val, computed


def plot_main(experiment_name):
    
    dir_path = "../order_statistics/" + experiment_name+"_figs/"
    isExist = os.path.exists(dir_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")

    fig, ax1 = plt.subplots()

    output_dir = "../output/"
    uniform_max = single_max(output_dir, experiment_name, "uniform")
    N = 8
    # print(uniform_max.class_data[4])
    # print(uniform_max.get_all_data(N, "x_a_awae"))

    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"Input $x_T$ or $x_A$")
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "x", "s"])
    lsty = itertools.cycle(["-", "--", "-."])
    spec = uniform_max.get_all_data(N, "num_S")
    raw_awae = uniform_max.get_all_data(N, "x_a_awae")
    raw_twae = uniform_max.get_all_data(N, "x_T_twae")
    plot_lines = []
    spec_legends = []

    for s, r_awae, r_twae in zip(spec, raw_awae, raw_twae):
        c = next(cc)
        alph = next(alphas)

        parsed_x_a, parsed_awae = parse_data_string(r_awae)
        (l1,) = ax1.plot(
            parsed_x_a,
            parsed_awae,
            marker="+",
            color=c,
            # markersize=11,
            alpha=alph,
            linestyle="--",
        )

        parsed_x_t, parsed_twae = parse_data_string(r_twae)
        (l1,) = ax1.plot(
            parsed_x_t,
            parsed_twae,
            marker="o",
            color=c,
            # markersize=11,
            alpha=alph,
            linestyle="-",
        )
        spec_legends.append(
            Line2D(
                [0],
                [0],
                color=c,
                marker="",
                alpha=alph,
                linestyle="-",
                label=r"$\lvert S\rvert\ = %s$" % s,
            ),
        )
        plot_lines.append([l1])

    val_legend = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            alpha=0.5,
            # markersize=11,
            linestyle="-",
            label="twae$(x_T)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="+",
            alpha=0.5,
            # markersize=11,
            linestyle="--",
            label="awae$(x_A)$",
        ),
    ]

    legend1 = plt.legend(handles=spec_legends, loc="lower right", fontsize=14)
    legend2 = plt.legend(handles=val_legend, loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=14, ncols=2)
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    path = dir_path + experiment_name+ "_uniform_" + str(N) + "_twae_awae"
    print(path)
    plt.savefig(
        path+".pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        path+".png",
        bbox_inches="tight",
        transparent=True,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # plot_main("median")
        # # plot_main("max")
        # plot_main("median_round")
        plot_main("median_min")
        # plot_main("max_m_1")
