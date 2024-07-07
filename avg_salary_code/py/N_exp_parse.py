import logging
from math import dist
import os
import re
import subprocess
import sys
import itertools
from scipy.interpolate import make_interp_spline

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, figure, rc, pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from natsort import natsorted
from numpy import genfromtxt

from plot_classes import *

# from plot_3xp_funcs import *

logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager


mpl.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb}",
    }
)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)


output_dir = "../output/"
# output_dir = "/Users/alessandrobaccarini/output/"
dir_path = "../../proof/figs/"
colors = [
    "red",
    "blue",
    "limegreen",
    "teal",
    "magenta",
    "saddlebrown",
    "mediumslateblue",
    "black",
    "red",
    "blue",
    "darkgreen",
    "magenta",
    "darkorange",
]


def h_X_T(sigma):
    return np.log(np.sqrt(sigma * 2.0 * np.pi * np.e)) / np.log(2.0)


def GET_BIT(X, N):
    return ((X) >> (N)) & int(1)


def gen_T_configs(N):
    result = []
    for i in range(1, pow(2, N)):
        result.append("")
        for j in range(0, N):
            result[i - 1] = result[i - 1] + str(GET_BIT(i, j))
    return result


def plot_all():
    overlap_percentage = []
    ctr = 0
    for N in range(2, 5):
        overlap_percentage.append([])
        for Delta in range(2, 25):
            # N = 3
            target_configs = gen_T_configs(N)
            num_entropies = len(target_configs)
            num_spec_configs = pow(2, N) - 1
            main_exp = "N_experiments"
            # Delta = 10

            brute_force_all = N_exp_brute_force(output_dir, main_exp, N, num_spec=Delta)
            # print(brute_force_all.class_data.dtype.names)
            spectator_vars = list(brute_force_all.class_data.dtype.names)[
                1:-2
            ]  # ignoring sigma and the h_X_T, min_cond_ent
            # print(spectator_vars)
            sizes = []
            for i in range(1, N + 1):
                for j in range(i, N + 1):
                    if i != j:
                        # print(i,j)
                        size = 0
                        for spec_str in spectator_vars:
                            if (str(i) in spec_str) and (str(j) in spec_str):
                                size += brute_force_all.class_data[spec_str]
                        sizes.append(size)
                # print(sizes, brute_force_all.class_data)
            int_sizes = [int(i) for i in sizes]
            res = all(ele == int_sizes[0] for ele in int_sizes)
            if res:
                print(
                    N,
                    Delta,
                    " optimal overlap : %s percent" % ((sizes[0] / Delta) * 100),
                    sizes,
                )
                # overlap_percentage[ctr].append( 1/ (Delta/ sizes[0]) * 100)
                overlap_percentage[ctr].append(((sizes[0] / Delta) * 100))
            else:
                print("!!!!! we have unequal elements - ", sizes)
        ctr += 1
    print(overlap_percentage)
    fig, ax1 = plt.subplots()
    deltas = list(range(2, 25))
    print(deltas)
    mm = itertools.cycle(["x", "o", "v"])
    ll = itertools.cycle(["-", "--", "-."])
    cc = itertools.cycle(colors)
    for N in range(2, 5):
        c = next(cc)
        m = next(mm)
        l = next(ll)
        ax1.plot(
            deltas,
            overlap_percentage[N - 2],
            marker=m,
            color=c,
            alpha=0.8,
            linestyle=l,
            label=r"$M = %s$" % (N),
        )
    ax1.set_ylim([15, 52])

    plt.legend(loc="lower right")
    # plt.show()
    plt.ylabel(r"Spectator overlap (\%)")
    plt.xlabel(r"$n$")

    plt.savefig(dir_path + "overlap_comparison.pdf", bbox_inches="tight")

    return 0


def plot_2_3():
    overlap_percentage = []
    ctr = 0
    for N in range(2, 4):
        overlap_percentage.append([])
        for Delta in range(2, 51):
            # N = 3
            target_configs = gen_T_configs(N)
            num_entropies = len(target_configs)
            num_spec_configs = pow(2, N) - 1
            main_exp = "N_experiments"
            # Delta = 10

            brute_force_all = N_exp_brute_force(output_dir, main_exp, N, num_spec=Delta)
            # print(brute_force_all.class_data.dtype.names)
            spectator_vars = list(brute_force_all.class_data.dtype.names)[
                1:-2
            ]  # ignoring sigma and the h_X_T, min_cond_ent
            # print(spectator_vars)
            sizes = []
            for i in range(1, N + 1):
                for j in range(i, N + 1):
                    if i != j:
                        # print(i,j)
                        size = 0
                        for spec_str in spectator_vars:
                            if (str(i) in spec_str) and (str(j) in spec_str):
                                size += brute_force_all.class_data[spec_str]
                        sizes.append(size)
                # print(sizes, brute_force_all.class_data)
            int_sizes = [int(i) for i in sizes]
            res = all(ele == int_sizes[0] for ele in int_sizes)
            if res:
                print(
                    N,
                    Delta,
                    " optimal overlap : %s percent" % ((sizes[0] / Delta) * 100),
                    sizes,
                )
                # overlap_percentage[ctr].append( 1/ (Delta/ sizes[0]) * 100)
                overlap_percentage[ctr].append(((sizes[0] / Delta) * 100))
            else:
                print("!!!!! we have unequal elements - ", sizes)
        ctr += 1
    print(overlap_percentage)
    fig, ax1 = plt.subplots()
    deltas = list(range(2, 51))
    print(deltas)
    mm = itertools.cycle(["x", "o", "v"])
    ll = itertools.cycle(["-", "--", "-."])
    cc = itertools.cycle(colors)
    for N in range(2, 4):
        c = next(cc)
        m = next(mm)
        l = next(ll)
        ax1.plot(
            deltas,
            overlap_percentage[N - 2],
            marker=m,
            color=c,
            alpha=0.8,
            linestyle=l,
            label=r"$M = %s$" % (N),
        )
    plt.legend(loc="lower right")
    # plt.show()
    plt.ylabel(r"Spectator overlap (\%)")
    plt.xlabel(r"$n$")
    ax1.set_ylim([15, 52])

    plt.savefig(dir_path + "overlap_comparison_2_3.pdf", bbox_inches="tight")

    return 0


def plot_percentages_and_abs():
    ctr = 0
    all_data = []
    H_X_T = []
    Deltas = []
    min_cond_entropy = []
    for N in range(1, 5):
        all_data.append([])
        H_X_T.append([])
        Deltas.append([])
        min_cond_entropy.append([])
        for Delta in range(2, 26):
            target_configs = gen_T_configs(N)
            main_exp = "N_experiments"
            brute_force_all = N_exp_brute_force(output_dir, main_exp, N, num_spec=Delta)
            # print(brute_force_all.class_data["min_cond_entropy"])
            # all_data[N-1].append([N, Delta, float(brute_force_all.class_data["H_X_T"]),float(brute_force_all.class_data["min_cond_entropy"])])
            H_X_T[N - 1].append(float(brute_force_all.class_data["H_X_T"]))
            Deltas[N - 1].append(Delta)
            min_cond_entropy[N - 1].append(
                float(brute_force_all.class_data["min_cond_entropy"])
            )

    # print(H_X_T)
    # print(Deltas)
    # print(min_cond_entropy)

    fig, ax1 = plt.subplots()
    deltas = list(range(2, 51))
    # print(deltas)
    mm = itertools.cycle(["x", "o", "v"])
    ll = itertools.cycle(["-", "--", "-."])
    cc = itertools.cycle(colors)
    for N in range(1, 5):
        c = next(cc)
        m = next(mm)
        l = next(ll)
        ax1.plot(
            Deltas[0],
            [a - b for a, b in zip(H_X_T[0], min_cond_entropy[N - 1])],
            marker=m,
            color=c,
            alpha=0.7,
            linestyle=l,
            label=r"$M = %s$" % (N),
        )
    plt.title(r"$h(\vec{X}_T) - h(\vec{X}_T\mid O_{1,\dots i}^{(\tau_{1,\dots,    i})})$",
              y = 1.05,fontsize=14)

    plt.legend(loc="upper right")
    plt.ylabel(r"Absolute entropy loss (bits)")
    plt.xlabel(r"$n$")
    plt.savefig(dir_path + "abs_loss_N_exp.pdf", bbox_inches="tight")

    fig, ax1 = plt.subplots()
    deltas = list(range(2, 51))
    mm = itertools.cycle(["x", "o", "v"])
    ll = itertools.cycle(["-", "--", "-."])
    cc = itertools.cycle(colors)
    abs_0 = [a - b for a, b in zip(H_X_T[0], min_cond_entropy[0])]
    abs_1 = [a - b for a, b in zip(H_X_T[0], min_cond_entropy[1])]
    abs_2 = [a - b for a, b in zip(H_X_T[0], min_cond_entropy[2])]
    abs_3 = [a - b for a, b in zip(H_X_T[0], min_cond_entropy[3])]
    
    
    # for N in range(2, 5):
    c = next(cc)
    m = next(mm)
    l = next(ll)
    # if N == 2:
    #     data =  
    #     # print(data)
    #     ax1.plot(
    #         Deltas[0],
    #        data,
    #         marker=m,
    #         color=c,
    #         alpha=0.7,
    #         linestyle=l,
    #         label=r"$M = %s \rightarrow %s $" % (N-1, N),
    #     )
    # else:
    ax1.plot(
        Deltas[0],
        [a/b*100 -100 for a,b in zip(abs_1, abs_0)],
        # [
        #     100 -((a - b) - (b-c)) / (a-b) * 100
        #     for a, b, c  in zip( min_cond_entropy[N - 3],min_cond_entropy[N - 2], min_cond_entropy[N - 1])
        # ],
        marker=m,
        color=c,
        alpha=0.7,
        linestyle=l,
        label=r"$M = %s \rightarrow %s $" % (1, 1),
    )
    c = next(cc)
    m = next(mm)
    l = next(ll)
    ax1.plot(
        Deltas[0],
        [a/b*100 -100 for a,b in zip(abs_2, abs_0)],
        # [
        #     100 -((a - b) - (b-c)) / (a-b) * 100
        #     for a, b, c  in zip( min_cond_entropy[N - 3],min_cond_entropy[N - 2], min_cond_entropy[N - 1])
        # ],
        marker=m,
        color=c,
        alpha=0.7,
        linestyle=l,
        label=r"$M = %s \rightarrow %s $" % (1, 2),
    )
    c = next(cc)
    m = next(mm)
    l = next(ll)
    ax1.plot(
        Deltas[0],
        [a/b*100 -100 for a,b in zip(abs_3, abs_0)],
        # [
        #     100 -((a - b) - (b-c)) / (a-b) * 100
        #     for a, b, c  in zip( min_cond_entropy[N - 3],min_cond_entropy[N - 2], min_cond_entropy[N - 1])
        # ],
        marker=m,
        color=c,
        alpha=0.7,
        linestyle=l,
        label=r"$M = %s \rightarrow %s $" % (1, 3),
    )        
    
        
    plt.title(
        # r"$   \frac{h(\vec{X}_T\mid O_{1,\dots M-2}^{(\tau_{1,\dots,M-2})}) - 2 h(\vec{X}_T\mid O_{1,\dots M-1}^{(\tau_{1,\dots,M-1})}) + h(\vec{X}_T\mid O_{1,\dots M}^{(\tau_{1,\dots,M})})}{h(\vec{X}_T\mid O_{1,\dots M-2}^{(\tau_{1,\dots,M-2})}) -  h(\vec{X}_T\mid O_{1,\dots M-1}^{(\tau_{1,\dots,M-1})})}  $"
         r"$\frac{\Delta_{(M-2) \rightarrow (M-1)} - \Delta_{(M-1) \rightarrow (M)}}{\Delta_{(M-2) \rightarrow (M-1)}},$" + "\n" +r" $\Delta_{(i-1) \rightarrow (i)} =  h(\vec{X}_T\mid O_{1,\dots i-1}^{(\tau_{1,\dots,i-1})}) - h(\vec{X}_T\mid O_{1,\dots i}^{(\tau_{1,\dots,    i})})$",
              y = 1.05,fontsize=14)
    
    ax1.set_ylim([0, 130])
    plt.legend(loc="upper right")
    plt.ylabel(r"Relative entropy loss (\%)")
    plt.xlabel(r"$n$")
    plt.savefig(dir_path + "rel_loss_N_exp.pdf", bbox_inches="tight")

    plt.close("all")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_percentages_and_abs()
        # plot_all()
        # plot_2_3()
