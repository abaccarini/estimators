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

logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager

mpl.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "font.size": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb}",
    }
)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)


output_dir = "../output/"

colors = [
    "red",
    "blue",
    "teal",
    "magenta",
    "green",
    "saddlebrown",
    "mediumslateblue",
    "black",
    "red",
    "blue",
    "darkgreen",
    "magenta",
    "darkorange",
]


def get_dist_name(exp_name):
    if exp_name == "sum_poisson":
        return r"Pois$(\lambda = N/2)$"
    elif exp_name == "sum_uniform":
        return r"$\mathcal{U}(0,N-1)$"
    elif exp_name == "lognormal_v2" or exp_name == "lognormal_v2_est":
        return r"$\log\mathcal{N}(\mu,\sigma^2)$"
    elif exp_name == "normal_v2":
        return r"$\mathcal{N}(\mu,\sigma^2)$"
    if exp_name == "gamma":
        return r"Gamma$(k, \theta)$"
    else:
        raise Exception("Invalid distribution name: ", exp_name)


def set_params_all(dist_type=None):
    # if dist_type is None:
    #     plt.title(r"$O = \vec{X}_T + X_S$")
    # else:
    # plt.title(
    #     r"$O = \vec{X}_T + X_S$, " + get_dist_name(dist_type) + "\nVarious Parameters"
    # )
    # plt.title(r"$O = \vec{X}_T + X_S$, " +get_dist_name(dist_type))
    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"No. spectators")


def set_params_abs(dist_type=None):
    # if dist_type is None:
    #     plt.title(r"$O = \vec{X}_T + X_S$, Absolute Entropy Loss")
    # else:
    #     plt.title(
    #         r"$O = \vec{X}_T + X_S$, " + get_dist_name(dist_type) + "\nAbsolute Entropy Loss"
    #     )
    # plt.title(r"$O = \vec{X}_T + X_S$, Absolute Entropy Loss" +"\n"+get_dist_name(dist_type))
    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"No. spectators")


def set_params_rel(dist_type=None):
    # if dist_type is None:
    #     plt.title(r"$O = \vec{X}_T + X_S$, Relative Entropy Loss")
    # else:
    #     plt.title(
    #         r"$O = \vec{X}_T + X_S$, " + get_dist_name(dist_type) + "\nRelative Entropy Loss"
    #     )
    plt.ylabel(r"Percent change (\%)")
    plt.xlabel(r"No. spectators")


def plot_all_discrete(discrete_data, N_vals, upper_bound, output_path):
    fig, ax1 = plt.subplots()
    set_params_abs(discrete_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []
    dists = []
    if (
        discrete_data.exp_name == "sum_poisson"
        or discrete_data.exp_name == "sum_poisson_min"
    ):
        dists = [r"$\lambda = %s$" % str(int(n / 2)) for n in N_vals]
    elif discrete_data.exp_name == "sum_uniform":
        dists = [r"$N = %s$" % n for n in N_vals]
    else:
        print("ERROR\n")
        return -1

    for N in N_vals:
        if (
            discrete_data.exp_name == "sum_poisson"
            or discrete_data.exp_name == "sum_poisson_min"
        ):
            lam = N / 2
        else:
            lam = None
        # print(lam)
        c = next(cc)
        alph = next(alphas)
        spec = discrete_data.get_all_data(N, lam).single_data["spectators"]
        abs_loss = [
            (a - b)
            for a, b in zip(
                discrete_data.get_all_data(N, lam).single_data["target_init_entropy"],
                discrete_data.get_all_data(N, lam).single_data["awae_results"],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])
        if N == N_vals[0]:

            plt.xlim([0, max(spec[0:upper_bound]) + 0.05])
            plt.ylim([0, max(abs_loss[0:upper_bound]) + 0.05])

    legend1 = plt.legend(
        plot_lines[0],
        [r"$H(\vec{X}_T) - H(\vec{X}_T \mid X_T + X_S)$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 2.0))

    # plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + discrete_data.exp_name + "_absolute_loss.pdf", bbox_inches="tight"
    )

    plt.savefig(
        output_path + discrete_data.exp_name + "_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_all(discrete_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []

    for N in N_vals:
        if (
            discrete_data.exp_name == "sum_poisson"
            or discrete_data.exp_name == "sum_poisson_min"
        ):
            lam = N / 2
        else:
            lam = None
        c = next(cc)
        alph = next(alphas)
        spec = discrete_data.get_all_data(N, lam).single_data["spectators"]
        awae = discrete_data.get_all_data(N, lam).single_data["awae_results"]
        H_T = discrete_data.get_all_data(N, lam).single_data["target_init_entropy"]

        (l1,) = ax1.plot(
            spec,
            awae,
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T\mid X_T + X_S)}$",
        )
        (l2,) = ax1.plot(
            spec,
            H_T,
            marker="",
            color=c,
            alpha=alph,
            linestyle="--",
            label=r"${H(\vec{X}_T)}$",
        )
        plot_lines.append([l1, l2])
        # if N == N_vals[0]:

        #     plt.xlim([0, max(spec)+0.25])
        #     plt.ylim([0, max(awae)+0.25])

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$H(\vec{X}_T\mid X_T + X_S)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle="--",
            label=r"$H(\vec{X}_T)$",
        ),
    ]
    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper right",
    )
    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"$H(\vec{X}_T \mid X_T + X_S)$", r"$H(\vec{X}_T)$"],
    #     # bbox_to_anchor=(1.04, 1),
    #     loc="upper right",
    # )
    plt.legend(
        [l[0] for l in plot_lines],
        dists,
        #    bbo\vec{x}_to_anchor=(1.04, 0),
        loc="lower right",
    )
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + discrete_data.exp_name + "_multiple_exps.pdf", bbox_inches="tight"
    )

    plt.savefig(
        output_path + discrete_data.exp_name + "_multiple_exps.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    # ax1.tick_params(length=0)

    set_params_rel(discrete_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    # print(discrete_data, N_vals,  upper_bound, output_path, lam)
    plot_lines = []

    for N in N_vals:
        if (
            discrete_data.exp_name == "sum_poisson"
            or discrete_data.exp_name == "sum_poisson_min"
        ):
            lam = N / 2
        else:
            lam = None
        c = next(cc)
        alph = next(alphas)
        spec = discrete_data.get_all_data(N, lam).single_data["spectators"]

        abs_loss = [
            ((a - b) / a) * 100.0
            for a, b in zip(
                discrete_data.get_all_data(N, lam).single_data["target_init_entropy"],
                discrete_data.get_all_data(N, lam).single_data["awae_results"],
            )
        ]
        values = {1.0: "x", 5.0: "s"}
        # print(abs_loss)
        for value in values:
            for i, a in enumerate(abs_loss):
                if a < value:
                    res = (i, a)
                    break

            # res = min(enumerate(abs_loss), key=lambda x: abs(value - x[1]))

            # print(res)
            if N == N_vals[0] or N == N_vals[-1]:
                plt.vlines(
                    res[0] + 1, 0, res[1], alpha=0.5, color="black", linestyle="--"
                )
                plt.plot(
                    res[0] + 1, res[1], alpha=0.5, marker=values[value], color="black"
                )
            if N == N_vals[0]:
                plt.hlines(
                    res[1],
                    xmin=0,
                    xmax=res[0] + 1,
                    alpha=0.5,
                    color="black",
                    linestyle="--",
                )
        # print()
        # ax.hlines(y=0.2, xmin=4, xmax=20, linewidth=2, color='r')

        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$",
        )
        plot_lines.append([l1])
        if N == N_vals[0]:

            plt.xlim([0, max(spec[0:upper_bound]) + 0.25])
            plt.ylim([0, max(abs_loss[0:upper_bound]) + 0.25])

    legend1 = plt.legend(
        plot_lines[0],
        [r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 2.0))

    # plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + discrete_data.exp_name + "_relative_loss.pdf", bbox_inches="tight"
    )

    plt.savefig(
        output_path + discrete_data.exp_name + "_relative_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.close("all")


def plot_all_discrete_min(discrete_data, N_vals, upper_bound, output_path):
    fig, ax1 = plt.subplots()
    set_params_abs(discrete_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []
    dists = []
    if discrete_data.exp_name == "sum_poisson_min":
        dists = [r"$\lambda = %s$" % str(int(n / 2)) for n in N_vals]
    elif discrete_data.exp_name == "sum_uniform":
        dists = [r"$N = %s$" % n for n in N_vals]
    else:
        print("ERROR\n")
        return -1

    for N in N_vals:
        if discrete_data.exp_name == "sum_poisson_min":
            lam = N / 2
        else:
            lam = None
        # print(lam)
        c = next(cc)
        alph = next(alphas)
        spec = discrete_data.get_all_data(N, lam).single_data["spectators"]
        abs_loss = [
            (a - b)
            for a, b in zip(
                discrete_data.get_all_data(N, lam).single_data["target_init_entropy"],
                discrete_data.get_all_data(N, lam).single_data["awae_results"],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H_{\infty}(\vec{X}_T) -  H_{\infty}(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])
        if N == N_vals[0]:

            plt.xlim([0, max(spec[0:upper_bound]) + 0.05])
            plt.ylim([0, max(abs_loss[0:upper_bound]) + 0.05])

    legend1 = plt.legend(
        plot_lines[0],
        [r"$H_{\infty}(\vec{X}_T) - H_{\infty}(\vec{X}_T \mid X_T + X_S)$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 2.0))

    # plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + discrete_data.exp_name + "_absolute_loss.pdf", bbox_inches="tight"
    )

    plt.savefig(
        output_path + discrete_data.exp_name + "_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_all(discrete_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []

    for N in N_vals:
        if discrete_data.exp_name == "sum_poisson_min":
            lam = N / 2
        else:
            lam = None
        c = next(cc)
        alph = next(alphas)
        spec = discrete_data.get_all_data(N, lam).single_data["spectators"]
        awae = discrete_data.get_all_data(N, lam).single_data["awae_results"]
        H_T = discrete_data.get_all_data(N, lam).single_data["target_init_entropy"]

        (l1,) = ax1.plot(
            spec,
            awae,
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T\mid X_T + X_S)}$",
        )
        (l2,) = ax1.plot(
            spec,
            H_T,
            marker="",
            color=c,
            alpha=alph,
            linestyle="--",
            label=r"${H(\vec{X}_T)}$",
        )
        plot_lines.append([l1, l2])
        # if N == N_vals[0]:

        #     plt.xlim([0, max(spec)+0.25])
        #     plt.ylim([0, max(awae)+0.25])

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$H_{\infty}(\vec{X}_T\mid X_T + X_S)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle="--",
            label=r"$H_{\infty}(\vec{X}_T)$",
        ),
    ]
    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper right",
    )
    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"$H(\vec{X}_T \mid X_T + X_S)$", r"$H(\vec{X}_T)$"],
    #     # bbox_to_anchor=(1.04, 1),
    #     loc="upper right",
    # )
    plt.legend(
        [l[0] for l in plot_lines],
        dists,
        #    bbo\vec{x}_to_anchor=(1.04, 0),
        loc="lower right",
    )
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + discrete_data.exp_name + "_multiple_exps.pdf", bbox_inches="tight"
    )

    plt.savefig(
        output_path + discrete_data.exp_name + "_multiple_exps.png",
        bbox_inches="tight",
        transparent=True,
    )

    # fig, ax1 = plt.subplots()
    # # ax1.tick_params(length=0)

    # set_params_rel(discrete_data.exp_name)
    # cc = itertools.cycle(colors)
    # alphas = itertools.cycle([0.5])
    # # print(discrete_data, N_vals,  upper_bound, output_path, lam)
    # plot_lines = []

    # for N in N_vals:
    #     if discrete_data.exp_name == "sum_poisson_min":
    #         lam = N / 2
    #     else:
    #         lam = None
    #     c = next(cc)
    #     alph = next(alphas)
    #     spec = discrete_data.get_all_data(N, lam).single_data["spectators"]

    #     abs_loss = [
    #         ((a - b) / a) * 100.0
    #         for a, b in zip(
    #             discrete_data.get_all_data(N, lam).single_data["target_init_entropy"],
    #             discrete_data.get_all_data(N, lam).single_data["awae_results"],
    #         )
    #     ]
    #     values = {1.0: 'x', 5.0: 's'}
    #     # print(abs_loss)
    #     for value in values:
    #         for i,a in enumerate(abs_loss):
    #            if a < value:
    #                res = (i, a)
    #                break

    #         # res = min(enumerate(abs_loss), key=lambda x: abs(value - x[1]))

    #         # print(res)
    #         if N == N_vals[0] or N == N_vals[-1]:
    #             plt.vlines(res[0] + 1, 0, res[1], alpha=0.5, color="black", linestyle="--")
    #             plt.plot(res[0] + 1, res[1], alpha=0.5, marker=values[value], color="black")
    #         if N == N_vals[0]:
    #             plt.hlines(
    #                 res[1],
    #                 xmin=0,
    #                 xmax=res[0] + 1,
    #                 alpha=0.5,
    #                 color="black",
    #                 linestyle="--",
    #             )
    #     # print()
    #     # ax.hlines(y=0.2, xmin=4, xmax=20, linewidth=2, color='r')

    #     (l1,) = ax1.plot(
    #         spec[0:upper_bound],
    #         abs_loss[0:upper_bound],
    #         marker="",
    #         color=c,
    #         alpha=alph,
    #         linestyle="-",
    #         label=r"$\frac{H_{\infty}(\vec{X}_T) -  H_{\infty}(\vec{X}_T\mid X_T + X_S)}{H_{\infty}(\vec{X}_T)}$",
    #     )
    #     plot_lines.append([l1])
    #     if N == N_vals[0]:

    #         plt.xlim([0, max(spec[0:upper_bound]) + 0.25])
    #         plt.ylim([0, max(abs_loss[0:upper_bound]) + 0.25])

    # legend1 = plt.legend(
    #     plot_lines[0], [r"$\frac{H_{\infty}(\vec{X}_T) -  H_{\infty}(\vec{X}_T\mid X_T + X_S)}{H_{\infty}(\vec{X}_T)}$"], loc="upper right"
    # )
    # plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    # plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound])+1, 2.0))

    # # plt.gca().add_artist(legend1)
    # plt.savefig(
    #     output_path + discrete_data.exp_name + "_relative_loss.pdf", bbox_inches="tight"
    # )

    # plt.savefig(
    #     output_path + discrete_data.exp_name + "_relative_loss.png", bbox_inches="tight" , transparent=True
    # )

    plt.close("all")


def plot_all_discrete_min_v_shannon(
    discrete_data, discrete_data_min, N_vals, upper_bound, output_path
):
    fig, ax1 = plt.subplots()
    set_params_abs(discrete_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []
    dists = []
    if discrete_data_min.exp_name == "sum_poisson_min":
        dists = [r"$\lambda = %s$" % str(int(n / 2)) for n in N_vals]
    elif discrete_data_min.exp_name == "sum_uniform":
        dists = [r"$N = %s$" % n for n in N_vals]
    else:
        print("ERROR\n")
        return -1
    legend_elements =[]
    for N in N_vals:
        if discrete_data.exp_name == "sum_poisson_min":
            lam = N / 2
        else:
            lam = None
        # print(lam)
        c = next(cc)
        alph = next(alphas)
        spec = discrete_data.get_all_data(N, lam).single_data["spectators"]
        abs_loss = [
            (a - b)
            for a, b in zip(
                discrete_data.get_all_data(N, lam).single_data["target_init_entropy"],
                discrete_data.get_all_data(N, lam).single_data["awae_results"],
            )
        ]
        spec_min = discrete_data_min.get_all_data(N, lam).single_data["spectators"]
        abs_loss_min = [
            (a - b)
            for a, b in zip(
                discrete_data_min.get_all_data(N, lam).single_data[
                    "target_init_entropy"
                ],
                discrete_data_min.get_all_data(N, lam).single_data["awae_results"],
            )
        ]

        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="x",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
        )
        (l2,) = ax1.plot(
            spec_min[0:upper_bound],
            abs_loss_min[0:upper_bound],
            marker="o",
            color=c,
            alpha=alph,
            linestyle="-.",
            label=r"${H_{\infty}(\vec{X}_T) -  H_{\infty}(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1, l2])
        if N == N_vals[0]:

            plt.xlim([0, max(spec[0:upper_bound]) + 0.05])
            plt.ylim([0, max(abs_loss[0:upper_bound]) + 0.05])

        legend_elements.append(
        Line2D(
            [0],
            [0],
            color=c,
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\lambda = %s$" % str(int(N / 2))
        ))
    legend2 = plt.legend(
        handles=legend_elements,
        loc="center right",
        # bbox_to_anchor=(1.04, 1),
    )   
    
    legend1 = plt.legend(
        plot_lines[0],
        [
            r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
            r"$H_{\infty}(\vec{X}_T) - H_{\infty}(\vec{X}_T \mid X_T + X_S)$",
        ],
        loc="upper right",
        # bbox_to_anchor=(0.04, 1.3),
    )
    # plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 2.0))

    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.savefig(output_path + "min_v_shannon_absolute_loss.pdf", bbox_inches="tight")

    plt.savefig(
        output_path + "min_v_shannon_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_all(discrete_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []

    for N in N_vals:
        if discrete_data.exp_name == "sum_poisson_min":
            lam = N / 2
        else:
            lam = None
        c = next(cc)
        alph = next(alphas)
        spec = discrete_data.get_all_data(N, lam).single_data["spectators"]
        awae = discrete_data.get_all_data(N, lam).single_data["awae_results"]
        H_T = discrete_data.get_all_data(N, lam).single_data["target_init_entropy"]

        spec_min = discrete_data_min.get_all_data(N, lam).single_data["spectators"]
        awae_min = discrete_data_min.get_all_data(N, lam).single_data["awae_results"]
        H_T_min = discrete_data_min.get_all_data(N, lam).single_data[
            "target_init_entropy"
        ]

        (l1,) = ax1.plot(
            spec[0:upper_bound],
            awae[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T\mid X_T + X_S)}$",
        )
        (l2,) = ax1.plot(
            spec[0:upper_bound],
            H_T[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="--",
            label=r"${H(\vec{X}_T)}$",
        )

        (l3,) = ax1.plot(
            spec_min[0:upper_bound],
            awae_min[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-.",
            label=r"${H_{\infty}(\vec{X}_T\mid X_T + X_S)}$",
        )
        (l4,) = ax1.plot(
            spec_min[0:upper_bound],
            H_T_min[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle=":",
            label=r"${H_{\infty}(\vec{X}_T)}$",
        )

        plot_lines.append([l1, l2, l3, l4])
        # if N == N_vals[0]:

        #     plt.xlim([0, max(spec)+0.25])
        #     plt.ylim([0, max(awae)+0.25])

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$H(\vec{X}_T\mid X_T + X_S)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle="--",
            label=r"$H(\vec{X}_T)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle="-.",
            label=r"$H_{\infty}(\vec{X}_T\mid X_T + X_S)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=alph,
            linestyle=":",
            label=r"$H_{\infty}(\vec{X}_T)$",
        ),
    ]
    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.04, 1),
    )
    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"$H(\vec{X}_T \mid X_T + X_S)$", r"$H(\vec{X}_T)$"],
    #     # bbox_to_anchor=(1.04, 1),
    #     loc="upper right",
    # )
    plt.legend(
        [l[0] for l in plot_lines],
        dists,
        #    bbo\vec{x}_to_anchor=(1.04, 0),
        loc="lower right",
    )
    plt.gca().add_artist(legend1)
    plt.savefig(output_path + "min_v_shannon.pdf", bbox_inches="tight")

    plt.savefig(
        output_path + "min_v_shannon.png", bbox_inches="tight", transparent=True
    )

    plt.close("all")


def plot_all_continuous(
    continuous_data, mu, sigma_vals, exp_name, integ_type, eps, upper_bound, output_path
):
    fig, ax1 = plt.subplots()
    set_params_abs(continuous_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    # print(sigma_vals)
    plot_lines = []
    N = 8  # doesnt matter for continuous
    for sigma in sigma_vals:

        c = next(cc)
        alph = next(alphas)
        print(exp_name, integ_type, eps, N, sigma)
        spec = continuous_data.get_all_data(
            exp_name, integ_type, eps, N, sigma
        ).single_data["num_spec"]
        abs_loss = [
            (a - b)
            for a, b in zip(
                continuous_data.get_all_data(
                    exp_name, integ_type, eps, N, sigma
                ).single_data["h_T"],
                continuous_data.get_all_data(
                    exp_name, integ_type, eps, N, sigma
                ).single_data["awae_differential"],
            )
        ]
        for i in range(len(abs_loss)):
            print(abs_loss[i])
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])

    dists = []
    if continuous_data.exp_name == "normal_v2":
        col = "black"

        dists = [r"$(%s, %s)$" % (int(mu), int(n)) for n in sigma_vals]
    elif continuous_data.exp_name == "lognormal_v2":
        col = colors[0]
        dists = [r"$(%.2f, %.2f)$" % (mu, n) for n in sigma_vals]
    else:
        print("ERROR\n")
        return -1

    legend1 = plt.legend(
        plot_lines[0],
        [r"$h(\vec{X}_T) - h(\vec{X}_T \mid X_T + X_S)$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 2.0))

    # plt.gca().add_artist(legend1)

    print(output_path + continuous_data.exp_name + "_absolute_loss.pdf")
    plt.savefig(
        output_path + continuous_data.exp_name + "_absolute_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + continuous_data.exp_name + "_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_all(continuous_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []

    for sigma in sigma_vals:

        c = next(cc)
        alph = next(alphas)
        spec = continuous_data.get_all_data(
            exp_name, integ_type, eps, N, sigma
        ).single_data["num_spec"]
        awae = continuous_data.get_all_data(
            exp_name, integ_type, eps, N, sigma
        ).single_data["awae_differential"]
        H_T = continuous_data.get_all_data(
            exp_name, integ_type, eps, N, sigma
        ).single_data["h_T"]

        (l1,) = ax1.plot(
            spec,
            awae,
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T\mid X_T + X_S)}$",
        )
        (l2,) = ax1.plot(
            spec,
            H_T,
            marker="",
            color=c,
            alpha=alph,
            linestyle="--",
            label=r"${h(\vec{X}_T)}$",
        )
        plot_lines.append([l1, l2])

    # dists = [r"$(%s, %s)$" % (mu, n) for n in sigma_vals]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=col,
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$h(\vec{X}_T\mid X_T + X_S)$",
        ),
        Line2D(
            [0],
            [0],
            color=col,
            marker="",
            alpha=alph,
            linestyle="--",
            label=r"$h(\vec{X}_T)$",
        ),
    ]
    if exp_name == "sh-v-diff-real":
        new_loc = "center right"
    else:
        new_loc = "upper right"
    legend1 = plt.legend(handles=legend_elements, loc=new_loc)

    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"$h(\vec{X}_T \mid X_T + X_S)$", r"$h(\vec{X}_T)$"],
    #     # bbo\vec{x}_to_anchor=(1.04, 1),
    #     loc="upper right",
    # )
    plt.legend(
        [l[0] for l in plot_lines],
        dists,
        #    bbo\vec{x}_to_anchor=(1.04, 0),
        loc="lower right",
    )
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + continuous_data.exp_name + "_multiple_exps.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + continuous_data.exp_name + "_multiple_exps.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_rel(continuous_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    # print(continuous_data, N_vals,  upper_bound, output_path, lam)
    plot_lines = []

    for sigma in sigma_vals:

        c = next(cc)
        alph = next(alphas)
        spec = continuous_data.get_all_data(
            exp_name, integ_type, eps, N, sigma
        ).single_data["num_spec"]
        abs_loss = [
            ((a - b) / a) * 100.0
            for a, b in zip(
                continuous_data.get_all_data(
                    exp_name, integ_type, eps, N, sigma
                ).single_data["h_T"],
                continuous_data.get_all_data(
                    exp_name, integ_type, eps, N, sigma
                ).single_data["awae_differential"],
            )
        ]
        values = {1.0: "x", 5.0: "s"}
        # print(abs_loss)
        for value in values:
            for i, a in enumerate(abs_loss):
                if a < value:
                    res = (i, a)
                    break

            # res = min(enumerate(abs_loss), key=lambda x: abs(value - x[1]))

            # print(res)
            if sigma == sigma_vals[0] or sigma == sigma_vals[-1]:
                plt.vlines(
                    res[0] + 1, 0, res[1], alpha=0.5, color="black", linestyle="--"
                )
                plt.plot(
                    res[0] + 1, res[1], alpha=0.5, marker=values[value], color="black"
                )
            if sigma == sigma_vals[0]:
                plt.hlines(
                    res[1],
                    xmin=0,
                    xmax=res[0] + 1,
                    alpha=0.5,
                    color="black",
                    linestyle="--",
                )

        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$",
        )
        plot_lines.append([l1])
        if sigma == sigma_vals[0]:

            plt.xlim([0, max(spec[0:upper_bound]) + 0.25])
            plt.ylim([0, max(abs_loss[0:upper_bound]) + 0.25])

    # dists = [r"$(%s, %s)$" % (mu, n) for n in sigma_vals]

    legend1 = plt.legend(
        plot_lines[0],
        [r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 2.0))

    # plt.gca().add_artist(legend1)
    print(output_path + continuous_data.exp_name + "_relative_loss.pdf")
    plt.savefig(
        output_path + continuous_data.exp_name + "_relative_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + continuous_data.exp_name + "_relative_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.close("all")


def plot_all_continuous_fw_v_est(
    continuous_data,
    continuous_data_est,
    mu,
    sigma_vals,
    exp_name,
    integ_type,
    eps,
    upper_bound,
    output_path,
):
    fig, ax1 = plt.subplots()
    set_params_abs(continuous_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    # print(sigma_vals)
    plot_lines = []
    N = 8  # doesnt matter for continuous
    for sigma in sigma_vals:

        c = next(cc)
        alph = next(alphas)
        # print(exp_name, integ_type, eps, N, sigma)
        spec = continuous_data.get_all_data(
            "sh-v-diff-real", integ_type, eps, N, sigma
        ).single_data["num_spec"]
        abs_loss = [
            (a - b)
            for a, b in zip(
                continuous_data.get_all_data(
                    "sh-v-diff-real", integ_type, eps, N, sigma
                ).single_data["h_T"],
                continuous_data.get_all_data(
                    "sh-v-diff-real", integ_type, eps, N, sigma
                ).single_data["awae_differential"],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])

        c = next(cc)
        alph = next(alphas)
        spec = continuous_data_est.get_all_data("diff_est_real", mu, sigma).single_data[
            "num_spec"
        ]
        abs_loss = [
            (a - b)
            for a, b in zip(
                continuous_data_est.get_all_data(
                    "diff_est_real", mu, sigma
                ).single_data["h_T"],
                continuous_data_est.get_all_data(
                    "diff_est_real", mu, sigma
                ).single_data["awae_differential"],
            )
        ]
        (l2,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${{h}(\vec{X}_T) -  \hat{h}(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l2])

    # dists = [r"$(%s, %s)$" % (mu, n) for n in sigma_vals]
    dists = [
        r"$h(\vec{X}_T) - h(\vec{X}_T \mid X_T + X_S)$",
        r"${{h}(\vec{X}_T) -  \hat{h}(\vec{X}_T\mid X_T + X_S)}$",
    ]

    legend1 = plt.legend(
        plot_lines[0], [r"$(%s, %s)$" % (mu, n) for n in sigma_vals], loc="center right"
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.gca().add_artist(legend1)
    # plt.show()
    plt.savefig(
        output_path + "lognorm_fw_v_est_absolute_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + "lognorm_fw_v_est_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.close("all")


def plot_all_continuous_est(
    continuous_data, mu, sigma_vals, exp_name, upper_bound, output_path
):
    fig, ax1 = plt.subplots()
    set_params_abs(continuous_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []
    N = 8  # doesnt matter for continuous
    for sigma in sigma_vals:

        c = next(cc)
        alph = next(alphas)
        spec = continuous_data.get_all_data(exp_name, mu, sigma).single_data["num_spec"]
        abs_loss = [
            (a - b)
            for a, b in zip(
                continuous_data.get_all_data(exp_name, mu, sigma).single_data["h_T"],
                continuous_data.get_all_data(exp_name, mu, sigma).single_data[
                    "awae_differential"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])

    dists = [r"$(%s, %s)$" % (mu, n) for n in sigma_vals]

    legend1 = plt.legend(
        plot_lines[0],
        [r"$h(\vec{X}_T) - h(\vec{X}_T \mid X_T + X_S)$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + continuous_data.exp_name + "_absolute_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + continuous_data.exp_name + "_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_all(continuous_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])

    plot_lines = []

    for sigma in sigma_vals:

        c = next(cc)
        alph = next(alphas)
        spec = continuous_data.get_all_data(exp_name, mu, sigma).single_data["num_spec"]
        awae = continuous_data.get_all_data(exp_name, mu, sigma).single_data[
            "awae_differential"
        ]
        H_T = continuous_data.get_all_data(exp_name, mu, sigma).single_data["h_T"]

        (l1,) = ax1.plot(
            spec,
            awae,
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T\mid X_T + X_S)}$",
        )
        (l2,) = ax1.plot(
            spec,
            H_T,
            marker="",
            color=c,
            alpha=alph,
            linestyle="--",
            label=r"${h(\vec{X}_T)}$",
        )
        plot_lines.append([l1, l2])

    dists = [r"$(%s, %s)$" % (mu, n) for n in sigma_vals]

    legend1 = plt.legend(
        plot_lines[0],
        [r"$h(\vec{X}_T \mid X_T + X_S)$", r"$h(\vec{X}_T)$"],
        # bbo\vec{x}_to_anchor=(1.04, 1),
        loc="upper right",
    )
    plt.legend(
        [l[0] for l in plot_lines],
        dists,
        #    bbo\vec{x}_to_anchor=(1.04, 0),
        loc="lower right",
    )
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + continuous_data.exp_name + "_multiple_exps.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + continuous_data.exp_name + "_multiple_exps.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_rel(continuous_data.exp_name)
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    # print(continuous_data, N_vals,  upper_bound, output_path, lam)
    plot_lines = []

    for sigma in sigma_vals:

        c = next(cc)
        alph = next(alphas)
        spec = continuous_data.get_all_data(exp_name, mu, sigma).single_data["num_spec"]
        abs_loss = [
            ((a - b) / a) * 100.0
            for a, b in zip(
                continuous_data.get_all_data(exp_name, mu, sigma).single_data["h_T"],
                continuous_data.get_all_data(exp_name, mu, sigma).single_data[
                    "awae_differential"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker="",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$",
        )
        plot_lines.append([l1])

    dists = [r"$(%s, %s)$" % (mu, n) for n in sigma_vals]

    legend1 = plt.legend(
        plot_lines[0],
        [r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + continuous_data.exp_name + "_relative_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + continuous_data.exp_name + "_relative_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.close("all")


def plot_relative_loss(
    poisson_data,
    uniform_data,
    normal_data,
    ln_data_apx,
    ln_data_est,
    N_poisson,
    N_uniform,
    N_vals,
    lam,
    integ_type,
    eps,
    sigma,
    mu_real,
    sigma_real,
    upper_bound,
    output_path,
):

    discrete_data = {poisson_data: lam, uniform_data: None}
    # continuous_data = [normal_data, lognormal_data]
    continuous_data = [normal_data]
    N_params = [N_poisson, N_uniform]
    fig, ax1 = plt.subplots()
    set_params_rel()

    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "v"])
    linesty = itertools.cycle(["-", "--"])
    plot_lines = []

    continuous_N = N_poisson

    mark = next(markers)
    # for dat, N in zip(discrete_data, N_params):
    #     lsty = next(linesty)
    #     c = next(cc)
    #     alph = next(alphas)
    #     spec = dat.get_all_data(N, discrete_data[dat]).single_data["spectators"]
    #     rel_loss = [
    #         ((a - b) / a) * 100.0
    #         for a, b in zip(
    #             dat.get_all_data(N, discrete_data[dat]).single_data[
    #                 "target_init_entropy"
    #             ],
    #             dat.get_all_data(N, discrete_data[dat]).single_data["awae_results"],
    #         )
    #     ]
    #     (l1,) = ax1.plot(
    #         spec[0:upper_bound],
    #         rel_loss[0:upper_bound],
    #         marker=mark,
    #         color=c,
    #         alpha=alph,
    #         linestyle=lsty,
    #         label=r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$",
    #     )
    #     plot_lines.append([l1])

    # dists = [
    #     r"Pois$(\lambda = %s/2$)" % N_poisson,
    #     r"$\mathcal{U}\{0,%s-1\}$" % N_uniform,
    # ]
    # legend1 = plt.legend(
    #     plot_lines[0], [r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$"], loc="upper right"
    # )

    # plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    # plt.gca().add_artist(legend1)
    # plt.savefig(
    #     output_path
    #     + "discrete_relative_loss_"

    # plt.png(
    #     output_path , transparent=True
    #     + "discrete_relative_loss_"

    #     + str(N_poisson)
    #     + "_"
    #     + str(N_uniform)
    #     + ".pdf",
    #     bbox_inches="tight",
    # )
    # # plt.cla()

    # fig, ax1 = plt.subplots()
    # set_params_rel()
    # cc = itertools.cycle(colors)

    # plot_lines = []
    # for dat in continuous_data:
    #     c = next(cc)
    #     alph = next(alphas)
    #     spec = dat.get_all_data(
    #         "sh-v-diff", integ_type, eps, continuous_N, sigma
    #     ).single_data["num_spec"]
    #     rel_loss = [
    #         ((a - b) / a) * 100.0
    #         for a, b in zip(
    #             dat.get_all_data(
    #                 "sh-v-diff", integ_type, eps, continuous_N, sigma
    #             ).single_data["h_T"],
    #             dat.get_all_data(
    #                 "sh-v-diff", integ_type, eps, continuous_N, sigma
    #             ).single_data["awae_differential"],
    #         )
    #     ]
    #     (l1,) = ax1.plot(
    #         spec[0:upper_bound],
    #         rel_loss[0:upper_bound],
    #         marker=mark,
    #         color=c,
    #         alpha=alph,
    #         linestyle="-",
    #         label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$",
    #     )
    #     plot_lines.append([l1])

    # dists = [
    #     r"$\mathcal{N}(0, %s)$" % (sigma),
    #     r"$\log\mathcal{N}(0, %s)$" % (sigma),
    # ]
    # legend1 = plt.legend(
    #     plot_lines[0], [r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$"], loc="upper right"
    # )

    # plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    # plt.gca().add_artist(legend1)
    # plt.savefig(
    #     output_path
    #     + "continous_relative_loss_"

    # plt.png(
    #     output_path , transparent=True
    #     + "continous_relative_loss_"

    #     + str(continuous_N)
    #     + "_"
    #     + str(sigma)
    #     + ".pdf",
    #     bbox_inches="tight",
    # )

    fig, ax1 = plt.subplots()
    set_params_rel()
    cc = itertools.cycle(colors)
    # linesty = itertools.cycle(["-", "--"])

    plot_lines = []
    lsty = next(linesty)
    for discrete, N in zip(discrete_data, N_params):
        c = next(cc)
        alph = next(alphas)
        print(N, discrete_data[discrete])
        spec = discrete.get_all_data(N, discrete_data[discrete]).single_data[
            "spectators"
        ]
        rel_loss = [
            ((a - b) / a) * 100.0
            for a, b in zip(
                discrete.get_all_data(N, discrete_data[discrete]).single_data[
                    "target_init_entropy"
                ],
                discrete.get_all_data(N, discrete_data[discrete]).single_data[
                    "awae_results"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            rel_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle=lsty,
            label=r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$",
        )

        plot_lines.append([l1])

    mark = next(markers)

    for dat in continuous_data:
        lsty = next(linesty)

        c = next(cc)
        alph = next(alphas)
        spec = dat.get_all_data(
            "sh-v-diff", integ_type, eps, continuous_N, sigma
        ).single_data["num_spec"]
        rel_loss = [
            ((a - b) / a) * 100.0
            for a, b in zip(
                dat.get_all_data(
                    "sh-v-diff", integ_type, eps, continuous_N, sigma
                ).single_data["h_T"],
                dat.get_all_data(
                    "sh-v-diff", integ_type, eps, continuous_N, sigma
                ).single_data["awae_differential"],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            rel_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle=lsty,
            label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$",
        )
        plot_lines.append([l1])

    c = next(cc)
    alph = next(alphas)
    # lsty = next(linesty)

    # print("sh-v-diff-real", integ_type, eps, 8, 0.145542)
    spec = ln_data_apx.get_all_data(
        "sh-v-diff-real", integ_type, eps, 8, sigma_real
    ).single_data["num_spec"]
    abs_loss = [
        ((a - b) / a) * 100.0
        for a, b in zip(
            ln_data_apx.get_all_data(
                "sh-v-diff-real", integ_type, eps, 8, sigma_real
            ).single_data["h_T"],
            ln_data_apx.get_all_data(
                "sh-v-diff-real", integ_type, eps, 8, sigma_real
            ).single_data["awae_differential"],
        )
    ]
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        abs_loss[0:upper_bound],
        marker=mark,
        color=c,
        alpha=alph,
        linestyle=lsty,
        label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$",
    )
    plot_lines.append([l1])

    dists = [
        r"Pois$(%s$)" % str(int(N_poisson / 2)),
        r"$\mathcal{U}(0,%s)$" % str(N_uniform - 1),
        r"$\mathcal{N}(0, %s)$" % str(int(sigma)),
        r"$\log\mathcal{N}_{\text{FW}}(%.2f, %.2f)$" % (mu_real, sigma_real),
    ]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            alpha=alph,
            linestyle="-",
            label=r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="v",
            alpha=alph,
            linestyle="-",
            label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$",
        ),
    ]

    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper right",
    )

    flat_plot_lines = [x for xs in plot_lines for x in xs]
    # print([[l[0], l[1]] for l in plot_lines])
    plt.legend(flat_plot_lines, dists, loc="upper right")
    # plt.gca().add_artist(legend1)
    print(
        output_path
        + "discrete_continuous_relative_loss_"
        + str(N_poisson)
        + "_"
        + str(N_uniform)
        + "_"
        + str(sigma)
        + ".pdf"
    )
    plt.savefig(
        output_path + "discrete_continuous_relative_loss" + ".pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + "discrete_continuous_relative_loss" + ".png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.close("all")
    return 0


def plot_absolute_loss(
    poisson_data,
    uniform_data,
    normal_data,
    N_poisson,
    N_uniform,
    N_vals,
    lam,
    integ_type,
    eps,
    sigma,
    upper_bound,
    output_path,
):

    discrete_data = {poisson_data: lam, uniform_data: None}
    continuous_data = [normal_data]

    fig, ax1 = plt.subplots()
    set_params_abs()

    N_params = [N_poisson, N_uniform]

    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "v"])
    plot_lines = []

    mark = next(markers)
    for dat, N in zip(discrete_data, N_params):
        c = next(cc)
        alph = next(alphas)
        spec = dat.get_all_data(N, discrete_data[dat]).single_data["spectators"]
        abs_loss = [
            ((a - b))
            for a, b in zip(
                dat.get_all_data(N, discrete_data[dat]).single_data[
                    "target_init_entropy"
                ],
                dat.get_all_data(N, discrete_data[dat]).single_data["awae_results"],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$",
        )
        plot_lines.append([l1])

    dists = [
        r"Pois$(\lambda = %s$)" % str(int(N_poisson / 2)),
        r"$\mathcal{U}(0,%s)$" % N_uniform - 1,
    ]
    legend1 = plt.legend(
        plot_lines[0],
        [r"$\frac{H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}{H(\vec{X}_T)}$"],
        loc="upper right",
    )

    plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + "discrete_absolute_loss_" + str(N) + ".pdf", bbox_inches="tight"
    )

    plt.savefig(
        output_path + "discrete_absolute_loss_" + str(N) + ".png",
        bbox_inches="tight",
        transparent=True,
    )

    # plt.cla()

    fig, ax1 = plt.subplots()
    set_params_abs()
    cc = itertools.cycle(colors)

    plot_lines = []
    for dat in continuous_data:
        c = next(cc)
        alph = next(alphas)
        spec = dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
            "num_spec"
        ]
        abs_loss = [
            (a - b)
            for a, b in zip(
                dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
                    "h_T"
                ],
                dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
                    "awae_differential"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$",
        )
        plot_lines.append([l1])

    dists = [
        r"$\mathcal{N}(0, %s)$" % (sigma),
        # r"$\log\mathcal{N}(0, %s)$" % (sigma),
    ]
    legend1 = plt.legend(
        plot_lines[0],
        [r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}{h(\vec{X}_T)}$"],
        loc="upper right",
    )

    plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path
        + "continous_absolute_loss_"
        + str(N_poisson)
        + "_"
        + str(N_uniform)
        + "_"
        + str(sigma)
        + ".pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path
        + "continous_absolute_loss_"
        + str(N_poisson)
        + "_"
        + str(N_uniform)
        + "_"
        + str(sigma)
        + ".png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    set_params_abs()
    cc = itertools.cycle(colors)

    plot_lines = []
    for discrete, N in zip(discrete_data, N_params):
        c = next(cc)
        alph = next(alphas)
        spec = discrete.get_all_data(N, discrete_data[discrete]).single_data[
            "spectators"
        ]
        abs_loss = [
            (a - b)
            for a, b in zip(
                discrete.get_all_data(N, discrete_data[discrete]).single_data[
                    "target_init_entropy"
                ],
                discrete.get_all_data(N, discrete_data[discrete]).single_data[
                    "awae_results"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
        )

        plot_lines.append([l1])

    mark = next(markers)

    for dat in continuous_data:
        c = next(cc)
        alph = next(alphas)
        spec = dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
            "num_spec"
        ]
        abs_loss = [
            (a - b)
            for a, b in zip(
                dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
                    "h_T"
                ],
                dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
                    "awae_differential"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])

    dists = [
        r"Pois$(%s/2$)" % N_poisson,
        r"$\mathcal{U}(0,%s-1)$" % N_uniform,
        r"$\mathcal{N}(0, %s)$" % (sigma),
        # r"$\log\mathcal{N}(0, %s)$" % (sigma),
    ]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="v",
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        ),
    ]

    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper right",
    )

    flat_plot_lines = [x for xs in plot_lines for x in xs]
    # print([[l[0], l[1]] for l in plot_lines])
    plt.legend(flat_plot_lines, dists, loc="center right")
    plt.gca().add_artist(legend1)
    plt.savefig(
        output_path
        + "discrete_continuous_absolute_loss_"
        + str(N_poisson)
        + "_"
        + str(N_uniform)
        + "_"
        + str(sigma)
        + ".pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path
        + "discrete_continuous_absolute_loss_"
        + str(N_poisson)
        + "_"
        + str(N_uniform)
        + "_"
        + str(sigma)
        + ".png",
        bbox_inches="tight",
        transparent=True,
    )

    # fig, ax1 = plt.subplots()
    # set_params_abs()
    # cc = itertools.cycle(colors)

    # # plt.yscale("log")

    # plot_lines = []
    # set_params_abs()
    # cc = itertools.cycle(colors)

    # plot_lines = []
    # for _N in N_vals:
    #     c = next(cc)
    #     alph = next(alphas)
    #     spec = poisson_data.get_all_data(_N, _N / 2).single_data["spectators"]
    #     abs_loss = [
    #         (a - b)
    #         for a, b in zip(
    #             poisson_data.get_all_data(_N, _N / 2).single_data[
    #                 "target_init_entropy"
    #             ],
    #             poisson_data.get_all_data(_N, _N / 2).single_data["awae_results"],
    #         )
    #     ]
    #     (l1,) = ax1.plot(
    #         spec[0:upper_bound],
    #         abs_loss[0:upper_bound],
    #         marker=mark,
    #         color=c,
    #         alpha=alph,
    #         linestyle="-",
    #         label=r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
    #     )

    #     plot_lines.append([l1])

    # mark = next(markers)

    # plt.savefig(
    #     output_path
    #     + "absolute_loss_pois_normal_"

    # plt.png(
    #     output_path , transparent=True
    #     + "absolute_loss_pois_normal_"

    #     + str(N_poisson)
    #     + "_"
    #     + str(N_uniform)
    #     + "_"
    #     + str(sigma)
    #     + ".pdf",
    #     bbox_inches="tight",
    # )

    plt.close("all")
    return 0


def plot_absolute_loss_ln(
    poisson_data,
    uniform_data,
    normal_data,
    ln_data_apx,
    ln_data_est,
    N_poisson,
    N_uniform,
    N_vals,
    lam,
    integ_type,
    eps,
    sigma,
    mu_real,
    sigma_real,
    upper_bound,
    output_path,
):

    discrete_data = {poisson_data: lam, uniform_data: None}
    continuous_data = [normal_data]
    N_params = [N_poisson, N_uniform]

    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "v", "x"])
    plot_lines = []
    fig, ax1 = plt.subplots()
    set_params_abs()
    cc = itertools.cycle(colors)
    linesty = itertools.cycle(["-", "--"])

    plot_lines = []
    mark = next(markers)
    lsty = next(linesty)

    for discrete, N in zip(discrete_data, N_params):
        c = next(cc)
        alph = next(alphas)
        spec = discrete.get_all_data(N, discrete_data[discrete]).single_data[
            "spectators"
        ]
        abs_loss = [
            (a - b)
            for a, b in zip(
                discrete.get_all_data(N, discrete_data[discrete]).single_data[
                    "target_init_entropy"
                ],
                discrete.get_all_data(N, discrete_data[discrete]).single_data[
                    "awae_results"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
        )

        plot_lines.append([l1])

    mark = next(markers)
    lsty = next(linesty)

    for dat in continuous_data:
        c = next(cc)
        alph = next(alphas)
        spec = dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
            "num_spec"
        ]
        abs_loss = [
            (a - b)
            for a, b in zip(
                dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
                    "h_T"
                ],
                dat.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data[
                    "awae_differential"
                ],
            )
        ]
        (l1,) = ax1.plot(
            spec[0:upper_bound],
            abs_loss[0:upper_bound],
            marker=mark,
            color=c,
            alpha=alph,
            linestyle="--",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])

    c = next(cc)
    alph = next(alphas)
    # print("sh-v-diff-real", integ_type, eps, 8, 0.145542)
    spec = ln_data_apx.get_all_data(
        "sh-v-diff-real", integ_type, eps, 8, sigma_real
    ).single_data["num_spec"]
    abs_loss = [
        (a - b)
        for a, b in zip(
            ln_data_apx.get_all_data(
                "sh-v-diff-real", integ_type, eps, 8, sigma_real
            ).single_data["h_T"],
            ln_data_apx.get_all_data(
                "sh-v-diff-real", integ_type, eps, 8, sigma_real
            ).single_data["awae_differential"],
        )
    ]
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        abs_loss[0:upper_bound],
        marker=mark,
        color=c,
        alpha=alph,
        linestyle="--",
        label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
    )
    plot_lines.append([l1])

    mark = next(markers)
    c = next(cc)
    alph = next(alphas)
    # print("sh-v-diff-real", integ_type, eps, 8, sigma_real)
    spec = ln_data_est.get_all_data("diff_est_real", mu_real, sigma_real).single_data[
        "num_spec"
    ]
    abs_loss = [
        (a - b)
        for a, b in zip(
            ln_data_est.get_all_data("diff_est_real", mu_real, sigma_real).single_data[
                "h_T"
            ],
            ln_data_est.get_all_data("diff_est_real", mu_real, sigma_real).single_data[
                "awae_differential"
            ],
        )
    ]

    dists = [
        r"Pois$(%s$)" % str(int(N_poisson / 2)),
        r"$\mathcal{U}(0,%s)$" % str(N_uniform - 1),
        r"$\mathcal{N}(0, %s)$" % str(int(sigma)),
        r"$\log\mathcal{N}_{\text{FW}}(%.2f, %.2f)$" % (mu_real, sigma_real),
        # r"$\log\mathcal{N}_{\text{KL}}(%.2f, %.2f)$" % (mu_real, sigma_real),
    ]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            alpha=alph,
            linestyle="-",
            label=r"${H(\vec{X}_T) -  H(\vec{X}_T\mid X_T + X_S)}$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="v",
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     color="black",
        #     marker="x",
        #     alpha=alph,
        #     linestyle="-",
        #     label=r"${{h}(\vec{X}_T) - \hat{h}(\vec{X}_T\mid X_T + X_S)}$",
        # ),
    ]

    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper right",
        # prop={'size': 13}
    )

    flat_plot_lines = [x for xs in plot_lines for x in xs]
    plt.legend(
        flat_plot_lines,
        dists,
        loc="upper right",
        #    prop={'size': 13}
    )
    # plt.gca().add_artist(legend1)
    plt.savefig(
        output_path + "dscrt_cnt_abs_ln" + ".pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + "dscrt_cnt_abs_ln" + ".png", bbox_inches="tight", transparent=True
    )

    plt.close("all")

    return 0


def plot_gamma(gamma_data, normal_data, lognormal_data):
    eps = -5
    integ_type = "gsl"
    sigma_real = 0.145542
    mu_real = 1.6702

    upper_bound = 25
    spec = gamma_data.class_data["num_spec"]
    abs_loss = [
        (a - b)
        for a, b in zip(
            gamma_data.class_data["h_T"],
            gamma_data.class_data["awae_differential"],
        )
    ]

    fig, ax1 = plt.subplots()
    set_params_abs(gamma_data.exp_name)
    markers = itertools.cycle(["o", "v", "x"])

    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    c = next(cc)
    mark = next(markers)
    alph = next(alphas)
    plot_lines = []
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        abs_loss[0:upper_bound],
        marker=mark,
        color=c,
        alpha=alph,
        linestyle="-",
        label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
    )
    plot_lines.append([l1])

    c = next(cc)
    alph = next(alphas)
    mark = next(markers)

    spec = normal_data.get_all_data("sh-v-diff", integ_type, eps, 8, 1.1).single_data[
        "num_spec"
    ]
    abs_loss = [
        (a - b)
        for a, b in zip(
            normal_data.get_all_data("sh-v-diff", integ_type, eps, 8, 1.1).single_data[
                "h_T"
            ],
            normal_data.get_all_data("sh-v-diff", integ_type, eps, 8, 1.1).single_data[
                "awae_differential"
            ],
        )
    ]
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        abs_loss[0:upper_bound],
        marker=mark,
        color=c,
        alpha=alph,
        linestyle="--",
        label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
    )
    plot_lines.append([l1])
    mark = next(markers)

    c = next(cc)
    alph = next(alphas)
    # print("sh-v-diff-real", integ_type, eps, 8, 0.145542)
    spec = lognormal_data.get_all_data(
        "sh-v-diff-real", integ_type, eps, 8, sigma_real
    ).single_data["num_spec"]
    abs_loss = [
        (a - b)
        for a, b in zip(
            lognormal_data.get_all_data(
                "sh-v-diff-real", integ_type, eps, 8, sigma_real
            ).single_data["h_T"],
            lognormal_data.get_all_data(
                "sh-v-diff-real", integ_type, eps, 8, sigma_real
            ).single_data["awae_differential"],
        )
    ]
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        abs_loss[0:upper_bound],
        marker=mark,
        color=c,
        alpha=alph,
        linestyle="--",
        label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
    )
    plot_lines.append([l1])



    dists = [
        r"Gamma$(%s, %s)$" % (7.3854, 0.7710),
        r"$\mathcal{N}(0, %s)$" % str((1.1)),
        r"$\log\mathcal{N}_{\text{FW}}(%.2f, %.2f)$" % (mu_real, sigma_real),
    ]

    legend1 = plt.legend(
        plot_lines[0],
        [r"$h(\vec{X}_T) - h(\vec{X}_T \mid X_T + X_S)$"],
        loc="upper right",
    )
    plt.legend([l[0] for l in plot_lines], dists, loc="center right")
    plt.gca().add_artist(legend1)
    # plt.show()
    dir_path = "../../proof/figs/single_computation/"
    
    plt.savefig(
        dir_path + "gamma_norm_lognorm.pdf", bbox_inches="tight",
    )

    plt.savefig(
        dir_path + "gamma_norm_lognorm.png", bbox_inches="tight", transparent=True
    )

    return 0


def plot_main():
    poisson_data = single_computation_discrete(
        output_dir, "sum_poisson", discrete_col_names
    )
    poisson_data_min = single_computation_discrete(
        output_dir, "sum_poisson_min", discrete_col_names
    )
    uniform_data = single_computation_discrete(
        output_dir, "sum_uniform", discrete_col_names
    )
    uniform_data_min = single_computation_discrete(
        output_dir, "sum_uniform", discrete_col_names
    )
    normal_data = single_computation_continuous(
        output_dir, "normal_v2", continous_col_names
    )
    lognormal_data_apx = single_computation_continuous(
        output_dir, "lognormal_v2", continous_col_names
    )
    lognormal_data_est = single_computation_continuous_est(
        output_dir, "lognormal_v2_est", continous_col_names_est
    )
    gamma_data = gamma_datatype(output_dir, "gamma", gamma_col_names)
    # plot_gamma(gamma_data, normal_data, lognormal_data_apx)

    N = 128
    lam = N / 2
    sigma = 0.25
    eps = -5
    upper_bound = 25
    integ_type = "gsl"
    dir_path = "../../proof/figs/single_computation/"

    # print(lognormal_data.class_data)
    # print(
    #     lognormal_data.get_all_data("diff_est",  0.0, 16.0).single_data["num_spec"]
    # )

    # print(
    #     lognormal_data.get_all_data("sh-v-diff", integ_type, eps, N, sigma).single_data["num_spec"]
    # )

    # N_vals = [8, 16, 32, 64, 128, 256 ]
    N_vals = [8, 16, 32, 64]
    # N_vals = [64, 128, 256, 512, 1024]
    # plot_all_discrete(poisson_data, N_vals, upper_bound, dir_path)
    # plot_all_discrete_min(poisson_data_min, N_vals, upper_bound, dir_path)
    plot_all_discrete_min_v_shannon(
        poisson_data, poisson_data_min, N_vals, upper_bound, dir_path
    )

    N_vals = [8]
    # plot_all_discrete_min(uniform_data_min, N_vals, upper_bound, dir_path)
    # print()

    # plot_all_discrete(uniform_data, N_vals, upper_bound, dir_path)
    # sigma_vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    sigma_vals = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    # plot_all_continuous(normal_data, 0.0,  sigma_vals, 'sh-v-diff', integ_type, eps, upper_bound, dir_path)
    # sigma_vals = [0.25, 0.5, 1.0, 2.0, 4.0]
    # plot_all_continuous(lognormal_data, sigma_vals, integ_type, eps, upper_bound, dir_path)

    # plot_all_continuous_est(lognormal_data, 0.0,  sigma_vals, 'diff_est',  upper_bound, dir_path)
    # sigma_vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    # sigma_vals = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

    sigma_real = 0.145542
    # print(sigma)
    mu_real = 1.6702
    # sv = [sigma]
    # plot_all_continuous_est(lognormal_data_est, mu_real,  [sigma_real], 'diff_est_real',  10, dir_path)

    # plot_all_continuous( lognormal_data_apx, mu_real, [sigma_real], "sh-v-diff-real", integ_type, eps, upper_bound, dir_path )

    N_poisson = 8
    lam = N_poisson / 2
    N_uniform = 8
    sigma_N = 4.0

    # plot_relative_loss(
    #     poisson_data,
    #     uniform_data,
    #     normal_data,
    #     lognormal_data_apx,
    #     lognormal_data_est,
    #     N_poisson,
    #     N_uniform,
    #     N_vals,
    #     lam,
    #     integ_type,
    #     eps,
    #     sigma_N,
    #     mu_real,
    #     sigma_real,
    #     10,
    #     dir_path,
    # )

    # plot_absolute_loss_ln(
    #     poisson_data,
    #     uniform_data,
    #     normal_data,
    #     lognormal_data_apx,
    #     lognormal_data_est,
    #     N_poisson,
    #     N_uniform,
    #     N_vals,
    #     lam,
    #     integ_type,
    #     eps,
    #     sigma_N,
    #     mu_real,
    #     sigma_real,
    #     10,
    #     dir_path,
    # )

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_main()
