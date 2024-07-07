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


colors = [
    "blue",
    "red",
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


logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager

mpl.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb}",
    }
)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 16})
rc("text", usetex=True)


def plot_vary_spec(
    joint_data, sigma_vals, sigma, spec_intervals, upper_bound, output_path
):
    fig, ax1 = plt.subplots()

    # plt.title(
    #     r"$O_1 = \vec{X}_T + X_S + X_{S_1}$, $O_2 = \vec{X}_T + X_S + X_{S_2}$"
    #     + "\n"
    #     + r"$\mathcal{N}(0, %s), $" % sigma
    #     + r" Varying $\lvert X_S\rvert$"
    # )
    plt.ylabel(r"Entropy (bits)")
    # plt.xlabel(r"No. spectators per experiment ($|S_{12} \cup S_{1}|=|S_{12} \cup S_2|$)")
    plt.xlabel(r"No. spectators per experiment")

    shared_spec = joint_data.get_all_data(sigma).single_data["num_shared"]
    unique_1 = joint_data.get_all_data(sigma).single_data["num_spec_1"]
    h_t_mid_o1_o2 = joint_data.get_all_data(sigma).single_data["diff_ent_cond"]
    h_t = joint_data.get_all_data(sigma).single_data["target_init"]
    awae = joint_data.get_all_data(sigma).single_data["awae"]

    alph = 0.5
    cc = itertools.cycle(colors)

    max_shared = int(shared_spec[-1] - shared_spec[0] + 1)
    max_unique = int(unique_1[-1] - unique_1[0] + 1)
    plot_lines = []
    c = next(cc)

    # plot_lines.append([l0])

    (l0,) = ax1.plot(
        unique_1[0:(upper_bound)],
        h_t[:(upper_bound)],
        marker="",
        color="magenta",
        alpha=alph,
        linestyle="-",
        label=r"$h(\vec{X}_T)$",
    )

    # x_data = [
    #     x  for x in unique_1[max_unique * spec : max_unique * (spec + 1)]
    # ]
    # y_data_2 = [x for x in awae[max_unique * spec : max_unique * (spec + 1)]]
    (l1,) = ax1.plot(
        unique_1[:(upper_bound)],
        awae[:(upper_bound)],
        marker="x",
        color="darkgreen",
        alpha=alph,
        linestyle="--",
        label=r"$h(\vec{X}_T \mid O_1)$",
    )
    flag = 0
    for spec in spec_intervals:
        x_data = [
            x + spec for x in unique_1[max_unique * spec : max_unique * (spec + 1)]
        ]
        y_data = [x for x in h_t_mid_o1_o2[max_unique * spec : max_unique * (spec + 1)]]
        # y_data_2 = [x for x in awae[max_unique * spec : max_unique * (spec + 1)]]
        # print(x_data)
        # print(len(x_data))
        # print(x_data[:(upper_bound - spec)])
        # print(len(x_data[:(upper_bound - spec)]))
        # print(y_data)
        # print(y_data[:upper_bound - spec])

        (l2,) = ax1.plot(
            x_data[: (upper_bound - spec)],
            y_data[: (upper_bound - spec)],
            marker="+",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$h(\vec{X}_T \mid O_1, O_2)$",
        )
        # if flag == 0:
        #     plot_lines.append([l0, l1, l2])
        #     flag = 1
        # else:

        plot_lines.append([l2])
        c = next(cc)

    dists = [r"$\lvert {S}_{12} \rvert = %s$" % n for n in spec_intervals]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="magenta",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$h(\vec{X}_T)$",
        ),
        Line2D(
            [0],
            [0],
            color="darkgreen",
            marker="x",
            alpha=alph,
            linestyle="--",
            label=r"$h(\vec{X}_T \mid O_1)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="+",
            alpha=alph,
            linestyle="-",
            label=r"$h(\vec{X}_T \mid O_1, O_2)$",
        ),
    ]

    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"$h(\vec{X}_T )$", r"$h(\vec{X}_T \mid O_1)$", r"$h(\vec{X}_T \mid O_1, O_2)$"],
    #     loc="upper right",
    # )
    legend1 = plt.legend(
        handles=legend_elements,
        # prop={"size": 14},
        loc="center right",
    )

    plt.legend(
        [l[0] for l in plot_lines],
        dists,
        #    prop={"size": 14},
        loc="lower right",
    )

    plt.gca().add_artist(legend1)
    plt.savefig( output_path + joint_data.exp_name + "_" + joint_data.sub_exp + ".pdf", bbox_inches="tight", )
    plt.savefig( output_path + joint_data.exp_name + "_" + joint_data.sub_exp + ".png", bbox_inches="tight", transparent=True)

    plt.close("all")

    return 0


def plot_one_vs_two_exp(
    joint_data_one_exp, joint_data_one_exp_second, joint_data_two_exp, total_num_spec, sigma, output_path
):
    fig, ax1 = plt.subplots()

    # plt.title(
    #     r"$O_1= \vec{X}_T + X_S + X_{S_1}$" + "\n"
    #     r"$O_2= \vec{X}_T + X_S + X_{S_2}$, $O_2' = X_S + X_{S_2}$"
    #     + "\n"
    #     + r"$\mathcal{N}(0, %s), $" % sigma
    #     + r" Participating Once v. Twice"
    # )
    shared_spec_one = joint_data_one_exp.get_all_data(sigma).single_data["num_shared"]
    unique_1_one = joint_data_one_exp.get_all_data(sigma).single_data["num_spec_1"]
    h_t_one = joint_data_one_exp.get_all_data(sigma).single_data["target_init"]
    awae_one = joint_data_one_exp.get_all_data(sigma).single_data["awae"]

    h_t_mid_o1_o2_one = joint_data_one_exp.get_all_data(sigma).single_data[
        "diff_ent_cond"
    ]

    h_t_mid_o1_o2_two = joint_data_two_exp.get_all_data(sigma).single_data[
        "diff_ent_cond"
    ]

    h_t_mid_o1_o2_one_second = joint_data_one_exp_second.get_all_data(sigma).single_data[
        "diff_ent_cond"
    ]

    x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

    plt.ylabel(r"Entropy (bits)")
    # plt.xlabel(r"Fraction of shared spectators ($n = %s$)" % total_num_spec)
    plt.xlabel(r"Fraction of shared spectators")

    (l2,) = plt.plot(
        x_vals,
        h_t_one,
        marker="",
        color=colors[1],
        alpha=0.5,
        linestyle="--",
        # label=r"$\sigma^2 = %s, h(X_{T})$" % data[1, 1],
        label=r"Initial ",
        # label=r"$h(\vec{X}_{T})$",
    )

    (l2,) = plt.plot(
        x_vals,
        awae_one,
        marker=""InpherInpher,
        color=colors[2],
        alpha=0.8,
        linestyle="dotted",
        label=r"First evaluation",
        # label=r"$h(\vec{X}_{T}{\mid} O_1)$",
    )

    (l2,) = plt.plot(
        x_vals,
        h_t_mid_o1_o2_two,
        marker="x",
        color=colors[0],
        linestyle="-",
        alpha=0.5,
        label=r"Participating twice",
        # label=r"$h(\vec{X}_{T} {\mid} O_1, O_2)$",
    )

    (l2,) = plt.plot(
        x_vals,
        h_t_mid_o1_o2_one,
        marker="o",
        color=colors[3],
        alpha=0.5,
        linestyle="-",
        label=r"Participating in the first only"
        # + "\n"
        # + r"$h(\vec{X}_{T} {\mid} O_1, O_2')$",
        # label=r"$h(\vec{X}_{T} {\mid} O_1, O_2')$",
    )
    
    (l2,) = plt.plot(
        x_vals,
        h_t_mid_o1_o2_one_second,
        marker="s",
        color=colors[4],
        alpha=0.5,
        linestyle="--",
        label=r"Participating in the second only"
        # + "\n"
        # + r"$h(\vec{X}_{T} {\mid} O_1', O_2)$ (equivalent to $h(\vec{X}_{T} {\mid} O_1, O_2')$)",
        # label=r"$h(\vec{X}_{T} {\mid} O_1', O_2)$",
    )
    # # if total_num_spec == 10:

    # comment out in order to print legend separately
    # ax1.legend(
    #     loc="lower left",
    #     # bbox_to_anchor=(-0.1, -0.68),
    #     bbox_to_anchor=(1.0, 0.3),
    #     prop={"size": 17},
    #     # borderpad=0.5,
    #     # handlelength=1.0,
    # )

    plt.savefig( output_path + "one_vs_two_exp_with_second_" + str(total_num_spec) + ".pdf", bbox_inches="tight", )
    plt.savefig( output_path + "one_vs_two_exp_with_second_" + str(total_num_spec) + ".png", bbox_inches="tight", transparent=True)

 
    ax1.legend(
        loc="lower left",
        # bbox_to_anchor=(-0.1, -0.68),
        # bbox_to_anchor=(0, -0.15),
        prop={"size": 18},
        ncol=2,
        columnspacing=0.0,
        handlelength=1.2,
        borderpad=0.1,
        # handlelength=1.0,
    )

    plt.savefig( output_path + "one_vs_two_exp_with_second_with_legend_" + str(total_num_spec) + ".pdf", bbox_inches="tight", )
    plt.savefig( output_path + "one_vs_two_exp_with_second_with_legend_" + str(total_num_spec) + ".png", bbox_inches="tight", transparent=True)

    label_params = ax1.get_legend_handles_labels()
    plt.autoscale(enable=True, axis="y", tight=True)
    figl2, axl2 = plt.subplots(figsize=(0, 0))
    axl2.axis(False)
    axl2.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), ncol=2)
    figl2.savefig(
        output_path + "legend_text_only"
        # + str(total_num_spec)
        + ".pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    figl2.savefig(
        output_path + "legend_text_only"
        # + str(total_num_spec)
        + ".png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
#     fig, ax1 = plt.subplots()

#     # plt.title(
#     #     r"$O_1= \vec{X}_T + X_S + X_{S_1}$" + "\n"
#     #     r"$O_2= \vec{X}_T + X_S + X_{S_2}$, $O_2' = X_S + X_{S_2}$"
#     #     + "\n"
#     #     + r"$\mathcal{N}(0, %s), $" % sigma
#     #     + r" Participating Once v. Twice"
#     # )
#     # shared_spec_one = joint_data_one_exp.get_all_data(sigma).single_data["num_shared"]
#     # unique_1_one = joint_data_one_exp.get_all_data(sigma).single_data["num_spec_1"]
#     # h_t_one = joint_data_one_exp.get_all_data(sigma).single_data["target_init"]
#     # awae_one = joint_data_one_exp.get_all_data(sigma).single_data["awae"]

#     # h_t_mid_o1_o2_one = joint_data_one_exp.get_all_data(sigma).single_data[
#     #     "diff_ent_cond"
#     # ]

#     # h_t_mid_o1_o2_two = joint_data_two_exp.get_all_data(sigma).single_data[
#     #     "diff_ent_cond"
#     # ]
#     abs_loss_zero = [((a - b) / a) * 100.0 for a, b in zip(h_t_one, awae_one)]
#     abs_loss_one = [((a - b) / a) * 100.0 for a, b in zip(h_t_one, h_t_mid_o1_o2_one)]
#     abs_loss_two = [((a - b) / a) * 100.0 for a, b in zip(h_t_one, h_t_mid_o1_o2_two)]

#     abs_loss_3 = [((a - b) / a) * 100.0 for a, b in zip(awae_one, h_t_mid_o1_o2_one)]
#     abs_loss_4 = [((a - b) / a) * 100.0 for a, b in zip(awae_one, h_t_mid_o1_o2_two)]
#     # print("abs_loss_zero", abs_loss_zero)
#     # print("abs_loss_one", abs_loss_one)
#     # print("abs_loss_two", abs_loss_two)
#     # print("abs_loss_3", abs_loss_3)
#     # print("abs_loss_4", abs_loss_4)
#     # print("i","abs_loss_zero", "abs_loss_one", "abs_loss_two", "abs_loss_3", "abs_loss_4")
#     # for i in range(len(abs_loss_zero)):
#     #     if x_vals[i] >0.4 and x_vals[i] < 0.6:
#     #         print(total_num_spec, i, x_vals[i], abs_loss_one[i], abs_loss_two[i])
#     # print()
#     x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

#     plt.ylabel(r"Percent change (\%)")
#     # plt.xlabel(r"Fraction of shared spectators ($n = %s$)" % total_num_spec)
#     plt.xlabel(r"Fraction of shared spectators")

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_zero,
#         marker="x",
#         color=colors[4],
#         linestyle="-",
#         alpha=0.5,
#         # label=r"$h(X_{T} \mid O_1, O_2)$",
#         label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}{h(\vec{X}_T)}$",
#     )

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_two,
#         marker="x",
#         color=colors[0],
#         linestyle="-",
#         alpha=0.5,
#         # label=r"$h(X_{T} \mid O_1, O_2)$",
#         label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2)}{h(\vec{X}_T)}$",
#     )

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_one,
#         marker="o",
#         color=colors[3],
#         alpha=0.5,
#         linestyle="-",
#         # label=r"$h(X_{T} \mid O_1, O_2')$",
#         label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2')}{h(\vec{X}_T)}$",
#     )
   
#     plt.axhline(y=5, color='r', linestyle='-')
#     plt.axhline(y=1, color='r', linestyle='-')


#     ax1.legend(
#         loc="upper left",
#         bbox_to_anchor=(0.0, -0.17),
#         ncol=2
#         # borderpad=0.5,
#         # handlelength=1.0,
#     )
#     plt.savefig( output_path + "one_vs_two_exp_relative_loss_" + str(total_num_spec) + ".pdf", bbox_inches="tight", )
#     plt.savefig( output_path + "one_vs_two_exp_relative_loss_" + str(total_num_spec) + ".png", bbox_inches="tight", transparent=True)
#     # plt.show()
#     label_params = ax1.get_legend_handles_labels()

#     plt.autoscale(enable=True, axis="y", tight=True)
#     figl, axl = plt.subplots(figsize=(0, 0))
#     # axl.margins(x=0)
#     # axl.margins(y=0)
#     axl.axis(False)
#     axl.legend(*label_params, ncol=3)
#     figl.savefig(
#         output_path + "legend_relative" + ".pdf", bbox_inches="tight", pad_inches=0
#     )
#     figl.savefig(
#         output_path + "legend_relative" + ".png", bbox_inches="tight", pad_inches=0,        transparent=True,
#     )

#     fig, ax1 = plt.subplots()

#     # plt.title(
#     #     r"$O_1= \vec{X}_T + X_S + X_{S_1}$" + "\n"
#     #     r"$O_2= \vec{X}_T + X_S + X_{S_2}$, $O_2' = X_S + X_{S_2}$"
#     #     + "\n"
#     #     + r"$\mathcal{N}(0, %s), $" % sigma
#     #     + r" Participating Once v. Twice"
#     # )
#     # shared_spec_one = joint_data_one_exp.get_all_data(sigma).single_data["num_shared"]
#     # unique_1_one = joint_data_one_exp.get_all_data(sigma).single_data["num_spec_1"]
#     # h_t_one = joint_data_one_exp.get_all_data(sigma).single_data["target_init"]
#     # awae_one = joint_data_one_exp.get_all_data(sigma).single_data["awae"]

#     # h_t_mid_o1_o2_one = joint_data_one_exp.get_all_data(sigma).single_data[
#     #     "diff_ent_cond"
#     # ]

#     # h_t_mid_o1_o2_two = joint_data_two_exp.get_all_data(sigma).single_data[
#     #     "diff_ent_cond"
#     # ]
#     abs_loss_zero = [(a - b) for a, b in zip(h_t_one, awae_one)]
#     abs_loss_one = [(a - b) for a, b in zip(h_t_one, h_t_mid_o1_o2_one)]
#     abs_loss_two = [(a - b) for a, b in zip(h_t_one, h_t_mid_o1_o2_two)]

#     abs_loss_3 = [(a - b) for a, b in zip(awae_one, h_t_mid_o1_o2_one)]
#     abs_loss_4 = [(a - b) for a, b in zip(awae_one, h_t_mid_o1_o2_two)]

#     # print("i","abs_loss_zero", "abs_loss_one", "abs_loss_two", "abs_loss_3", "abs_loss_4")
#     # for i in range(len(abs_loss_zero)):
#     #     print(i, abs_loss_zero[i], abs_loss_one[i], abs_loss_two[i], abs_loss_3[i], abs_loss_4[i]
#     #     )
#     # print("abs_loss_zero", abs_loss_zero)
#     # print("abs_loss_one", abs_loss_one)
#     # print("abs_loss_two", abs_loss_two)
#     # print("abs_loss_3", abs_loss_3)
#     # print("abs_loss_4", abs_loss_4)

#     x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

#     plt.ylabel(r"Entropy (bits)")
#     plt.xlabel(r"Percentage of shared spectators ($n = %s$)" % total_num_spec)

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_zero,
#         marker="x",
#         color=colors[4],
#         linestyle="-",
#         alpha=0.5,
#         # label=r"$h(X_{T} \mid O_1, O_2)$",
#         label=r"$h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)$",
#     )

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_two,
#         marker="x",
#         color=colors[0],
#         linestyle="-",
#         alpha=0.5,
#         # label=r"$h(X_{T} \mid O_1, O_2)$",
#         label=r"$h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2)$",
#     )

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_one,
#         marker="o",
#         color=colors[3],
#         alpha=0.5,
#         linestyle="-",
#         # label=r"$h(X_{T} \mid O_1, O_2')$",
#         label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2')}$",
#     )
#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_4,
#         marker="x",
#         color=colors[1],
#         linestyle="-",
#         alpha=0.5,
#         # label=r"$h(X_{T} \mid O_1, O_2)$",
#         label=r"${h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2)}$",
#     )

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_3,
#         marker="o",
#         color=colors[2],
#         alpha=0.5,
#         linestyle="-",
#         # label=r"$h(X_{T} \mid O_1, O_2')$",
#         label=r"${h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2')}$",
#     )

#     ax1.legend(
#         loc="upper left",
#         bbox_to_anchor=(-0.3, -0.17),
#         ncol=2
#         # borderpad=0.5,
#         # handlelength=1.0,
#     )
#     plt.savefig( output_path + "one_vs_two_exp_absolute_loss" + ".pdf", bbox_inches="tight", )
#     plt.savefig( output_path + "one_vs_two_exp_absolute_loss" + ".png", bbox_inches="tight", transparent=True)

#     fig, ax1 = plt.subplots()

#     # plt.title(
#     #     r"$O_1= \vec{X}_T + X_S + X_{S_1}$" + "\n"
#     #     r"$O_2= \vec{X}_T + X_S + X_{S_2}$, $O_2' = X_S + X_{S_2}$"
#     #     + "\n"
#     #     + r"$\mathcal{N}(0, %s), $" % sigma
#     #     + r" Participating Once v. Twice"
#     # )
#     # shared_spec_one = joint_data_one_exp.get_all_data(sigma).single_data["num_shared"]
#     # unique_1_one = joint_data_one_exp.get_all_data(sigma).single_data["num_spec_1"]
#     # h_t_one = joint_data_one_exp.get_all_data(sigma).single_data["target_init"]
#     # awae_one = joint_data_one_exp.get_all_data(sigma).single_data["awae"]

#     # h_t_mid_o1_o2_one = joint_data_one_exp.get_all_data(sigma).single_data[
#     #     "diff_ent_cond"
#     # ]

#     # h_t_mid_o1_o2_two = joint_data_two_exp.get_all_data(sigma).single_data[
#     #     "diff_ent_cond"
#     # ]
#     # abs_loss_zero = [(1-((a - b) / a) ) * 100.0 for a, b in zip(h_t_one, awae_one)]
#     abs_loss_one = [
#         (1 - ((a - b - (b - c)) / (a - b))) * 100.0
#         for a, b, c in zip(h_t_one, awae_one, h_t_mid_o1_o2_one)
#     ]
#     abs_loss_two = [
#         (1 - ((a - b - (b - c)) / (a - b))) * 100.0
#         for a, b, c in zip(h_t_one, awae_one, h_t_mid_o1_o2_two)
#     ]

#     # abs_loss_3 = [((a - b) / a) * 100.0 for a, b in zip(awae_one, h_t_mid_o1_o2_one)]
#     # abs_loss_4= [((a - b) / a) * 100.0 for a, b in zip(awae_one, h_t_mid_o1_o2_two)]
#     # print("abs_loss_zero", abs_loss_zero)
#     # print("abs_loss_one", abs_loss_one)
#     # print("abs_loss_two", abs_loss_two)
#     # print("abs_loss_3", abs_loss_3)
#     # print("abs_loss_4", abs_loss_4)
#     # print("i","abs_loss_zero", "abs_loss_one", "abs_loss_two", "abs_loss_3", "abs_loss_4")
#     # for i in range(len(abs_loss_one)):
#     #     print(
#     #         i,
#     #         h_t_one[i],
#     #         awae_one[i],
#     #         h_t_mid_o1_o2_one[i],
#     #         h_t_mid_o1_o2_two[i],
#     #         abs_loss_one[i],
#     #         abs_loss_two[i],
#     #     )

#     x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

#     plt.ylabel(r"Percent change (\%)")
#     # plt.xlabel(r"Fraction of shared spectators ($n = %s$)" % total_num_spec)
#     plt.xlabel(r"Fraction of shared spectators")

#     # (l2,) = plt.plot(
#     #     x_vals,
#     #     abs_loss_zero,
#     #     marker="x",
#     #     color=colors[4],
#     #     linestyle="-",
#     #     alpha=0.5,
#     #     # label=r"$h(X_{T} \mid O_1, O_2)$",
#     #     label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}{h(\vec{X}_T)}$",

#     # )

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_two,
#         marker="x",
#         color=colors[0],
#         linestyle="-",
#         alpha=0.5,
#         # label=r"$h(X_{T} \mid O_1, O_2)$",
#         label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2)}{h(\vec{X}_T)}$",
#     )

#     (l2,) = plt.plot(
#         x_vals,
#         abs_loss_one,
#         marker="o",
#         color=colors[3],
#         alpha=0.5,
#         linestyle="-",
#         # label=r"$h(X_{T} \mid O_1, O_2')$",
#         label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2')}{h(\vec{X}_T)}$",
#     )
#     # (l2,) = plt.plot(
#     #     x_vals,
#     #     abs_loss_3,
#     #     marker="x",
#     #     color=colors[1],
#     #     linestyle="-",
#     #     alpha=0.5,
#     #     # label=r"$h(X_{T} \mid O_1, O_2)$",
#     #     label=r"$\frac{h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2)}{h(\vec{X}_T \mid O_1)}$",

#     # )

#     # (l2,) = plt.plot(
#     #     x_vals,
#     #     abs_loss_4,
#     #     marker="o",
#     #     color=colors[2],
#     #     alpha=0.5,
#     #     linestyle="-",
#     #     # label=r"$h(X_{T} \mid O_1, O_2')$",
#     #     label=r"$\frac{h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2')}{h(\vec{X}_T \mid O_1)}$",

#     # )

#     # ax1.legend(
#     #     loc="upper left",
#     #     bbox_to_anchor=(0.0, -0.17),
#     #     ncol=2
#     #     # borderpad=0.5,
#     #     # handlelength=1.0,
#     # )
#     plt.savefig( output_path + "one_vs_two_exp_relative_loss_v2_" + str(total_num_spec) + ".pdf", bbox_inches="tight", )
#     plt.savefig( output_path + "one_vs_two_exp_relative_loss_v2_" + str(total_num_spec) + ".png", bbox_inches="tight", transparent=True)

#     # label_params = ax1.get_legend_handles_labels()

#     # plt.autoscale(enable=True, axis='y', tight=True)
#     # figl, axl = plt.subplots(figsize=(0,0))
#     # # axl.margins(x=0)
#     # # axl.margins(y=0)
#     # axl.axis(False)
#     # axl.legend(*label_params, ncol=3)
#     # figl.savefig(output_path + "legend_relative"
#     #     + ".pdf",
#     #     bbox_inches="tight",
#     #     pad_inches = 0
#     #     )

#     plt.close("all")
#     return 0


# def plot_one_vs_two_exp_percent(total_num_spec, sigma, output_path):
#     cc = itertools.cycle(colors)

#     fig, ax1 = plt.subplots()

#     plt.ylabel(r"Percent change (\%)")
#     # plt.xlabel(r"Fraction of shared spectators ($n = %s$)" % total_num_spec)
#     plt.xlabel(r"Fraction of shared spectators")

#     output_dir = "../output/"
#     main_exp = "joint_sum_normal"
#     plot_lines = []

#     for sp in total_num_spec:
#         c = next(cc)

#         joint_data_one_exp = joint_computation_continuous(
#             output_dir,
#             main_exp + "_two_targets",
#             "one_target_one_exp_vary_spec_percent",
#             joint_col_names,
#             num_spec=sp,
#         )
#         joint_data_two_exp = joint_computation_continuous(
#             output_dir,
#             main_exp + "_two_targets",
#             "one_target_two_exp_vary_spec_percent",
#             joint_col_names,
#             num_spec=sp,
#         )

#         shared_spec_one = joint_data_one_exp.get_all_data(sigma).single_data[
#             "num_shared"
#         ]
#         unique_1_one = joint_data_one_exp.get_all_data(sigma).single_data["num_spec_1"]
#         h_t_one = joint_data_one_exp.get_all_data(sigma).single_data["target_init"]
#         awae_one = joint_data_one_exp.get_all_data(sigma).single_data["awae"]

#         h_t_mid_o1_o2_one = joint_data_one_exp.get_all_data(sigma).single_data[
#             "diff_ent_cond"
#         ]

#         h_t_mid_o1_o2_two = joint_data_two_exp.get_all_data(sigma).single_data[
#             "diff_ent_cond"
#         ]

#         x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

#         # abs_loss_zero = [(1-((a - b) / a) ) * 100.0 for a, b in zip(h_t_one, awae_one)]
#         abs_loss_one = [
#             (1 - ((a - b - (b - c)) / (a - b))) * 100.0
#             for a, b, c in zip(h_t_one, awae_one, h_t_mid_o1_o2_one)
#         ]
#         abs_loss_two = [
#             (1 - ((a - b - (b - c)) / (a - b))) * 100.0
#             for a, b, c in zip(h_t_one, awae_one, h_t_mid_o1_o2_two)
#         ]

#         for i in range(len(abs_loss_one)):
#             print(
#                 i,
#                 x_vals[i],
#                 h_t_one[i],
#                 awae_one[i],
#                 h_t_mid_o1_o2_one[i],
#                 h_t_mid_o1_o2_two[i],
#                 abs_loss_one[i],
#                 abs_loss_two[i],
#             )
#         print()
#         x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

#         (l1,) = plt.plot(
#             x_vals,
#             abs_loss_two,
#             marker="o",
#             color=c,
#             alpha=0.5,
#             linestyle="--",
#             # label=r"$h(X_{T} \mid O_1, O_2')$",
#             label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2)}{h(\vec{X}_T)}$",
#         )
#         (l2,) = plt.plot(
#             x_vals,
#             abs_loss_one,
#             marker="x",
#             color=c,
#             linestyle="-",
#             alpha=0.5,
#             # label=r"$h(X_{T} \mid O_1, O_2)$",
#             label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2')}{h(\vec{X}_T)}$",
#         )

#         plot_lines.append([l1, l2])

#     legend_elements = [
#         Line2D(
#             [0],
#             [0],
#             color="black",
#             marker="o",
#             alpha=0.5,
#             linestyle="--",
#             # label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)- (h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2))}{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}$",
#             # label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)- (h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2))}{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}$",
#             label=r"$\frac{\text{Abs. loss after $O_1$} - (\text{Abs. loss between $O_1$ and $O_2$})}{\text{Abs. loss after $O_1$} }$",
#         ),
#         Line2D(
#             [0],
#             [0],
#             color="black",
#             marker="x",
#             alpha=0.5,
#             linestyle="-",
#             # label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1) - (h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2'))}{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}$",
#             label=r"$\frac{\text{Abs. loss after $O_1$} - (\text{Abs. loss between $O_1$ and $O_2'$})}{\text{Abs. loss after $O_1$} }$",
#         ),
#     ]
#     legend_elements2 = [
#         Line2D(
#             [0],
#             [0],
#             color=c,
#             marker="",
#             alpha=0.5,
#             linestyle="-",
#             label=r"$n = %s$" % s,
#         )
#         for s, c in zip(total_num_spec, colors[: len(total_num_spec)])
#     ]

#     legend1 = plt.legend(
#         handles=legend_elements,
#         loc="upper center",
#         # bbox_to_anchor=(.04, 1),
#         bbox_to_anchor=(0.5, -0.17),
#     )
#     legend2 = plt.legend(
#         handles=legend_elements2,
#         loc="upper center",
#         # bbox_to_anchor=(.04, 1),
#         # bbox_to_anchor=(0.5, -0.17),
#     )
#     plt.gca().add_artist(legend1)
#     plt.gca().add_artist(legend2)

#     dists = [r"$n = %s$" % (n) for n in total_num_spec]

#     # plt.legend([l[0] for l in plot_lines], dists, loc="upper right")

#     # plt.xlim([0.3, 0.7])
#     plt.xlim([0.4, 0.6])
#     plt.ylim([0, 100])

#     plt.savefig( output_path + "one_vs_two_exp_relative_loss_v2_all" + ".pdf", bbox_inches="tight", )
#     plt.savefig( output_path + "one_vs_two_exp_relative_loss_v2_all" + ".png", bbox_inches="tight", transparent=True)

    plt.close("all")
    return 0


def ratio(total_num_spec, sigma, output_path):
    cc = itertools.cycle(colors)

    fig, ax1 = plt.subplots()

    plt.ylabel(r"Percent change (\%)")
    # plt.xlabel(r"Fraction of shared spectators ($n = %s$)" % total_num_spec)
    plt.xlabel(r"Fraction of shared spectators")

    output_dir = "../output/"
    main_exp = "joint_sum_normal"
    plot_lines = []

    for sp in total_num_spec:
        c = next(cc)

        joint_data_one_exp = joint_computation_continuous(
            output_dir,
            main_exp + "_two_targets",
            "one_target_one_exp_vary_spec_percent",
            joint_col_names,
            num_spec=sp,
        )
        joint_data_two_exp = joint_computation_continuous(
            output_dir,
            main_exp + "_two_targets",
            "one_target_two_exp_vary_spec_percent",
            joint_col_names,
            num_spec=sp,
        )

        shared_spec_one = joint_data_one_exp.get_all_data(sigma).single_data[
            "num_shared"
        ]
        unique_1_one = joint_data_one_exp.get_all_data(sigma).single_data["num_spec_1"]
        h_t_one = joint_data_one_exp.get_all_data(sigma).single_data["target_init"]
        awae_one = joint_data_one_exp.get_all_data(sigma).single_data["awae"]

        h_t_mid_o1_o2_one = joint_data_one_exp.get_all_data(sigma).single_data[
            "diff_ent_cond"
        ]

        h_t_mid_o1_o2_two = joint_data_two_exp.get_all_data(sigma).single_data[
            "diff_ent_cond"
        ]

        x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

        # abs_loss_zero = [(1-((a - b) / a) ) * 100.0 for a, b in zip(h_t_one, awae_one)]
        ratio_one = [
            (b - c) / (a - b)
            for a, b, c in zip(h_t_one, awae_one, h_t_mid_o1_o2_one)
        ]
        ratio_two = [
            (b - c) / (a - b)
            for a, b, c in zip(h_t_one, awae_one, h_t_mid_o1_o2_two)
        ]

        # for i in range(len(ratio_one)):
        #     print(
        #         i,
        #         h_t_one[i],
        #         awae_one[i],
        #         h_t_mid_o1_o2_one[i],
        #         h_t_mid_o1_o2_two[i],
        #         ratio_one[i],
        #         ratio_two[i],
        #     )

        x_vals = shared_spec_one / (shared_spec_one + unique_1_one)

        (l1,) = plt.plot(
            x_vals,
            ratio_two,
            marker="o",
            color=c,
            alpha=0.5,
            linestyle="--",
            # label=r"$h(X_{T} \mid O_1, O_2')$",
            label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2)}{h(\vec{X}_T)}$",
        )
        (l2,) = plt.plot(
            x_vals,
            ratio_one,
            marker="x",
            color=c,
            linestyle="-",
            alpha=0.5,
            # label=r"$h(X_{T} \mid O_1, O_2)$",
            label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1, O_2')}{h(\vec{X}_T)}$",
        )

        plot_lines.append([l1, l2])

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            alpha=0.5,
            linestyle="--",
            # label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)- (h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2))}{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}$",
            # label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)- (h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2))}{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}$",
            label=r"$\frac{(\text{Abs. loss between $O_1$ and $O_2$})}{\text{Abs. loss after $O_1$} }$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="x",
            alpha=0.5,
            linestyle="-",
            # label=r"$\frac{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1) - (h(\vec{X}_T \mid O_1) -  h(\vec{X}_T\mid O_1, O_2'))}{h(\vec{X}_T) -  h(\vec{X}_T\mid O_1)}$",
            label=r"$\frac{(\text{Abs. loss between $O_1$ and $O_2'$})}{\text{Abs. loss after $O_1$} }$",
        ),
    ]
    legend_elements2 = [
        Line2D(
            [0],
            [0],
            color=c,
            marker="",
            alpha=0.5,
            linestyle="-",
            label=r"$n = %s$" % s,
        )
        for s, c in zip(total_num_spec, colors[: len(total_num_spec)])
    ]

    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper center",
        # bbox_to_anchor=(.04, 1),
        bbox_to_anchor=(0.5, -0.17),
    )
    legend2 = plt.legend(
        handles=legend_elements2,
        loc="upper center",
        # bbox_to_anchor=(.04, 1),
        # bbox_to_anchor=(0.5, -0.17),
    )
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    dists = [r"$n = %s$" % (n) for n in total_num_spec]

    # plt.legend([l[0] for l in plot_lines], dists, loc="upper right")

    # plt.xlim([0.4, 0.6])
    # plt.ylim([0, 65])

    plt.savefig( output_path + "ratio_loss" + ".pdf", bbox_inches="tight", )
    plt.savefig( output_path + "ratio_loss" + ".png", bbox_inches="tight", transparent=True)

    plt.close("all")
    return 0


def plot_vary_loss(
    joint_data, sigma_vals, sigma, spec_intervals, upper_bound, output_path
):
    fig, ax1 = plt.subplots()

    # plt.title(
    #     r"$O_1 = \vec{X}_T + X_S + X_{S_1}$, $O_2 = \vec{X}_T + X_S + X_{S_2}$"
    #     + "\n"
    #     + r"$\mathcal{N}(0, %s), $ " % sigma
    #     + r"Absolute Entropy Loss"
    # )
    plt.ylabel(r"Entropy (bits)")
    # plt.xlabel(r"No. spectators per experiment ($|S_{12} \cup S_{1}|=|S_{12} \cup S_2|$)")
    plt.xlabel(r"No. spectators per experiment")

    shared_spec = joint_data.get_all_data(sigma).single_data["num_shared"]
    unique_1 = joint_data.get_all_data(sigma).single_data["num_spec_1"]
    h_t_mid_o1_o2 = joint_data.get_all_data(sigma).single_data["diff_ent_cond"]
    h_t = joint_data.get_all_data(sigma).single_data["target_init"]
    awae = joint_data.get_all_data(sigma).single_data["awae"]

    # max_shared = int(shared_spec[-1] - shared_spec[0] + 1)
    # max_unique = int(unique_1[-1] - unique_1[0] + 1)
    # plot_lines = []
    # c = next(cc)

    # # plot_lines.append([l0])

    # (l0,) = ax1.plot(
    #     unique_1[0:(upper_bound)],
    #     h_t[:(upper_bound)],
    #     marker="",
    #     color="black",
    #     alpha=alph,
    #     linestyle="-",
    #     label=r"$h(\vec{X}_T)$",
    # )

    alph = 0.5
    cc = itertools.cycle(colors)

    loss_zero_to_one = [((a - b) / a) * 100.0 for a, b in zip(h_t, awae)]
    loss_zero_to_two = [((a - b) / a) * 100.0 for a, b in zip(h_t, h_t_mid_o1_o2)]

    loss_one_to_two = [((a - b) / a) * 100.0 for a, b in zip(awae, h_t_mid_o1_o2)]

    max_shared = int(shared_spec[-1] - shared_spec[0] + 1)
    max_unique = int(unique_1[-1] - unique_1[0] + 1)
    plot_lines = []
    c = next(cc)

    # plot_lines.append([l0])

    for spec in spec_intervals:
        x_data = [
            x + spec for x in unique_1[max_unique * spec : max_unique * (spec + 1)]
        ]
        h_t_mid_o1_o2_slice = [
            x for x in h_t_mid_o1_o2[max_unique * spec : max_unique * (spec + 1)]
        ]

        awae_slice = awae[max_unique * spec : max_unique * (spec + 1)]
        h_t_slice = h_t[max_unique * spec : max_unique * (spec + 1)]

        (l1,) = ax1.plot(
            x_data[: (upper_bound - spec)],
            h_t_slice[: (upper_bound - spec)] - awae_slice[: (upper_bound - spec)],
            marker="+",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$h(\vec{X}_T) - h(\vec{X}_T \mid O_1)$",
        )

        (l2,) = ax1.plot(
            x_data[: (upper_bound - spec)],
            h_t_slice[: (upper_bound - spec)]
            - h_t_mid_o1_o2_slice[: (upper_bound - spec)],
            marker="^",
            color=c,
            alpha=alph,
            linestyle="--",
            label=r"$h(\vec{X}_T) - h(\vec{X}_T \mid O_1, O_2)$",
        )
        (l3,) = ax1.plot(
            x_data[: (upper_bound - spec)],
            awae_slice[: (upper_bound - spec)]
            - h_t_mid_o1_o2_slice[: (upper_bound - spec)],
            marker="s",
            color=c,
            alpha=alph,
            linestyle="dotted",
            label=r"$h(\vec{X}_T \mid O_1) - h(\vec{X}_T \mid O_1, O_2)$",
        )
        plot_lines.append([l1, l2, l3])
        c = next(cc)

    dists = [r"$\lvert {S}_{12} \rvert = %s$" % n for n in spec_intervals]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="+",
            alpha=alph,
            linestyle="-",
            label=r"$h(\vec{X}_T) - h(\vec{X}_T \mid O_1)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="^",
            alpha=alph,
            linestyle="--",
            label=r"$h(\vec{X}_T) - h(\vec{X}_T \mid O_1, O_2)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="s",
            alpha=alph,
            linestyle="-",
            label=r"$h(\vec{X}_T \mid O_1) - h(\vec{X}_T \mid O_1, O_2)$",
        ),
    ]
    legend1 = plt.legend(
        handles=legend_elements,
        # prop={"size": 14},
        loc="upper right",
    )

    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [
    #         r"$h(\vec{X}_T) - h(\vec{X}_T \mid O_1)$",
    #         r"$h(\vec{X}_T) - h(\vec{X}_T \mid O_1, O_2)$",
    #         r"$h(\vec{X}_T \mid O_1) - h(\vec{X}_T \mid O_1, O_2)$",
    #     ],
    #     loc="upper right",
    # )
    # legend1 = plt.legend(
    #     handles=legend_elements,
    #     prop={"size": 14},
    #     loc="center right",
    # )

    plt.legend(
        [l[0] for l in plot_lines],
        dists,
        #    prop={"size": 14},
        loc="center right",
    )

    plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend_elements)
    plt.savefig( output_path + "loss_" + joint_data.exp_name + "_" + joint_data.sub_exp + ".pdf", bbox_inches="tight", )
    plt.savefig( output_path + "loss_" + joint_data.exp_name + "_" + joint_data.sub_exp + ".png", bbox_inches="tight", transparent=True)

    plt.close("all")

    return 0
