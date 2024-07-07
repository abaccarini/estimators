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
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb,sansmath}",
    }
)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 18})
rc("text", usetex=True)


output_dir = "../output/"

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


def plot_all(
    data,
    normal_mixed_alt,
    normal_data,
    normal_fixed_full_data,
    N,
    sigma_vals,
    upper_bound,
    output_path,
):
    fig, ax1, = plt.subplots()
    fig.set_size_inches(6, 5)

    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(
        r"Total No. spectators $(\lvert S_\mathsf{B}\rvert + \lvert S_\mathsf{C}\rvert + \lvert S_\mathsf{D}\rvert)$"
    )
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "x", "s"])
    lsty = itertools.cycle(["-", "--", "-."])
    spec = [None]
    # print(sigma_vals)
    plot_lines = []

    eps = -5
    integ_type = "gsl"
    exp_name = "sh-v-diff"
    c = next(cc)
    alph = next(alphas)
    # print(exp_name, integ_type, eps, 8, 8)
    spec = normal_data.get_all_data(exp_name, integ_type, eps, 8, 8).single_data[
        "num_spec"
    ]
    spec = [3 * a - 1 for a in spec]
    spec1 = normal_mixed_alt.get_all_data(100, 4).single_data["num_spec_1"]
    spec2 = normal_mixed_alt.get_all_data(100, 4).single_data["num_spec_2"]
    spec3 = normal_mixed_alt.get_all_data(100, 4).single_data["num_spec_3"]
    spec_alt = [a + b + c for (a, b, c) in zip(spec1, spec2, spec3)]
    abs_loss = [
        (a - b)
        for a, b in zip(
            normal_data.get_all_data(exp_name, integ_type, eps, 8, 4).single_data[
                "h_T"
            ],
            normal_data.get_all_data(exp_name, integ_type, eps, 8, 4).single_data[
                "awae_differential"
            ],
        )
    ]
    # print(abs_loss)
    orig_abs_loss = abs_loss[1::3]
    print(spec)
    print(spec[0:upper_bound])
    print(len(spec[0:upper_bound]))
    print(orig_abs_loss)
    print(orig_abs_loss[0:upper_bound])
    print(len(orig_abs_loss[0:upper_bound]))
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        orig_abs_loss[0:upper_bound],
        marker="+",
        color="red",
        markersize=11,
        alpha=alph,
        linestyle="--",
        label=r"Normal same",
    )
    plot_lines.append([l1])

    
    years_dict = dict()
    for s in sigma_vals[0]:
        alph = next(alphas)
        m = next(markers)
        l = next(lsty)
        spec1_bound = normal_fixed_full_data.get_all_data(50, s).single_data[
            "num_spec_1"
        ]
        spec2_bound = normal_fixed_full_data.get_all_data(50, s).single_data[
            "num_spec_2"
        ]
        spec3_bound = normal_fixed_full_data.get_all_data(50, s).single_data[
            "num_spec_3"
        ]
        spec_alt_bound = [a + b + c for (a, b, c) in zip(spec1_bound, spec2_bound, spec3_bound)]
        abs_loss_bound = [
            (a - b)
            for a, b in zip(
                normal_fixed_full_data.get_all_data(50, s).single_data["h_T"],
                normal_fixed_full_data.get_all_data(50, s).single_data[
                    "awae_differential"
                ],
            )
        ]
        zipped = zip(spec_alt_bound, abs_loss_bound)
        for a, b in zip(spec_alt_bound, abs_loss_bound):
            if (int(a) < 33):
                if str(int(a)) in years_dict:
                    years_dict[str(int(a))].append(b)
                else:
                    # create a new array in this slot
                    years_dict[str(int(a))] = [b]
                # results.append((int(a),b))
    
    filtered_specs = []
    mins = [] 
    maxes = []
    del years_dict['0']
    del years_dict['1']

    for r in years_dict:
        filtered_specs.append(int(r))
        mins.append(min(years_dict[r]))
        maxes.append(max(years_dict[r]))
        print(r, min(years_dict[r]),max(years_dict[r]),)
        # zipped = sorted(zipped, key=lambda x: x[1])
    ax1.fill_between(filtered_specs, mins, maxes, facecolor='gray',alpha=.5)

   
    for sigma in sigma_vals:
        c = next(cc)
        alph = next(alphas)
        for s in sigma:
            m = next(markers)
            l = next(lsty)

            abs_loss = [
                (a - b)
                for a, b in zip(
                    normal_mixed_alt.get_all_data(N, s).single_data["h_T"],
                    normal_mixed_alt.get_all_data(N, s).single_data[
                        "awae_differential"
                    ],
                )
            ]
            (l1,) = ax1.plot(
                spec_alt[0:upper_bound],
                abs_loss[0:upper_bound],
                marker=m,
                color=c,
                alpha=alph,
                linestyle=l,
                # label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
            )
            plot_lines.append([l1])

    dists = []
    col = "black"
    companies = [r"\textsf{B}", r"\textsf{C}", r"\textsf{D}"]
    params = [
        r"\sigma_\text{B}^2",
        r"1.1^2\sigma_\text{B}^2",
        r"0.9^2\sigma_\text{B}^2",
    ]
    # dists = [r"Same $\sigma^2$"] + [r"$\sigma^2_T =  %s$" % (float(n)) for n in sigma_vals]
    dists = [r"$\textsf{B},\textsf{C},\textsf{D} \sim \mathcal{N}(0, \sigma_\text{B}^2$)"] + [
        r"$\mathsf{id}_T = %s$" % (n) for (n, s) in zip(companies, params)
    ]
    # + [r"$\mathsf{id}_T = %s \sim \mathcal{N}(0, %s)$" % (n,s) for (n,s)  in zip(companies,params)]

    legend_elements = [
        # Line2D(
        #     [0],
        #     [0],
        #     color="black",
        #     marker="",
        #     alpha=0.5,
        #     linestyle="-",
        #     label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        # ),
        Line2D(
            [0],
            [0],
            color="blue",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 4$",
            # label=r"Spectator groups fixed",
            # label=r"$\lvert S_\textsf{B}\rvert = \lvert S_\textsf{C}\rvert = \lvert S_\textsf{D}\rvert$",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 8$",
            # label=r"Target moves between groups" +"\n" + r"e.g. $\lvert S_\textsf{B} \cup T\rvert = \lvert S_\textsf{C}\rvert = \lvert S_\textsf{D}\rvert$",
        ),
        Line2D(
            [0],
            [0],
            color="magenta",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 16$",
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     color="darkorange",
        #     marker="",
        #     alpha=alph,
        #     linestyle="-",
        #     label=r"$\sigma_\text{B}^2 = 32$",
        # ),
    ]
    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"$h(\vec{X}_T) - h(\vec{X}_T \mid X_T + X_S)$"],
    #     loc="upper right",
    # )
    legend1 = plt.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.0, 0.3),
        prop={"size": 16},
    )

    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 3.0))

    # textstr = r'$\sigma^2_{S_1} = %s$'%sigma_vals[0] + '\n' + r'$\sigma^2_{S_2} = %s$'%sigma_vals[1] + '\n' r'$\sigma^2_{S_3} =  %s$'%sigma_vals[2]
    # props = dict(boxstyle='round', facecolor='none', alpha=0.1)
    # ax1.text(1.05, 0.95, textstr, transform=ax1.transAxes, verticalalignment='top', bbox=props)
    plt.gca().add_artist(legend1)

    print(output_path + data.exp_name + "_absolute_loss.pdf")
    plt.savefig(
        output_path + data.exp_name + "_absolute_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + data.exp_name + "_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax1 = plt.subplots()
    # plt.figure(figsize=(4,4))
    fig.set_size_inches(6, 4.7)
    plt.ylabel(r"Percent change")

    plt.xlabel(
        r"Total No. spectators $(\lvert S_\mathsf{B}\rvert + \lvert S_\mathsf{C}\rvert + \lvert S_\mathsf{D}\rvert)$"
    )
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "x", "s"])
    spec = [None]
    # print(sigma_vals)
    plot_lines = []

    eps = -5
    integ_type = "gsl"
    exp_name = "sh-v-diff"
    c = next(cc)
    alph = next(alphas)
    # print(exp_name, integ_type, eps, 8, 8)
    spec = normal_data.get_all_data(exp_name, integ_type, eps, 8, 8).single_data[
        "num_spec"
    ]
    spec = [3 * a - 1 for a in spec]
    # print(spec)
    abs_loss = [
        ((a - b) / a) * 100.0
        for a, b in zip(
            normal_data.get_all_data(exp_name, integ_type, eps, 8, 4).single_data[
                "h_T"
            ],
            normal_data.get_all_data(exp_name, integ_type, eps, 8, 4).single_data[
                "awae_differential"
            ],
        )
    ]
    # print(abs_loss)
    abs_loss = abs_loss[1::3]
    # print(abs_loss)
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        abs_loss[0:upper_bound],
        marker="+",
        markersize=11,
        color=c,
        alpha=alph,
        linestyle="--",
        label=r"Normal same",
    )
    plot_lines.append([l1])


 
    years_dict = dict()
    for s in sigma_vals[0]:
        alph = next(alphas)
        m = next(markers)
        l = next(lsty)
        spec1_bound = normal_fixed_full_data.get_all_data(50, s).single_data[
            "num_spec_1"
        ]
        spec2_bound = normal_fixed_full_data.get_all_data(50, s).single_data[
            "num_spec_2"
        ]
        spec3_bound = normal_fixed_full_data.get_all_data(50, s).single_data[
            "num_spec_3"
        ]
        spec_alt_bound = [a + b + c for (a, b, c) in zip(spec1_bound, spec2_bound, spec3_bound)]
        abs_loss_bound = [
            ((a - b) / a) * 100.0
            for a, b in zip(
                normal_fixed_full_data.get_all_data(50, s).single_data["h_T"],
                normal_fixed_full_data.get_all_data(50, s).single_data[
                    "awae_differential"
                ],
            )
        ]
        zipped = zip(spec_alt_bound, abs_loss_bound)
        for a, b in zip(spec_alt_bound, abs_loss_bound):
            if (int(a) < 33):
                if str(int(a)) in years_dict:
                    years_dict[str(int(a))].append(b)
                else:
                    # create a new array in this slot
                    years_dict[str(int(a))] = [b]
                # results.append((int(a),b))
    
    filtered_specs = []
    mins = [] 
    maxes = []
    del years_dict['0']
    del years_dict['1']

    for r in years_dict:
        filtered_specs.append(int(r))
        mins.append(min(years_dict[r]))
        maxes.append(max(years_dict[r]))
        print(r, min(years_dict[r]),max(years_dict[r]),)
        # zipped = sorted(zipped, key=lambda x: x[1])
    ax1.fill_between(filtered_specs, mins, maxes, facecolor='gray',alpha=.5)




    c = next(cc)
    for sigma in sigma_vals[0]:
        m = next(markers)
        alph = next(alphas)
        abs_loss = [
            ((a - b) / a) * 100.0
            for a, b in zip(
                data.get_all_data(N, sigma).single_data["h_T"],
                data.get_all_data(N, sigma).single_data["awae_differential"],
            )
        ]
        # for i,a in zip(spec[0:upper_bound], abs_loss[0:upper_bound]):
        # print(sigma, i, a)
        (l1,) = ax1.plot(
            spec_alt[0:upper_bound],
            abs_loss[0:upper_bound],
            marker=m,
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])
        values = {1.0: "s", 5.0: "s"}
        for value in values:
            for i, a in enumerate(abs_loss):
                if a < value:
                    res = (3 * i, a)
                    break

            # res = min(enumerate(abs_loss), key=lambda x: abs(value - x[1]))

            # print(res)
            if sigma == sigma_vals[0][1]:
                plt.vlines(
                    res[0] + 2, 0.8, res[1], alpha=0.5, color="black", linestyle="--"
                )
                plt.plot(
                    res[0] + 2, res[1], alpha=0.5, marker=values[value], color="black"
                )
            if sigma == sigma_vals[0][1]:
                plt.hlines(
                    res[1],
                    xmin=2,
                    xmax=res[0] + 2,
                    alpha=0.5,
                    color="black",
                    linestyle="--",
                )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="blue",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 4$",
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     color="blue",
        #     marker="",
        #     alpha=alph,
        #     linestyle="-",
        #     label=r"$\sigma_\text{B}^2 = 4$",
        # ),
    ]

    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"$h(\vec{X}_T) - h(\vec{X}_T \mid X_T + X_S)$"],
    #     loc="upper right",
    # )

    legend1 = plt.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.0, 0.3),
        prop={"size": 16},
    )
    dists = []
    col = "black"
    companies = [r"\textsf{B}", r"\textsf{C}", r"\textsf{D}"]
    params = [
        r"\sigma_\text{B}^2",
        r"1.1^2\sigma_\text{B}^2",
        r"0.9^2\sigma_\text{B}^2",
    ]
    # dists = [r"Same $\sigma^2$"] + [r"$\sigma^2_T =  %s$" % (float(n)) for n in sigma_vals]
    dists = [r"$\textsf{B},\textsf{C},\textsf{D}  \sim \mathcal{N}(0, \sigma_\text{B}^2)$"] + [
        r"$\mathsf{id}_T = %s$" % (n) for (n, s) in zip(companies, params)
    ]

    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")

    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 3.0))

    plt.gca().add_artist(legend1)
    print(output_path + data.exp_name + "_relative_loss.pdf")
    plt.savefig(
        output_path + data.exp_name + "_relative_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + data.exp_name + "_relative_loss.png",
        bbox_inches="tight",
        transparent=True,
    )

    return 0


def plot_case2(case2data, normal_data, N, sigma_vals, upper_bound, output_path):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(6, 5)

    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"Total No. spectators")
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "x", "s"])
    lsty = itertools.cycle(["-", "--", "-."])
    spec = [None]
    # print(sigma_vals)
    plot_lines = []

    eps = -5
    integ_type = "gsl"
    exp_name = "sh-v-diff"
    c = next(cc)
    alph = next(alphas)
    spec = normal_data.get_all_data(exp_name, integ_type, eps, 8, 8).single_data[
        "num_spec"
    ]
    # spec = [3 * a -1 for a in spec]
    abs_loss = [
        (a - b)
        for a, b in zip(
            normal_data.get_all_data(exp_name, integ_type, eps, 8, 4).single_data[
                "h_T"
            ],
            normal_data.get_all_data(exp_name, integ_type, eps, 8, 4).single_data[
                "awae_differential"
            ],
        )
    ]
    # orig_abs_loss = abs_loss[1::3]
    (l1,) = ax1.plot(
        spec[0:upper_bound],
        abs_loss[0:upper_bound],
        marker="+",
        markersize=11,
        color="red",
        alpha=alph,
        linestyle="--",
        label=r"Normal same",
    )
    plot_lines.append([l1])

    num_spec = (case2data.get_all_data(50, sigma_vals[0]).single_data["num_spec"],)
    print(num_spec)
    print(num_spec[0][0:upper_bound])

    for sigma in sigma_vals:
        c = next(cc)
        alph = next(alphas)
        m = next(markers)
        l = next(lsty)

        abs_loss = (case2data.get_all_data(50, sigma).single_data["abs_loss"],)
        print(abs_loss)
        print(abs_loss[0:upper_bound])
        (l1,) = ax1.plot(
            num_spec[0][0:upper_bound],
            abs_loss[0][0:upper_bound],
            marker="o",
            color=c,
            alpha=0.3,
            linestyle="-",
            # label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
        )
        plot_lines.append([l1])

    # plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 2.0))

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="blue",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 4$",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 8$",
        ),
        Line2D(
            [0],
            [0],
            color="magenta",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 16$",
        ),
    ]

    legend_elements2 = [
        Line2D(
            [0],
            [0],
            color="red",
            marker="+",
            markersize=11,
            alpha=alph,
            linestyle="--",
            label=r"$\textsf{B},\textsf{C},\textsf{D} \sim \mathcal{N}(0, \sigma_\text{B}^2$)",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            marker="o",
            alpha=alph,
            linestyle="-",
            label=r"$\textsf{B}\sim \mathcal{N}(0,\sigma_\text{B}^2)$"
            + "\n"
            + r"$\textsf{C}\sim \mathcal{N}(0,1.1^2\sigma_\text{B}^2)$"
            + "\n"
            + r"$\textsf{D}\sim \mathcal{N}(0,0.9^2\sigma_\text{B}^2)$",
        ),
    ]

    legend1 = plt.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.0, 0.3),
        prop={"size": 16},
    )

    legend2 = plt.legend(
        handles=legend_elements2,
        loc="upper right",
        # bbox_to_anchor=(1.0, 0.3),
        prop={"size": 16},
    )
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    print(output_path + case2data.exp_name + "_absolute_loss.pdf")
    plt.savefig(
        output_path + case2data.exp_name + "_absolute_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + case2data.exp_name + "_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )


def plot_full(data, N, sigma_vals, upper_bound, output_path):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(6, 5)

    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(
        r"Total No. spectators $(\lvert S_\mathsf{B}\rvert + \lvert S_\mathsf{C}\rvert + \lvert S_\mathsf{D}\rvert)$"
    )
    cc = itertools.cycle(colors)
    alphas = itertools.cycle([0.5])
    markers = itertools.cycle(["o", "x", "s"])
    lsty = itertools.cycle(["-", "--", "-."])
    spec = [None]
    plot_lines = []

    c = next(cc)
    alph = next(alphas)

    # for sigma in sigma_vals:

    for sigma in sigma_vals:
        c = next(cc)
        alph = next(alphas)
        for s in sigma:
            alph = next(alphas)
            m = next(markers)
            l = next(lsty)
            spec1 = data.get_all_data(N, s).single_data["num_spec_1"]
            spec2 = data.get_all_data(N, s).single_data["num_spec_2"]
            spec3 = data.get_all_data(N, s).single_data["num_spec_3"]
            spec_alt = [a + b + c for (a, b, c) in zip(spec1, spec2, spec3)]
            # print(spec_alt)
            abs_loss = [
                (a - b)
                for a, b in zip(
                    data.get_all_data(N, s).single_data["h_T"],
                    data.get_all_data(N, s).single_data["awae_differential"],
                )
            ]

            (l1,) = ax1.plot(
                spec_alt,
                abs_loss,
                marker=m,
                color=c,
                alpha=alph,
                linestyle="",
                # label=r"${h(\vec{X}_T) -  h(\vec{X}_T\mid X_T + X_S)}$",
            )
            plot_lines.append([l1])

    dists = []
    col = "black"
    companies = [r"\textsf{B}", r"\textsf{C}", r"\textsf{D}"]
    params = [
        r"\sigma_\text{B}^2",
        r"1.1^2\sigma_\text{B}^2",
        r"0.9^2\sigma_\text{B}^2",
    ]
    dists = [r"$\mathsf{id}_T = %s$" % (n) for (n, s) in zip(companies, params)]

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="blue",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 4$",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 8$",
        ),
        Line2D(
            [0],
            [0],
            color="magenta",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\sigma_\text{B}^2 = 16$",
        ),
    ]
    legend1 = plt.legend(
        handles=legend_elements,
        loc="center right",
        bbox_to_anchor=(1.0, 0.3),
        prop={"size": 16},
    )
    plt.gca().set_xlim(right=upper_bound, left=3)
    plt.legend([l[0] for l in plot_lines], dists, loc="upper right")
    # plt.xticks(np.arange(min(spec[0:upper_bound]), max(spec[0:upper_bound]) + 1, 3.0))

    # plt.gca().add_artist(legend1)

    print(output_path + data.exp_name + "_absolute_loss.pdf")
    plt.savefig(
        output_path + data.exp_name + "_absolute_loss.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        output_path + data.exp_name + "_absolute_loss.png",
        bbox_inches="tight",
        transparent=True,
    )


def plot_main():
    dir_path = "../../proof/figs/single_mixed/"
    isExist = os.path.exists(dir_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")

    normal_mixed = single_mixed(output_dir, "normal_mixed", mixed_cols)
    normal_mixed_alt = single_mixed(output_dir, "normal_mixed_alt", mixed_cols)
    normal_mixed_case2 = single_mixed(output_dir, "normal_mixed_case2", case2cols)

    normal_mixed_full = single_mixed(output_dir, "normal_mixed_full", mixed_cols)

    normal_data = single_computation_continuous(
        output_dir, "normal_v2", continous_col_names
    )
    N = 100
    sigma_vals = [
        [4, 4.84, 3.24],
        [8.0, 9.68, 6.48],
        [16, 19.36, 12.96],
        # [32, 38.72, 25.92],
    ]

    # sigma_vals = [[4,4.84,3.24]]
    upper_bound = 11
    # dat = normal_mixed.get_all_data(
    #                 N, sigma
    #             ).single_data["h_T"],

    plot_all(
        normal_mixed_alt,
        normal_mixed_alt,
        normal_data,
        normal_mixed_full,
        N,
        sigma_vals,
        upper_bound,
        dir_path,
    )

    sigma_vals = [4, 8, 16]
    plot_case2(normal_mixed_case2, normal_data, N, sigma_vals, 33, dir_path)

    sigma_vals = [[4, 4.84, 3.24]]

    plot_full(
        normal_mixed_full,
        50,
        sigma_vals,
        32,
        dir_path,
    )

    # plot_all(
    #     normal_mixed_alt,
    #     normal_mixed_alt,
    #     normal_data,
    #     N,
    #     sigma_vals,
    #     upper_bound,
    #     dir_path,
    # )

    # print(dat)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_main()
