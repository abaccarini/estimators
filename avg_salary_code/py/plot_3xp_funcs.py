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
    "green",
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

colors_iter = [
    "darkorange",
    "mediumspringgreen",
    "magenta",
    "black",
    "saddlebrown",
    "mediumslateblue",
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


num_spec_cols = {
    "S_1": r"S_{1}",
    "S_2": r"S_{2}",
    "S_3": r"S_{3}",
    "S_12": r"S_{12}",
    "S_13": r"S_{13}",
    "S_23": r"S_{23}",
    "S_123": r"S_{123}",
}


def get_ranges(var_names, const_names, all_data):
    final_str = r"\begin{eqnarray*}"
    # final_str =r""
    for c in const_names:
        x_val = [
            int(i)
            for i in all_data.get_all_data(participation_configs[0]).single_data[
                c
            ]  # doesnt matter wich exp config we use, ('111', '110', '100') so we use '111'
        ]
        x_val.sort()
        final_str += r"|%s| =& %s\\" % (num_spec_cols[c], x_val[0])

    for v in var_names:
        x_val = [
            int(i)
            for i in all_data.get_all_data(participation_configs[0]).single_data[
                v
            ]  # doesnt matter wich exp config we use, ('111', '110', '100') so we use '111'
        ]
        x_val.sort()
        final_str += r"|%s| \in& \left[%s, %s \right]\\" % (
            num_spec_cols[v],
            x_val[0],
            x_val[-1],
        )
    final_str = final_str[:-2] + r"\end{eqnarray*}"
    return final_str


def plot_3xp(all_data, total_num_spec, sigma, output_path):
    fig, ax1 = plt.subplots()

    H_X_T = [
        all_data.get_all_data(part).single_data["H_X_T"]
        for part in participation_configs
    ]
    # print(H_X_T)
    H_X_T_O_1 = [
        all_data.get_all_data(part).single_data["H_X_T_O_1"]
        for part in participation_configs
    ]
    H_X_T_O_12 = {
        part: all_data.get_all_data(part).single_data["H_X_T_O_12"]
        for part in participation_configs
    }
    H_X_T_O_123 = {
        part: all_data.get_all_data(part).single_data["H_X_T_O_123"]
        for part in participation_configs
    }
    const_names = []
    vars = []
    var_names = []

    for ns in num_spec_cols:
        x_val = [
            int(i)
            for i in all_data.get_all_data(participation_configs[0]).single_data[ns]
        ]
        if len(set(x_val)) == 1:
            # print("constant")
            const_names.append(ns)
        else:
            # print("vars")
            vars.append(x_val)
            var_names.append(ns)

    # print("const_names", const_names)
    # print("vars", vars)
    # print("var_names", var_names)
    # if all_data.get_all_data(participation_configs[0]).single_data["num_shared"]
    # denom =[]
    # for v in vars:
    #     denom += v
    idx = 0
    idx_2 = 0
    for i, v in enumerate(vars):
        if v[0] == 0:
            idx = i
        if v[-1] == 0:
            idx_2 = i

    num_vars = len(vars)
    # print("num_vars", num_vars)
    denom = [(x + y ) for x,y in zip(vars[idx],vars[idx_2])]

    x_vals = [x / y for x, y in zip(vars[idx], denom)]
    # x_vals = [x for x, y in zip(vars[idx], denom)]
    # print("denom", denom, len(denom))
    # print("vars[idx]", vars[idx], len(vars[idx]))
    # print("x_vals", x_vals)
    vars_without_num = var_names[:idx] + var_names[idx+1:]
    
    
    # denom_str = (r"".join([r"|" + num_spec_cols[x] + r"|+" for x in var_names]))[:-1]
    denom_str = r"|" + (r"".join([num_spec_cols[x] + r", " for x in vars_without_num]))[:-2] + r"|+|" + num_spec_cols[var_names[idx]] + r"|"   

    plt.title(r"$(\text{Total per exec.}, \sigma^2) = (%s, %s)$"% (total_num_spec, sigma))
    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(
        r"$|%s| / \left( %s \right)$" % (num_spec_cols[var_names[idx]], denom_str)
    )

    var_str = get_ranges(var_names, const_names, all_data)
    ax1.text(
        1.11,
        1.1,
        r"%s" % (var_str),
        transform=ax1.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
    )

    (l2,) = plt.plot(
        x_vals,
        H_X_T[0],
        marker="",
        color=colors[1],
        alpha=0.8,
        linestyle="--",
        label=r"$h(\vec{X}_{T})$",
    )
    (l2,) = plt.plot(
        x_vals,
        H_X_T_O_1[0],
        marker="",
        color=colors[2],
        alpha=0.8,
        linestyle="-",
        label=r"$h(\vec{X}_{T} \mid O_1)$",
        # label=r"$h(X_{T} \mid O_1)$",
    )
    markers = itertools.cycle(["x", "o", "v"])
    linesty = itertools.cycle(["-", "--", "-."])
    color = itertools.cycle(colors_iter)

    (l2,) = plt.plot(
        x_vals,
        H_X_T_O_12["111"],
        marker="s",
        color=colors[0],
        linestyle="-",
        alpha=0.6,
        label=r"%s" % participation_mapping["111"][0]
        # label=r"$h(X_{T} \mid O_1, O_2)$",
    )
    (l2,) = plt.plot(
        x_vals,
        H_X_T_O_12["100"],
        marker="s",
        color='deepskyblue',
        linestyle="--",
        alpha=0.7,
        label=r"%s" % participation_mapping["100"][0]
        # label=r"$h(X_{T} \mid O_1, O_2)$",
    )
    for part in participation_configs:
        mark = next(markers)
        lsty = next(linesty)
        cc = next(color)

        # if (part != "111") and (part != "110"):
        #     # print(part)

        #     (l2,) = plt.plot(
        #         x_vals,
        #         H_X_T_O_12[part],
        #         marker="s",
        #         color=cc,
        #         linestyle=lsty,
        #         alpha=0.5,
        #         label=r"%s" % participation_mapping[part][0]
        #         # label=r"$h(X_{T} \mid O_1, O_2)$",
        #     )
        #     mark = next(markers)
        #     lsty = next(linesty)
        (l2,) = plt.plot(
            x_vals,
            H_X_T_O_123[part],
            marker="x",
            color=cc,
            linestyle=lsty,
            alpha=0.7,
            label=r"%s" % participation_mapping[part][1],
        )

    label_params = ax1.get_legend_handles_labels()
    ax1.legend(
        *label_params,
        loc="upper left",
        bbox_to_anchor=(
            1.1,
            0.5,
        ),
        ncol=1
    )

    plt.savefig(
        output_path + all_data.sub_exp + "_" + str(total_num_spec) + ".pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        output_path + all_data.sub_exp + "_" + str(total_num_spec) + ".png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.close("all")
    return 0
