import json
import numpy as np
import logging
import os
import re
import subprocess
import sys
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# logging.getLogger("matplotlib.font_manager").disabled = True

plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,
        # "font.family": "sans-serif",
        # "font.sans-serif": "Helvetica",
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb,sfmath}",
    }
)


colors = [
    "red",
    "blue",
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


func_names = ["max", "min", "median", "median_min", "var", "var_nd", "var_mean"]
distributions = ["uniform_int", "uniform_real", "normal", "lognormal"]
data_dir = "../output/"
fig_dir = "../figs/"


def verify_args(fname, dist):
    if dist not in distributions:
        print(
            "Unknown distribution provided:",
            dist,
            "\nSupported distributions:",
            distributions,
        )
        exit(1)
    if fname not in func_names:
        print("Unknown function provided:", fname, "\nSupported functions:", func_names)
        exit(1)


def function_str(fname):
    if fname == "max":
        return r"$f_{\max}(\vec{x}) = \max_{i} x_i$"
    if fname == "min":
        return r"$f_{\min}(\vec{x}) = \min_{i} x_i$"
    if fname == "var":
        return r"$f_{\sigma}(\vec{x}) = \frac{1}{n-1}\sum_i (x_i - \mu)^2 $"

    if fname == "var_nf":
        return r"$f_{\sigmr}(\vec{x}) = \sum_i (x_i - \mu)^2 $"
    else:
        return ""


def plot_discrete(fname, dist, param_str):
    verify_args(fname, dist)

    path = data_dir + fname + "/" + dist
    files = os.listdir(path)
    print(files)
    file = [f for f in files if Path(f).stem == param_str][0]
    print(file)

    fig, ax = plt.subplots()
    plt.ylabel(r"Entropy (bits)")

    func_str = function_str(fname)

    with open(path + "/" + file) as json_file:
        json_data = json.load(json_file)

    plot_lines = []
    cc = itertools.cycle(colors)
    alph = 0.5

    for js in json_data["awae_data"][:4]:
        c = next(cc)
        print(js[0], js[1])
        numSpec = js[0]
        awae_dict = js[1]
        x_A = np.array([np.array(xi) for xi in js[1]])[
            :, 0
        ]  # col slice, attacker inputs
        awae = np.array([np.array(xi) for xi in js[1]])[:, 1]  # col slice, awaes
        (l2,) = plt.plot(
            x_A,
            awae,
            marker="o",
            color=c,
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S\rvert\ = %s$" % numSpec,
        )
        plot_lines.append(l2)

        print(x_A)
        print(awae)
        # for j in

    legend1 = plt.legend(handles=plot_lines, loc="upper left", fontsize=14)
    plt.gca().add_artist(legend1)
    
    
    plt.xticks(np.arange(0, 8, 1), minor=True)  # set minor ticks on x-axis
    plt.yticks(np.arange(0, 8, 1), minor=True)  # set minor ticks on y-axis
    plt.tick_params(which="minor", length=0)  # remove minor tick lines

    ax.hlines(
        y=json_data["target_init_entropy"], xmin=0, xmax=7, linewidth=2, color="black"
    )
    plt.grid()
    plt.grid(which="minor", alpha=0.1)  # draw grid for minor ticks on x-axis

    # plt.text(2, 2, func_str)

    plt.text(
        0.976,
        0.03,
        func_str,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3,"edgecolor":'black'},
    )

    out_path = fig_dir + fname + "/" + dist
    Path(out_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        out_path + "/" + param_str + "_discrete_leakage.pdf",
        bbox_inches="tight",
    )
    plt.close("all")


def main():
    plot_discrete("max", "uniform_int", "(0,7)")
    plot_discrete("var", "uniform_int", "(0,7)")
    plot_discrete("var_nd", "uniform_int", "(0,7)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
