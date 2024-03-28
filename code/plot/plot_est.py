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
from matplotlib.lines import Line2D
from labellines import labelLine, labelLines

# logging.getLogger("matplotlib.font_manager").disabled = True

plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,
        # "font.family": "sans-serif",
        # "font.sans-serif": "Helvetica",
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb,sfmath,mathtools}\newcommand\floor[1]{\lfloor#1\rfloor} \newcommand\ceil[1]{\lceil#1\rceil} ",
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
distributions = ["uniform_int", "uniform_real", "normal", "lognormal", "poisson"]
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
    if fname == "median":
        return r"$f_{\text{median}}(\vec{x}) = x_{\floor{(n +1)/2}}$"
    if fname == "var":
        return r"$f_{\sigma^2}(\vec{x}) = \frac{1}{n-1}\sum_i (x_i - \mu)^2 $"
    if fname == "var_nd":
        return r"$f_{\sigma^2}(\vec{x}) = \sum_i (x_i - \mu)^2 $"
    else:
        return ""


def getBounds(dist, param_str):
    if dist == "poisson":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        return (0, 3 * int(raw_str) + 1)
    if dist == "uniform_int":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (int(bounds[0]), int(bounds[1]) + 1)



def getParams(dist, param_str):
    if dist == "poisson":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        return int(raw_str)
    if dist == "uniform_int":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (int(bounds[0]), int(bounds[1]) )



def plot_discrete(fname, dist, param_str):
    verify_args(fname, dist)

    path = data_dir + fname + "/" + dist
    files = os.listdir(path)
    print(files)
    file = [f for f in files if Path(f).stem == param_str][0]
    # print(file)
    lower_bound, upper_bound = getBounds(dist, param_str)

    fig, ax = plt.subplots()
    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"Input  $x_A$")
    
    plt.grid()
    plt.grid(which="minor", alpha=0.1)  # draw grid for minor ticks on x-axis
    ax2 = ax.twiny()
    ax2.spines["bottom"].set_position(("axes", -0.01))
    ax2.xaxis.set_ticks_position("top")
    ax2.spines["bottom"].set_visible(False)
    
    if dist == "poisson":
        lam = getParams(dist, param_str)
        ax2.axvline(lam, linestyle="--", alpha = 0.5, color='black')
        ax2.set_xticks([lam])
        ax2.set_xticklabels([r'$\lambda = %s$'%lam ], rotation=0, color='red')
    if dist == "uniform_int":
        a,b = getParams(dist, param_str)
        n = float( a + b )/2.0
        ax2.axvline(float( a + b )/2.0, linestyle="--", alpha = 0.5, color='black')
        ax2.set_xticks([n])
        ax2.set_xticklabels([r'$ \frac{a + b}{2}= %s$'%n ], rotation=0, color='red')

    

    func_str = function_str(fname)

    with open(path + "/" + file) as json_file:
        json_data = json.load(json_file)

    plot_lines = []
    cc = itertools.cycle(colors)
    alph = 0.5
    
    
    # print("bounds : ", lower_bound, upper_bound)
    for js in json_data["awae_data"][:5]:
        c = next(cc)
        numSpec = js[0]
        awae_dict = js[1]
        x_A = np.array([np.array(xi) for xi in js[1]])[
            :, 0
        ]  # col slice, attacker inputs
        awae = np.array([np.array(xi) for xi in js[1]])[:, 1]  # col slice, awaes
        label = r"$\lvert S\rvert\ = %s$" % numSpec
        (l2,) = plt.plot(
            x_A[lower_bound:upper_bound],
            awae[lower_bound:upper_bound],
            marker="o",
            color=c,
            alpha=alph,
            linestyle="-",
            label=label
        )


        plot_lines.append(l2)
    # labelLines(ax.get_lines(), zorder=4)
    legend1 = plt.legend(handles=plot_lines,
                         loc="best",
                         bbox_to_anchor=(1.32, 0.7),
                         fontsize=14
                         )
    plt.gca().add_artist(legend1)

    plt.xticks(
        np.arange(lower_bound, upper_bound, 1), minor=True
    )  # set minor ticks on x-axis
    plt.yticks(
        np.arange(lower_bound, upper_bound, 1), minor=True
    )  # set minor ticks on y-axis
    plt.tick_params(which="minor", length=0)  # remove minor tick lines

    target_init_label = r"$H(X_T)$"
    ax.hlines(
        y=json_data["target_init_entropy"],
        xmin=lower_bound,
        xmax=(upper_bound) - 1,
        linewidth=2,
        color="black",
        label=target_init_label,
    )

    hline_legened = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=1.0,
            linestyle="-",
            label=target_init_label,
        )
    ]
    legend2 = plt.legend(handles=hline_legened, loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=14)
    plt.gca().add_artist(legend2)

    plt.text(
        0.8,
        -0.2,
        func_str,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3, "edgecolor": "black"},
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
    plot_discrete("median", "uniform_int", "(0,7)")

    plot_discrete("max", "poisson", "(4)")
    plot_discrete("var_nd", "poisson", "(4)")
    plot_discrete("median", "poisson", "(4)")
    
    plot_discrete("max", "poisson", "(8)")
    plot_discrete("var_nd", "poisson", "(8)")
    plot_discrete("median", "poisson", "(8)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
