import json
import numpy as np
import os
import math
import sys
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

plt.rcParams.update(
    {
        "font.size": 18,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb,sfmath,mathtools}\newcommand\floor[1]{\lfloor#1\rfloor} \newcommand\ceil[1]{\lceil#1\rceil} ",
    }
)


colors = [
    "red",
    "blue",
    "green",
    "magenta",
    "darkorange",
    "slategrey",
    "cyan",
    "blueviolet",
    "dodgerblue",
    "lightcoral",
]

func_names = [
    "max",
    "var_mu",
    "median",
    "median_min",
    "var",
]

distributions = ["uniform_int", "uniform_real", "normal", "lognormal", "poisson"]
data_dir = "../output_k1/"
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
        return r"$f_{\sigma^2}(\vec{x}) = \frac{1}{n}\sum_i (x_i - \mu)^2 $"
    if fname == "var_nd":
        return r"$f_{\sigma^2}(\vec{x}) = \sum_i (x_i - \mu)^2 $"
    if fname == "var_nd_mu":
        return r"$f_{(\mu, \sigma^2)}(\vec{x}) =  8\cdot \mu + \sum_i (x_i - \mu)^2 $"
    if fname == "var_mu":
        # return r"$f_{(\mu, \sigma^2)}(\vec{x}) =   (\frac{1}{n}\sum_i x_i ,\frac{1}{n}\sum_i (x_i - \mu)^2)  $"
        return r"$f_{(\mu, \sigma^2)}(\vec{x}) =   (f_{\mu}(\vec{x}),f_{\sigma^2}(\vec{x}))  $"
    else:
        return ""


def getBounds(dist, param_str):
    if dist == "poisson":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        return (0, 3 * int(raw_str) + 1)
    if dist == "uniform_int":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (int(bounds[0]), int(bounds[1]))
    if dist == "lognormal":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (0, 4 * float(bounds[1]))
    if dist == "normal":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (-4 * float(bounds[1]), 4 * float(bounds[1]))


def getParams(dist, param_str):
    if dist == "poisson":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        return float(raw_str)
    if dist == "uniform_int":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (float(bounds[0]), float(bounds[1]))
    if dist == "lognormal":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (float(bounds[0]), float(bounds[1]))
    if dist == "normal":
        raw_str = param_str[param_str.find("(") + 1 : param_str.find(")")]
        bounds = raw_str.split(",")
        return (float(bounds[0]), float(bounds[1]))


oe_str = {0: "even", 1: "odd"}
global oe_key
oe_key = 0


def plot_discrete(fname, dist, param_str):
    verify_args(fname, dist)

    path = data_dir + fname + "/" + dist
    files = os.listdir(path)
    # print(files)
    file = [f for f in files if Path(f).stem == param_str][0]
    # print(file)
    lower_bound, upper_bound = getBounds(dist, param_str)
    max_numspec = 6
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
        tick_upper = 3
        lam = getParams(dist, param_str)
        mean = lam
        stdev = np.sqrt(float(mean))
        # ax2.axvline(lam, linestyle="--", alpha=0.5, color="black")
        xtick = (
            [mean + (-1.0) * stdev * i for i in range(1, tick_upper)]
            + [mean]
            + [mean + stdev * i for i in range(1, tick_upper)]
        )
        xlabels = (
            [
                r"$\lambda {-} %s \sigma$" % (i if i > 1 else "")
                for i in range(1, tick_upper)
            ]
            + [r"$\lambda$"]
            + [
                r"$\lambda {+} %s \sigma$" % (i if i > 1 else "")
                for i in range(1, tick_upper)
            ]
        )

        ax2.set_xticks(xtick)
        ax2.set_xticklabels(xlabels)
        plt.xticks(fontsize=14)

        # ax2.set_xticklabels([r"$\lambda = %s$" % lam], rotation=0, color="red")
    if dist == "uniform_int":
        tick_upper = 2
        a, b = getParams(dist, param_str)
        # print(a, b)
        b = b - 1.0
        mean = (a + b) / 2.0
        stdev = np.sqrt(((b - a + 1.0) * (b - a + 1.0) - 1.0) / 12.0)
        xtick = (
            [mean - stdev * i for i in range(1, tick_upper)]
            + [mean]
            + [mean + stdev * i for i in range(1, tick_upper)]
        )
        xlabels = (
            [r"$n {-} %s \sigma$" % (i if i > 1 else "") for i in range(1, tick_upper)]
            + [r"$n$"]
            + [
                r"$n {+} %s \sigma$" % (i if i > 1 else "")
                for i in range(1, tick_upper)
            ]
        )
        # print("uniform a,b", a, b)
        # print("uniform stdev", stdev)
        # print("uniform xticks", xtick)
        ax2.set_xticks(xtick)
        ax2.set_xticklabels(xlabels)
        plt.xticks(fontsize=14)

    func_str = function_str(fname)

    with open(path + "/" + file) as json_file:
        json_data = json.load(json_file)

    plot_lines = []
    cc = itertools.cycle(colors)
    alph = 0.5

    t_init = json_data["target_init_entropy"]
    # print("bounds : ", lower_bound, upper_bound)
    # print(json_data["awae_data"])
    # return
    if fname == "median" or fname == "median_min":
        max_numspec = 11
    leakage_no_A = json_data["leakage_no_attacker"]

    for numSpec, val in json_data["awae_data"].items():
        leakage_val = leakage_no_A[numSpec]
        # print("leakage_val", leakage_val)

        def plotfn(col=None):
            if col is None:
                c = next(cc)
            else:
                c = col
            x_A = np.array([np.array(xi) for xi, vv in val.items()])
            awae = np.array([t_init - np.array(vv) for xi, vv in val.items()])
            label = r"$\lvert S\rvert\ = %s$" % numSpec
            (l2,) = plt.plot(
                x_A[lower_bound:upper_bound],
                awae[lower_bound:upper_bound],
                marker="o",
                color=c,
                alpha=alph,
                linestyle="-",
                label=label,
            )
            ax.hlines(
                y=t_init - leakage_val,
                xmin=lower_bound,
                xmax=(upper_bound) - 1,
                linewidth=2,
                color=c,
                linestyle="--",
            )

            plot_lines.append(l2)

        if int(numSpec) < max_numspec:
            if (int(numSpec) % 2 == oe_key) and (
                fname == "median" or fname == "median_min"
            ):
                col = colors[int(numSpec) - 1]
                plotfn(col)
            elif fname != "median" and fname != "median_min":
                plotfn()

    # legend1 = plt.legend(
    #     handles=plot_lines, loc="best", bbox_to_anchor=(1.32, 0.7), fontsize=14
    # )
    # plt.gca().add_artist(legend1)

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
            alpha=0.9,
            linestyle="--",
            label=r"$H(X_T \mid O)$",
        )
    ]
    legend2 = plt.legend(
        handles=hline_legened, loc="best", 
        # bbox_to_anchor=(1.3, 1.0),
        fontsize=14
    )
    plt.gca().add_artist(legend2)

    plt.text(
        0.5,
        0.95,
        target_init_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.95, "pad": 0, "edgecolor": "white"},
    )

    plt.text(
        0.5,
        -0.25,
        func_str,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3, "edgecolor": "black"},
    )

    out_path = fig_dir + fname + "/" + dist
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if fname == "median" or fname == "median_min":
        plt.savefig(
            out_path + "/" + param_str + "_" + oe_str[oe_key] + "_discrete_leakage.pdf",
            # out_path + "/" + param_str + "_discrete_leakage.pdf",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            # out_path + "/" + param_str + "_" +oe_str[oe_key]+"_discrete_leakage.pdf",
            out_path + "/" + param_str + "_discrete_leakage.pdf",
            bbox_inches="tight",
        )
    plt.close("all")


def plot_cont(fname, dist, param_str):
    verify_args(fname, dist)
    path = data_dir + fname + "/" + dist
    files = os.listdir(path)
    file = [f for f in files if Path(f).stem == param_str][0]
    lower_bound, upper_bound = getBounds(dist, param_str)
    max_numspec = 6
    fig, ax = plt.subplots()
    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"Input  $x_A$")

    plt.grid()
    ax2 = ax.twiny()
    ax2.spines["bottom"].set_position(("axes", -0.01))
    ax2.xaxis.set_ticks_position("top")
    ax2.spines["bottom"].set_visible(False)

    func_str = function_str(fname)

    with open(path + "/" + file) as json_file:
        json_data = json.load(json_file)

    plot_lines = []
    cc = itertools.cycle(colors)
    alph = 0.5
    t_init = json_data["target_init_entropy"]
    if fname == "median" or fname == "median_min":
        max_numspec = 11

    if dist == "normal":
        mu, sigma = getParams(dist, param_str)
        tick_upper = 4
        xtick = (
            [(-1.0) * sigma * i for i in range(1, tick_upper)]
            + [mu]
            + [sigma * i for i in range(1, tick_upper)]
        )
        xlabels = (
            [
                r"$\mu {-} %s \sigma$" % (i if i > 1 else "")
                for i in range(1, tick_upper)
            ]
            + [r"$\mu$"]
            + [
                r"$\mu {+} %s \sigma$" % (i if i > 1 else "")
                for i in range(1, tick_upper)
            ]
        )
        ax2.set_xticks(xtick)
        ax2.set_xticklabels(xlabels)
        plt.xticks(fontsize=14, rotation=45)

    if dist == "lognormal":
        # ax2 = ax.twiny()
        mu, sigma = getParams(dist, param_str)
        mean = np.exp(mu + (sigma * sigma) / 2)
        # ax2.axvline(mean, linestyle="--", alpha=0.5, color="black")
        ax2.set_xticks([mean])
        ax2.set_xticklabels([r"mean $= %.2f$" % mean], rotation=0, color="black")
    leakage_no_A = json_data["leakage_no_attacker"]

    for numSpec, val in json_data["awae_data"].items():
        leakage_val = leakage_no_A[numSpec]

        def plotfn(upper_bound, col=None):
            if col is None:
                c = next(cc)
            else:
                c = col
            x_A = np.array([float(xi) for xi, vv in val.items()])
            if max(x_A) > upper_bound:
                upper_bound = max(x_A)
            awae = np.array([t_init - np.array(vv) for xi, vv in val.items()])
            # awae = [x for x in awae if x > lower_bound and x < upper_bound]
            yhat = savgol_filter(awae, 51, 3)  # window size 51, polynomial order 3
            label = r"$\lvert S\rvert\ = %s$" % numSpec
            (l2,) = plt.plot(
                x_A, yhat, marker="", color=c, alpha=alph, linestyle="-", label=label
            )
            plot_lines.append(l2)
            ax.hlines(
                y=t_init - leakage_val,
                xmin=lower_bound,
                xmax=(upper_bound),
                linewidth=2,
                color=c,
                linestyle="--",
            )

        # if int(numSpec) < max_numspec:
        #     if (int(numSpec) % 2 == oe_key) and (
        #         fname == "median" or fname == "median_min"
        #     ):
        #         plotfn(upper_bound)
        #     elif fname != "median" and fname != "median_min":
        #         plotfn(upper_bound)

        if int(numSpec) < max_numspec:
            if (int(numSpec) % 2 == oe_key) and (
                fname == "median" or fname == "median_min"
            ):
                col = colors[int(numSpec) - 1]
                plotfn(upper_bound, col)
            elif fname != "median" and fname != "median_min":
                plotfn(upper_bound)

    legend1 = plt.legend(
        handles=plot_lines,
        loc="best",
        bbox_to_anchor=(1.32, 0.7),
        fontsize=14,
        # handles=plot_lines,
        # loc="best",
        # fontsize=14,
    )
    # plt.gca().add_artist(legend1)

    hline_legened = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="",
            alpha=0.9,
            linestyle="--",
            label=r"$H(X_T \mid O)$",
        )
    ]
    legend2 = plt.legend(
        handles=hline_legened, loc="best", 
        # bbox_to_anchor=(1.3, 1.0),
        fontsize=14
    )
    plt.gca().add_artist(legend2)


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
        xmax=(upper_bound),
        linewidth=2,
        color="black",
        # label=target_init_label,
    )

    plt.text(
        0.5,
        0.95,
        target_init_label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.95, "pad": 0, "edgecolor": "white"},
    )

    # plt.gca().add_artist(legend2)

    plt.text(
        0.5,
        -0.25,
        func_str,
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 3, "edgecolor": "black"},
    )

    plt.tick_params(which="minor", length=0)  # remove minor tick lines

    out_path = fig_dir + fname + "/" + dist
    Path(out_path).mkdir(parents=True, exist_ok=True)
    if fname == "median" or fname == "median_min":
        plt.savefig(
            out_path + "/" + param_str + "_" + oe_str[oe_key] + "_cont_leakage.pdf",
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            out_path + "/" + param_str + "_cont_leakage.pdf",
            bbox_inches="tight",
        )
        all_specs_legend = [
            Line2D(
                [0],
                [0],
                color=colors[int(numSpec) - 1],
                marker="",
                alpha=1.0,
                linestyle="-",
                label=r"$\lvert S\rvert\ = %s$" % numSpec,
            )
            for numSpec in range(1, 11)
        ]

        # fig, ax2 = plt.subplots()
        # plt.gca().add_artist(legend1)
        legend1 = plt.legend(
            handles=all_specs_legend,
            loc="center right",
            bbox_to_anchor=(1.0, 0.3),
            prop={"size": 16},
        )
        plt.autoscale(enable=True, axis="y", tight=True)
        figl2, axl2 = plt.subplots(figsize=(0, 0))
        axl2.axis(False)

        def flip(items, ncol):
            return itertools.chain(*[items[i::ncol] for i in range(ncol)])

        reorder = lambda l, nc: sum((l[i::nc] for i in range(nc)), [])

        axl2.legend(
            handles=reorder(all_specs_legend, 5),
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            columnspacing=0.5,
            borderpad=0.5,
            fontsize="small",
            ncol=5,
            handlelength=1,
        )
        figl2.savefig(
            fig_dir + "legend_text_only" + ".pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close("all")


def main():

    json_fname = data_dir + "p_strs.json"
    fname = open(json_fname)
    param_strs_dict = json.load(fname)

    disc_dists = [
        "uniform_int",
        "poisson",
    ]

    cont_dists = [
        "normal",
        # "lognormal",
    ]

    global oe_key
    oe_key = 0
    for dname in cont_dists:
        for func in func_names:
            for p_str in param_strs_dict[dname]:
                print(dname, p_str)
                plot_cont(func, dname, p_str)

    for dname in disc_dists:
        for func in func_names:
            for p_str in param_strs_dict[dname]:
                print(dname, p_str)
                plot_discrete(func, dname, p_str)

    oe_key = 1
    for dname in cont_dists:
        for func in func_names:
            for p_str in param_strs_dict[dname]:
                print(dname, p_str)
                plot_cont(func, dname, p_str)
    for dname in disc_dists:
        for func in func_names:
            for p_str in param_strs_dict[dname]:
                print(dname, p_str)
                plot_discrete(func, dname, p_str)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
