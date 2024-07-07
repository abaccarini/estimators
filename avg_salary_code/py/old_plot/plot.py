# import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
import numpy as np
import subprocess
from matplotlib import cm
from matplotlib import rc
from numpy import genfromtxt
import itertools
from matplotlib.lines import Line2D

import logging

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


def getExperimentName(experiment):
    return None
    if experiment == "comparison":
        equation = r"$f(x,y) = \left( x  \stackrel{?}{>}  y \right)$"
        return equation
    elif experiment == "sum":
        equation = r"$f(x_i,y_j,z_k) = \sum_{i,j,k} x_i + y_j + z_k$"
        return equation
    elif experiment == "dot_prod":
        equation = r"$f(\vec{x},\vec{y}) = \vec{x} \cdot \vec{y} = \sum_{i} x_i y_i$"
        return equation
    elif experiment == "advanced":
        equation = r"$f(x,y,z) = 3xy - 2xz$"
        return equation
    elif experiment == "decisionTree":
        equation = r"$f(x,y,z) =$ tree"
        return equation
    else:
        print("experiment name not found")
        return 0


def getDistName(id):
    if id == 1:
        return "binomial"
    elif id == 0:
        return "uniform"


def main(experiment):
    print()
    experimentName = experiment.split("/")[-2]

    # cmd = 'ls -I  '+experimentName +".pdf " + str(experiment) +'/'
    cmd = "ls  " + str(experiment) + "/"
    print(cmd)
    result = subprocess.check_output(cmd, shell=True)

    # result = str(result)

    result = str(result)[2:-1]  # removing first 2 and last characters

    ls_experiments = str(result).split("\\n")
    ls_experiments = ls_experiments[:-1]
    # print(ls_experiments)

    plotExperiments(experiment, ls_experiments)
    return 0


def plotExperiments(experiment, experimentDirectories):

    experimentName = experiment.split("/")[-2]
    out_path = "../../proof/figs/twae_vs_awae.pdf"
    out_path_png = "../../proof/figs/twae_vs_awae.png"
    # out_path = experiment+'/'+experimentName +'.pdf'
    print(out_path)
    # print(experimentName)
    fig, ax = plt.subplots()
    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"Input $x_T$ or $x_A$")
    plt.title(getExperimentName(experimentName), y=1.05)

    # with PdfPages(out_path) as pdf:

    plot_lines = []
    all_specs = []
    cc = itertools.cycle(colors)

    for single_experiment in experimentDirectories:

        c = next(cc)
        # ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        # fig = plt.figure()

        # extracting all data
        awae = genfromtxt(
            experiment + "/" + single_experiment + "/awae.csv", delimiter=","
        )
        twae = genfromtxt(
            experiment + "/" + single_experiment + "/twae.csv", delimiter=","
        )
        params = genfromtxt(
            experiment + "/" + single_experiment + "/parameters.csv", delimiter=","
        )
        params = params[:, 1]
        num_spec = params[3]
        all_specs.append(num_spec)
        print(params)
        print(single_experiment)

        (l1,) = plt.plot(
            twae[:, 0],
            twae[:, -1],
            marker="o",
            color=c,
            alpha=alph,
            linestyle="-",
            label="twae$(x_T)$",
        )
        (l2,) = plt.plot(
            awae[:, 0],
            awae[:, -1],
            marker="x",
            color=c,
            alpha=alph,
            linestyle="--",
            label="awae$(x_A)$",
        )

        plot_lines.append([l1, l2])

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            alpha=alph,
            linestyle="-",
            label=r"twae$(x_T)$",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="x",
            alpha=alph,
            linestyle="--",
            label=r"awae$(x_A)$",
        ),
    ]

    legend_elements_2 = [
        Line2D(
            [0],
            [0],
            color="red",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S \rvert = 1$",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S \rvert = 2$",
        ),
        Line2D(
            [0],
            [0],
            color="teal",
            marker="",
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S \rvert = 3$",
        ),
    ]
    dists = [r"$\lvert S \rvert = %s$" % int(n) for n in all_specs]

    legend1 = plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        # loc="lower center",
        # prop={"size": 14},
        ncol=2,
    )

    legend2 = plt.legend(
        handles=legend_elements_2,
        loc="lower center",
        # bbox_to_anchor=(0.5, -0.17),
        # loc="lower center",
        # prop={"size": 14},
        ncol=1,
    )
    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"twae$(x_T)$", r"awae$(x_A)$"],
    #     loc="upper right",
    # )
    # plt.legend([l[0] for l in plot_lines], dists, loc="lower center")
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path_png, bbox_inches="tight", transparent=True)


def plotExperiments2():

    out_path = "../../../proof/figs/min_awae.pdf"
    out_path_png = "../../../proof/figs/min_awae.png"
    print(out_path)
    # print(experimentName)
    fig, ax = plt.subplots()
    plt.ylabel(r"Entropy (bits)")
    plt.xlabel(r"Input $x_A$")
    # plt.title(getExperimentName(experimentName), y=1.05)

    # with PdfPages(out_path) as pdf:

    plot_lines = []
    all_specs = []
    alph = 0.5
    cc = itertools.cycle(colors)

    awae_1 = [3.0458] * 16
    awae_2 = [3.0458] * 16
    awae_3 = [3.23182] * 16
    awae_4 = [3.29848] * 16
    inputs = list(range(0, 16))

    c = next(cc)
    # ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    # fig = plt.figure()

    # extracting all data
    # params = params[:, 1]
    # num_spec = params[3]
    # all_specs.append(num_spec)
    # print(params)
    # # print(single_experiment)

    (l1,) = plt.plot(
        inputs,
        awae_1,
        marker="x",
        color=c,
        alpha=alph,
        linestyle="-",
        label=r"$\lvert S \rvert = 1$",
        # label="$\text{awae}_{\infty}(x_A)$",
    )
    c = next(cc)

    (l2,) = plt.plot(
        inputs,
        awae_2,
        marker="s",
        color=c,
        alpha=alph,
        linestyle="--",
        label=r"$\lvert S \rvert = 2$",
    )
    c = next(cc)

    (l3,) = plt.plot(
        inputs,
        awae_3,
        marker="o",
        color=c,
        alpha=alph,
        linestyle="-.",
        label=r"$\lvert S \rvert = 3$",
    )
    c = next(cc)

    (l4,) = plt.plot(
        inputs,
        awae_4,
        marker="v",
        color=c,
        alpha=alph,
        linestyle=":",
        label=r"$\lvert S \rvert = 4$",
    )

    plot_lines.append([l1, l2])

    legend_elements = [
        # Line2D(
        #     [0],
        #     [0],
        #     color="black",
        #     marker="o",
        #     alpha=alph,
        #     linestyle="-",
        #     label=r"twae$(x_T)$",
        # ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="x",
            alpha=alph,
            linestyle="-",
            label=r"$\text{awae}_{\infty}(x_A)$",
        ),
    ]

    legend_elements_2 = [
        Line2D(
            [0],
            [0],
            color="red",
            marker="x",
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S \rvert = 1$",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            marker="s",
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S \rvert = 2$",
        ),
        Line2D(
            [0],
            [0],
            color="teal",
            marker="o",
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S \rvert = 3$",
        ),        
        Line2D(
            [0],
            [0],
            color="magenta",
            marker="x",
            alpha=alph,
            linestyle="-",
            label=r"$\lvert S \rvert = 4$",
        ),
    ]
    dists = [r"$\lvert S \rvert = %s$" % int(n) for n in range(1,5)]

    legend1 = plt.legend(
        # handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        # loc="lower center",
        # prop={"size": 14},
        ncol=2,
    )

    legend2 = plt.legend(
        # handles=legend_elements_2,
        # loc="best",
        loc="lower right",
        bbox_to_anchor=(1.0, 0.17),
        # bbox_to_anchor=(0.5, -0.17),
        # loc="lower center",
        # prop={"size": 14},
        ncol=1,
    )
    # legend1 = plt.legend(
    #     plot_lines[0],
    #     [r"twae$(x_T)$", r"awae$(x_A)$"],
    #     loc="upper right",
    # )
    # plt.legend([l[0] for l in plot_lines], dists, loc="lower center")
    # plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(out_path_png, bbox_inches="tight", transparent=True)
    # plt.show()


if __name__ == "__main__":
    # cmd = "../output_old/sum/n16"
    # main(cmd)
    plotExperiments2()
