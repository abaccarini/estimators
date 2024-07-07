# import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
import itertools
from decimal import Decimal

import numpy as np
import subprocess
from matplotlib import cm
from matplotlib import rc
from numpy import genfromtxt
import logging
from plot import getExperimentName
from test import H_T

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

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 16})
rc("text", usetex=True)


def last_4chars(x):
    return x[-4:]


def get_dist_name(exp_name, params=None):
    if exp_name == "sum_poisson":
        return r"Poisson $\lambda = N/2$"
    elif exp_name == "sum_uniform":
        return r"Uniform $\{0,N-1\}$"
    elif exp_name == "sum_lognorm":
        if params is not None:
            return r"$\log\mathcal{N}(%s, %s)$" % (params[0], params[1])
        else:
            return r"$\log\mathcal{N}$"
    elif exp_name == "sum_normal":
        if params is not None:
            return r"$\mathcal{N}(%s, %s)$" % (params[0], params[1])
        else:
            return r"$\mathcal{N}$"
    else:
        return "INVALID NAME"


def main(files, experimentName):
    # experiment =  experimentName + "sum_poisson/"

    # experimentName = experiment.split("/")[-2]
    # out_path = "../output/" + experimentName + "/"+experimentName +".pdf"
    # print(out_path)
    # cmd = 'ls -I  '+experimentName +".pdf " + str(experiment) +'/'
    # print(files)
    # files.sort(key=int)

    # print(files)
    # print(files)
    # print(files)

    for file in files:
        out_path = "../output/" + experimentName + "_" + file + ".pdf"
        N_vals = []
        all_awaes = []
        with PdfPages(out_path) as pdf:
            experiment = str(experimentName) + "/" + file
            cmd = "ls  ../output/" + str(experimentName) + "/" + file
            result = subprocess.check_output(cmd, shell=True)

            new_files = os.listdir("../output/" + experimentName + "/" + file)
            # files.sort(key=int)
            new_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            # print(new_files)
            for new_file in new_files:
                # print(file)
                # print(file)
                params = file.split("_")

                # print(params)

                N = new_file[:-4]
                # print(N)
                N_vals.append(N)
                # result = str(result)[2:-1]

                # ls_experiments = str(result).split("\\n")
                # ls_experiments = ls_experiments[:-1]

                # # print(ls_experiments)
                # awae = genfromtxt(
                #     "../output/" + experiment + "/" + ls_experiments[0] + "/awae.csv",
                #     delimiter=",",
                # )
                awae = genfromtxt(
                    "../output/" + experiment + "/" + new_file,
                    delimiter=",",
                )

                all_awaes.append(awae)
                fig, ax = plt.subplots()

                # plt.ylabel(r"$H(X_T \mid O)$ (bits)")
                plt.ylabel(r"Entropy (bits)")

                plt.xlabel(r"No. Spectators")
                plt.title(r"$O = X_T + X_S$")
                # lam = str(ls_experiments[0]).split("_")
                # lam = float(lam[-1])
                (l2,) = plt.plot(
                    awae[:, 0],
                    awae[:, 2],
                    marker="",
                    color="b",
                    linestyle="-",
                    label=r"$H(X_T)$",
                )

                (l2,) = plt.plot(
                    awae[:, 0],
                    awae[:, 1],
                    marker="",
                    color="r",
                    linestyle="-",
                    label=r"$ H(X_T \mid O)$",
                )

                plt.text(
                    -0.1,
                    1.1,
                    r"$N = %s$" % int(N),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
                )
                plt.text(
                    0.78,
                    0.32,
                    r"$(%s,%s)$" % (float(params[0]), float(params[1])),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
                )

                ax.legend(loc="lower right", borderpad=0.5, handlelength=1.0)

                # plt.savefig(fig,bbox_inches='tight')
                pdf.savefig(fig, bbox_inches="tight")
            upper_bound = 8
            print("plotting deltas")

            differences = []
            for idx, awae in enumerate(all_awaes):

                differences.append([awae[0, 2] - x for x in awae[:, 1]])

            fig, ax = plt.subplots()
            plt.ylabel(r"$H(X_T) - H(X_T \mid O)$ (bits)")
            plt.xlabel(r"No. Spectators")
            plt.title(r"$O = X_T + X_S$")
            for idx, diff in enumerate(differences):

                (l2,) = plt.plot(
                    all_awaes[0][0:upper_bound, 0],
                    diff[0:upper_bound],
                    marker="",
                    # color="r",
                    linestyle="-",
                    label=(r"$N = %s$" % int(N_vals[idx])),
                )
            # plt.yticks(np.arange(0.0, 1.0, step=0.1))
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.0, 0.99),
                borderpad=0.5,
                handlelength=1.0,
            )
            pdf.savefig(fig, bbox_inches="tight")
            # plt.clf()
        plt.close("all")
    return 0


def old_main(files, experimentName):
    # experiment =  experimentName + "sum_poisson/"

    # experimentName = experiment.split("/")[-2]
    out_path = "../output/" + experimentName + ".pdf"
    # out_path = "../output/" + experimentName + "/"+experimentName +".pdf"
    # print(out_path)
    # cmd = 'ls -I  '+experimentName +".pdf " + str(experiment) +'/'
    # print(files)
    files.sort(key=int)
    N_vals = []
    all_awaes = []
    with PdfPages(out_path) as pdf:
        for file in files:
            experiment = str(experimentName) + "/" + file
            cmd = "ls  ../output/" + str(experimentName) + "/" + file
            result = subprocess.check_output(cmd, shell=True)
            param = str(experiment).split("/")
            N = int(param[-1])
            N_vals.append(N)
            result = str(result)[2:-1]

            ls_experiments = str(result).split("\\n")
            ls_experiments = ls_experiments[:-1]

            awae = genfromtxt(
                "../output/" + experiment + "/" + ls_experiments[0] + "/awae.csv",
                delimiter=",",
            )
            all_awaes.append(awae)
            fig, ax = plt.subplots()

            # plt.ylabel(r"$H(X_T \mid O)$ (bits)")
            plt.ylabel(r"Entropy (bits)")
            plt.xlabel(r"No. Spectators")
            plt.title(r"$O = X_T + X_S$")
            # lam = str(ls_experiments[0]).split("_")
            # lam = float(lam[-1])
            (l2,) = plt.plot(
                awae[:, 0],
                awae[:, 2],
                marker="",
                color="b",
                linestyle="-",
                label=r"$H(X_T)$",
            )

            (l2,) = plt.plot(
                awae[:, 0],
                awae[:, 1],
                marker="",
                color="r",
                linestyle="-",
                label=r"$ H(X_T \mid O)$",
            )

            plt.text(
                -0.1,
                1.1,
                r"$N = %s$" % int(N),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
            )
            ax.legend(loc="lower right", borderpad=0.5, handlelength=1.0)

            # plt.savefig(fig,bbox_inches='tight')
            pdf.savefig(fig, bbox_inches="tight")
        upper_bound = 8
        print("plotting deltas")

        differences = []
        for idx, awae in enumerate(all_awaes):

            differences.append([awae[0, 2] - x for x in awae[:, 1]])

        fig, ax = plt.subplots()
        plt.ylabel(r"$H(X_T) - H(X_T \mid O)$ (bits)")
        plt.xlabel(r"No. Spectators")
        plt.title(r"$O = X_T + X_S$")
        for idx, diff in enumerate(differences):

            (l2,) = plt.plot(
                all_awaes[0][0:upper_bound, 0],
                diff[0:upper_bound],
                marker="",
                # color="r",
                linestyle="-",
                label=(r"$N = %s$" % int(N_vals[idx])),
            )
        # plt.yticks(np.arange(0.0, 1.0, step=0.1))
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.4, 0.99),
            borderpad=0.5,
            handlelength=1.0,
        )
        pdf.savefig(fig, bbox_inches="tight")
        # plt.clf()
    plt.close("all")

    # print(N_vals)
    # print(all_awaes)
    return 0


def plot_multiple_exps(all_exps, N):
    # dir_path = "../output/"
    dir_path = "../../proof/figs/"
    # out_path = dir_path + exp_name + "_all_exps.pdf"

    out_path = dir_path + "all_exps.pdf"
    # print(out_path)
    # Check whether the specified path exists or not
    # isExist = os.path.exists(dir_path)
    data = []
    data.append(
        genfromtxt(
            "../output/" + all_exps[0] + "/" + str(N) + "/awae.csv",
            delimiter=",",
        )
    )
    data_poisson = genfromtxt(
        "../output/" + all_exps[0] + "/" + str(N) + "/awae.csv",
        delimiter=",",
    )
    # print(data_poisson)

    data_uniform = genfromtxt(
        "../output/" + all_exps[1] + "/" + str(N) + "/sum_uniform/awae.csv",
        delimiter=",",
    )

    data.append(
        genfromtxt(
            "../output/" + all_exps[1] + "/" + str(N) + "/sum_uniform/awae.csv",
            delimiter=",",
        )
    )

    # data.append(
    #     genfromtxt(
    #         "../output/"
    #         + all_exps[2]
    #         + "/"
    #         + str(N)
    #         + "/sum_lognorm_0.000000_0.250000/awae.csv",
    #         delimiter=",",
    #     )
    # )

    with PdfPages(out_path) as pdf:

        fig, ax = plt.subplots()
        colors = ["b", "r", "g", "c"]
        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = plt.plot(
                dat[:, 0],
                dat[:, 1],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T \mid O)$",
            )
            (l2,) = plt.plot(
                dat[:, 0],
                dat[:, 2],
                marker="",
                color=c,
                linestyle="--",
                label=r"$H(X_T)$",
            )
            plot_lines.append([l1, l2])

        plt.title(r"$O = X_T + X_S$, Multiple Distributions")
        plt.text(
            -0.1,
            1.1,
            r"$N = %s$" % int(N),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
        )
        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"No. Spectators")

        dists = [
            r"Poisson, $\lambda = N/2$",
            r"Uniform, $\{0,N-1\}$",
            # r"Lognormal, $\sigma^2 = 0.25^2$",
        ]
        legend1 = plt.legend(
            plot_lines[0],
            [
                r"$H(X_T \mid O)$",
                r"$H(X_T)$",
            ],
            bbox_to_anchor=(1.5, 0.99),
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.7, 0.3))
        plt.gca().add_artist(legend1)
        plt.savefig(dir_path + "multiple_dists.pdf", bbox_inches="tight")

        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()
        colors = ["b", "r", "g", "c"]
        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = plt.plot(
                dat[:50, 0],
                dat[:50, 2] - dat[:50, 1],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T) - H(X_T \mid O)$",
            )

            plot_lines.append([l1])

        plt.title(r"$O = X_T + X_S$, Entropy Loss")

        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"No. Spectators")

        plt.text(
            -0.1,
            1.1,
            r"$N = %s$" % int(N),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
        )
        dists = [
            r"Poisson, $\lambda = N/2$",
            r"Uniform, $\{0,N-1\}$",
            # r"Lognormal, $\sigma^2 = 0.25^2$",
        ]
        legend1 = plt.legend(
            plot_lines[0], [r"$H(X_T) - H(X_T \mid O)$"], bbox_to_anchor=(1.6, 0.99)
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.7, 0.3))
        plt.gca().add_artist(legend1)
        plt.savefig(dir_path + "entropy_loss.pdf", bbox_inches="tight")

        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()
        colors = ["b", "r", "g", "c"]
        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = plt.plot(
                dat[:50, 0],
                [((a - b) / a) * 100.0 for a, b in zip(dat[:50, 2], dat[:50, 1])],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T) - H(X_T \mid O)$",
            )

            plot_lines.append([l1])

        plt.title(r"$O = X_T + X_S$, Relative Entropy Loss")

        plt.ylabel(r"Percent Loss")
        plt.xlabel(r"No. Spectators")

        plt.text(
            -0.1,
            1.1,
            r"$N = %s$" % int(N),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
        )
        dists = [
            r"Poisson, $\lambda = N/2$",
            r"Uniform, $\{0,N-1\}$",
            # r"Lognormal, $\sigma^2 = 0.25^2$",
        ]
        legend1 = plt.legend(
            plot_lines[0],
            [r"$\frac{H(X_T) - H(X_T \mid O)}{H(X_T)}$"],
            bbox_to_anchor=(1.6, 0.99),
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.7, 0.3))
        plt.gca().add_artist(legend1)
        plt.savefig(dir_path + "rel_entropy_loss.pdf", bbox_inches="tight")

        pdf.savefig(fig, bbox_inches="tight")

    plt.close("all")
    return 0


def plot_exps_entropy_loss_discrete(exp_name, N):
    dir_path = "../../proof/figs/"
    out_path = dir_path + exp_name + "_all_exps.pdf"
    # print(out_path)
    # Check whether the specified path exists or not
    # isExist = os.path.exists(dir_path)

    data = []
    if exp_name == "sum_poisson":

        for n in N:
            data.append(
                genfromtxt(
                    "../output/" + exp_name + "/" + str(n) + "/results.csv",
                    delimiter=",",
                )
            )

    elif exp_name == "sum_uniform":
        for n in N:
            data.append(
                genfromtxt(
                    "../output/" + exp_name + "/" + str(n) + "/results.csv",
                    delimiter=",",
                )
            )
    else:
        print("invalid experiment name")
        return -1

    with PdfPages(out_path) as pdf:

        fig, ax1 = plt.subplots()
        # ax1.set_yscale('log', base=2)

        colors = [
            "red",
            "blue",
            "darkgreen",
            "magenta",
            "darkorange",
            "teal",
            "saddlebrown",
            "mediumslateblue",
            "black",
            "lime",
            "red",
            "blue",
            "darkgreen",
            "magenta",
            "darkorange",
        ]
        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:
            c = next(cc)
            (l1,) = ax1.plot(
                dat[:50, 0],
                dat[:50, 1],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T \mid O)$",
            )
            (l2,) = ax1.plot(
                dat[:50, 0],
                dat[:50, 2],
                marker="",
                color=c,
                linestyle="--",
                label=r"$H(X_T)$",
            )

            plot_lines.append([l1, l2])

        plt.title(r"$O = X_T + X_S$, %s" % get_dist_name(exp_name))
        plt.ylabel(r"Shannon Entropy (bits)")
        plt.xlabel(r"No. Spectators")

        dists = [r"$N = %s$" % n for n in N]
        #     r"Poisson, $\lambda = N/2$",
        #     r"Uniform, $\{0,N-1\}$",
        #     r"Lognormal, $\sigma^2 = 0.25^2$",
        # ]
        legend1 = plt.legend(
            plot_lines[0],
            [
                r"$H(X_T \mid O)$",
                r"$H(X_T)$",
                r"$h(X_T \mid O)$",
                r"$h(X_T)$",
            ],
            bbox_to_anchor=(1.0, 0.99),
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.42, 0.5))
        plt.gca().add_artist(legend1)
        plt.savefig(dir_path + exp_name + "_all_exps.pdf", bbox_inches="tight")

        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()
        # ax.set_yscale('log', base=2)

        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = plt.plot(
                dat[:20, 0],
                dat[:20, 2] - dat[:20, 1],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T) - H(X_T \mid O)$",
            )
            plot_lines.append([l1])

        plt.title(r"$O = X_T + X_S$, %s" % get_dist_name(exp_name))
        plt.ylabel(r"Shannon Entropy (bits)")
        plt.xlabel(r"No. Spectators")

        dists = [r"$N = %s$" % n for n in N]
        legend1 = plt.legend(
            plot_lines[0], [r"$H(X_T) - H(X_T \mid O)$"], bbox_to_anchor=(1.0, 0.99)
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.42, 0.5))
        plt.gca().add_artist(legend1)
        plt.savefig(
            dir_path + exp_name + "_all_exps_absolute_loss.pdf", bbox_inches="tight"
        )

        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()
        # ax.set_yscale('log', base=2)

        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = plt.plot(
                dat[:20, 0],
                [((a - b) / a) * 100.0 for a, b in zip(dat[:20, 2] - dat[:20, 1])],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T) - H(X_T \mid O)$",
            )
            plot_lines.append([l1])

        plt.title(r"$O = X_T + X_S$, %s" % get_dist_name(exp_name))
        plt.ylabel(r"Shannon Entropy (bits)")
        plt.xlabel(r"No. Spectators")

        dists = [r"$N = %s$" % n for n in N]
        legend1 = plt.legend(
            plot_lines[0], [r"$H(X_T) - H(X_T \mid O)$"], bbox_to_anchor=(1.0, 0.99)
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.42, 0.5))
        plt.gca().add_artist(legend1)
        plt.savefig(
            dir_path + exp_name + "_all_exps_relative_loss.pdf", bbox_inches="tight"
        )
        pdf.savefig(fig, bbox_inches="tight")

    plt.close("all")
    return 0


def plot_exps_entropy_loss_continuous(exp_name, N, params_string):
    dir_path = "../../proof/figs/"
    out_path = dir_path + exp_name + "_all_exps.pdf"
    # print(out_path)
    # Check whether the specified path exists or not
    # isExist = os.path.exists(dir_path)
    params = params_string.split("_")
    print(params)
    params = [Decimal(para).normalize() for para in params]
    # print(Decimal(params[1]).normalize())
    data = []

    if exp_name == "sum_lognorm":
        for n in N:
            data.append(
                genfromtxt(
                    "../output/"
                    + exp_name
                    + "/"
                    + params_string
                    + "/"
                    + str(n)
                    + ".csv",
                    delimiter=",",
                )
            )
    elif exp_name == "sum_normal":
        for n in N:
            data.append(
                genfromtxt(
                    "../output/"
                    + exp_name
                    + "/"
                    + params_string
                    + "/"
                    + str(n)
                    + ".csv",
                    delimiter=",",
                )
            )
    else:
        print("invalid experiment name")
        return -1

    with PdfPages(out_path) as pdf:

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        colors = [
            "red",
            "blue",
            "darkgreen",
            "magenta",
            "darkorange",
            "teal",
            "saddlebrown",
            "mediumslateblue",
            "black",
            "lime",
            "red",
            "blue",
            "darkgreen",
            "magenta",
            "darkorange",
        ]
        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = ax1.plot(
                dat[:50, 0],
                dat[:50, 1],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T \mid O)$",
            )
            (l2,) = ax1.plot(
                dat[:50, 0],
                dat[:50, 2],
                marker="",
                color=c,
                linestyle="--",
                label=r"$H(X_T)$",
            )

            (l3,) = ax2.plot(
                dat[:50, 0],
                dat[:50, 6],
                marker="x",
                color=c,
                linestyle="-.",
                label=r"$h(X_T \mid O)$",
            )
            (l4,) = ax2.plot(
                dat[:50, 0],
                dat[:50, 3],
                marker="",
                color=c,
                linestyle=":",
                label=r"$h(X_T)$",
            )

            plot_lines.append([l1, l2, l3, l4])

        plt.title(r"$O = X_T + X_S$, %s" % get_dist_name(exp_name, params))
        # plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"No. Spectators")

        ax1.set_ylabel(r"Shannon Entropy (bits)")
        ax2.tick_params(axis="y")
        ax2.set_ylabel(r"Differential Entropy (bits)")

        dists = [r"$N = %s$" % n for n in N]
        #     r"Poisson, $\lambda = N/2$",
        #     r"Uniform, $\{0,N-1\}$",
        #     r"Lognormal, $\sigma^2 = 0.25^2$",
        # ]
        legend1 = plt.legend(
            plot_lines[0],
            [
                r"$H(X_T \mid O)$",
                r"$H(X_T)$",
                r"$h(X_T \mid O)$",
                r"$h(X_T)$",
            ],
            bbox_to_anchor=(1.2, 0.99),
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.52, 0.5))
        plt.gca().add_artist(legend1)
        plt.savefig(
            dir_path + exp_name + "_" + params_string + "_all_exps.pdf",
            bbox_inches="tight",
        )

        pdf.savefig(fig, bbox_inches="tight")

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = ax1.plot(
                dat[:20, 0],
                dat[:20, 2] - dat[:20, 1],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T) - H(X_T \mid O)$",
            )
            (l2,) = ax2.plot(
                dat[:20, 0],
                dat[:20, 3] - dat[:20, 6],
                marker="x",
                color=c,
                linestyle="-.",
                label=r"$h(X_T) - h(X_T \mid O)$",
            )
            plot_lines.append([l1, l2])

        plt.title(r"$O = X_T + X_S$, %s" % get_dist_name(exp_name, params))
        # plt.ylabel(r"Shannon Entropy (bits)")
        plt.xlabel(r"No. Spectators")
        ax1.set_ylabel(r"Shannon Entropy (bits)")
        ax2.tick_params(axis="y")
        ax2.set_ylabel(r"Differential Entropy (bits)")

        dists = [r"$N = %s$" % n for n in N]
        legend1 = plt.legend(
            plot_lines[0],
            [r"$H(X_T) - H(X_T \mid O)$", r"$h(X_T) - h(X_T \mid O)$"],
            bbox_to_anchor=(1.0, 0.99),
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.52, 0.5))
        plt.gca().add_artist(legend1)
        plt.savefig(
            dir_path + exp_name + "_" + params_string + "_all_exps_loss.pdf",
            bbox_inches="tight",
        )

        pdf.savefig(fig, bbox_inches="tight")

    plt.close("all")
    return 0


def plot_exps_entropy_loss_continuous_vary_sigmas(exp_name, N, params_strings):
    dir_path = "../../proof/figs/"
    out_path = dir_path + exp_name + "_all_exps.pdf"
    # print(out_path)
    # Check whether the specified path exists or not
    # isExist = os.path.exists(dir_path)
    # params = params_string.split("_")
    # print(params)
    # params = [Decimal(para).normalize() for para in params]
    # print(Decimal(params[1]).normalize())

    all_data = [[]]
    if exp_name == "sum_normal" or exp_name == "sum_lognrom":
        for n in N:
            data = []
            for params in params_strings:
                data.append(
                    genfromtxt(
                        "../output/" + exp_name + "/" + params + "/" + str(n) + ".csv",
                        delimiter=",",
                    )
                )
            all_data.append(data)
    else:
        print("invalid experiment name")
        return -1

    with PdfPages(out_path) as pdf:

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        colors = [
            "red",
            "blue",
            "darkgreen",
            "magenta",
            "darkorange",
            "teal",
            "saddlebrown",
            "mediumslateblue",
            "black",
            "lime",
            "red",
            "blue",
            "darkgreen",
            "magenta",
            "darkorange",
        ]
        cc = itertools.cycle(colors)
        plot_lines = []

        for dat in data:

            c = next(cc)
            (l1,) = ax1.plot(
                dat[:50, 0],
                dat[:50, 1],
                marker="",
                color=c,
                linestyle="-",
                label=r"$H(X_T \mid O)$",
            )
            (l2,) = ax1.plot(
                dat[:50, 0],
                dat[:50, 2],
                marker="",
                color=c,
                linestyle="--",
                label=r"$H(X_T)$",
            )

            (l3,) = ax2.plot(
                dat[:50, 0],
                dat[:50, 6],
                marker="x",
                color=c,
                linestyle="-.",
                label=r"$h(X_T \mid O)$",
            )
            (l4,) = ax2.plot(
                dat[:50, 0],
                dat[:50, 3],
                marker="",
                color=c,
                linestyle=":",
                label=r"$h(X_T)$",
            )

            plot_lines.append([l1, l2, l3, l4])

        plt.title(r"$O = X_T + X_S$, %s" % get_dist_name(exp_name, params))
        # plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"No. Spectators")

        ax1.set_ylabel(r"Shannon Entropy (bits)")
        ax2.tick_params(axis="y")
        ax2.set_ylabel(r"Differential Entropy (bits)")

        dists = [r"$N = %s$" % n for n in N]
        #     r"Poisson, $\lambda = N/2$",
        #     r"Uniform, $\{0,N-1\}$",
        #     r"Lognormal, $\sigma^2 = 0.25^2$",
        # ]
        legend1 = plt.legend(
            plot_lines[0],
            [
                r"$H(X_T \mid O)$",
                r"$H(X_T)$",
                r"$h(X_T \mid O)$",
                r"$h(X_T)$",
            ],
            bbox_to_anchor=(1.2, 0.99),
        )
        plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.52, 0.5))
        plt.gca().add_artist(legend1)
        plt.savefig(
            dir_path + exp_name + "_" + params_string + "_all_exps.pdf",
            bbox_inches="tight",
        )

        pdf.savefig(fig, bbox_inches="tight")

        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()

        # cc = itertools.cycle(colors)
        # plot_lines = []

        # for dat in data:

        #     c = next(cc)
        #     (l1,) = ax1.plot(
        #         dat[:20, 0],
        #         dat[:20, 2] - dat[:20, 1],
        #         marker="",
        #         color=c,
        #         linestyle="-",
        #         label=r"$H(X_T) - H(X_T \mid O)$",
        #     )
        #     (l2,) = ax2.plot(
        #         dat[:20, 0],
        #         dat[:20, 3] - dat[:20, 6],
        #         marker="x",
        #         color=c,
        #         linestyle="-.",
        #         label=r"$h(X_T) - h(X_T \mid O)$",
        #     )
        #     plot_lines.append([l1, l2])

        # plt.title(r"$O = X_T + X_S$, %s" % get_dist_name(exp_name, params))
        # # plt.ylabel(r"Shannon Entropy (bits)")
        # plt.xlabel(r"No. Spectators")
        # ax1.set_ylabel(r"Shannon Entropy (bits)")
        # ax2.tick_params(axis="y")
        # ax2.set_ylabel(r"Differential Entropy (bits)")

        # dists = [r"$N = %s$" % n for n in N]
        # legend1 = plt.legend(
        #     plot_lines[0],
        #     [r"$H(X_T) - H(X_T \mid O)$", r"$h(X_T) - h(X_T \mid O)$"],
        #     bbox_to_anchor=(1.0, 0.99),
        # )
        # plt.legend([l[0] for l in plot_lines], dists, bbox_to_anchor=(1.52, 0.5))
        # plt.gca().add_artist(legend1)
        # plt.savefig(
        #     dir_path + exp_name + "_" + params_string + "_all_exps_loss.pdf",
        #     bbox_inches="tight",
        # )

        # pdf.savefig(fig, bbox_inches="tight")

    plt.close("all")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:

        experimentName = "sum_normal"
        cmd = "ls  ../output/" + experimentName + "/"
        files = os.listdir("../output/" + experimentName + "/")
        files = [item for item in files if ".pdf" not in item]
        # main(files, experimentName)
        # for file in files:

        # exps = ["sum_poisson", "sum_uniform", "sum_lognorm", "sum_normal"]
        exps = ["sum_poisson", "sum_uniform"]
        for exp in exps:
            cmd = "ls  ../output/" + exp + "/"
            files = os.listdir("../output/" + exp + "/")
            # print(files)
            files = [item for item in files if ".pdf" not in item]
            # print(files)
            # old_main(files, exp)
        # plot_multiple_exps(exps, 1024)

        N_vals = [8, 16, 32, 64, 128, 256]
        plot_exps_entropy_loss_discrete("sum_poisson", N_vals)
        plot_exps_entropy_loss_discrete("sum_uniform", N_vals)

        param_strings = [
            "0.000000_0.500000",
            "0.000000_1.000000",
            "0.000000_2.000000",
            "0.000000_4.000000",
            "0.000000_8.000000",
            "0.000000_32.000000",
            "0.000000_64.000000",
            "0.000000_16.000000",
            "0.000000_128.000000",
            "0.000000_256.000000",
            "0.000000_512.000000",
            "0.000000_1024.000000",
        ]
        N_vals = [8, 16, 32, 64]
        # plot_exps_entropy_loss_continuous("sum_normal", N_vals, param_strings[3])
        # plot_exps_entropy_loss_continuous("sum_lognorm", N_vals, param_strings[3])

        # cmd = "ls  ../output/sum_uniform/"
        # files = os.listdir("../output/sum_uniform/")
        # experimentName = 'sum_uniform'
        # # print(files)
        # files = [item for item in files if ".pdf" not in item]
        # # print(files)
        # for file in files:
        #     main("sum_uniform/" + file)

    else:
        main(sys.argv[1])
