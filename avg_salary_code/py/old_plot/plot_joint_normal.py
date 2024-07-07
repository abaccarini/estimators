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
from matplotlib import figure
from numpy import genfromtxt
import logging
from plot import getExperimentName
from test import H_T
from natsort import natsorted


logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager

color_arr = [
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


def get_dist(exp_name):
    exp_name = exp_name.split("_")
    if "normal" in exp_name:
        return r"$X\sim  \mathcal{N} (\mu, \sigma^2)$"
    elif "poisson" in exp_name:
        return r"$X\sim  \text{Pois} (\lambda = N/2)$"
    else:
        print("DISTRIBUTION NOT FOUND", exp_name)
        return 0


def get_title(data, numTargets):

    if numTargets == 2:
        if data[1, 8] == 0 and data[2, 8] == 0:
            return r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}"
        else:
            return r"\begin{eqnarray*}O_1 &=X_{T_1} +X_S+ X_{S_1}\\O_2 &=X_{T_2}+X_S + X_{S_2}\end{eqnarray*}"
    elif numTargets == 1:

        if data[1, 8] == 0 and data[2, 8] == 0:
            return r"\begin{align*}O_1 = X_{T} + X_{S_1} \\ O_2 =X_{T} + X_{S_2}\end{align*}"
        else:
            return r"\begin{align*}O_1 &=X_{T} +X_S+ X_{S_1}\\O_2 &=X_{T}+X_S + X_{S_2}\end{align*}"
    elif numTargets == -1:

        if data[1, 8] == 0 and data[2, 8] == 0:
            return r"\begin{align*}O_1 = X_{T} + X_{S_1} \\ O_2 =X_{S_2}\end{align*}"
        else:
            return r"\begin{align*}O_1 &=X_{T} +X_S+ X_{S_1}\\O_2 &=X_S + X_{S_2}\end{align*}"


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


def main1(folder, experiment, experimentName, numTargets):
    # for file in files:
    dir_path = "../output/" + experimentName + "_plots/"
    out_path = dir_path + experiment + "_normal.pdf"
    print(out_path)
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")

    with PdfPages(out_path) as pdf:
        print(folder)
        # folder.sort(key=lambda x: (float(x), len(x)))
        # l = sorted(folder, key=lambda k: [*map(int, k.split('.'))])
        folder = natsorted(folder)
        folder.reverse()
        # print(l)

        # plt.figure(figsize=(1,15))
        # plt.rcParams["figure.figsize"] = [6, 6]
        fig, ax = plt.subplots()

        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"Unique Spectators per Exp")
        # plt.title(
        #     r"\begin{eqnarray*}O_1 &=X_T + X_{S_1}\\O_2 &=X_T + X_{S_2}\end{eqnarray*}",
        #     y=1.1,
        # )
        data = []
        for idx, dir in enumerate(folder):

            print(dir)
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            # print(result_path)
            # print(result_path)
            # print(result_path)
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            # temp_data = data
            plt.title(
                get_title(data, numTargets),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )

            # lam = str(ls_experiments[0]).split("_")
            # lam = float(lam[-1])
            (l2,) = plt.plot(
                data[:, 9],
                data[:, 13],
                marker="",
                color=color_arr[idx],
                linestyle="-",
                label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
            )

            (l2,) = plt.plot(
                data[:, 9],
                data[:, 14],
                marker="",
                color=color_arr[idx],
                linestyle="--",
                # label=r"$h(X_T)$" ,
            )
            (l2,) = plt.plot(
                data[:, 9],
                data[:, 15],
                marker="",
                color=color_arr[idx],
                linestyle="dotted",
                # label=r"$h(X_T)$" ,
            )

        #  \begin{eqnarray*}
        # - - = h(X_T) \\ \cdots = h(X_T\mid O_1)  \\ -- = h(X_T\mid O_1, O_2)
        #  \end{eqnarray*}

        ax.text(
            1.11,
            1.4,
            r"\begin{eqnarray*} - - &=& h(X_T) \\ \cdots &=& h(X_T\mid O_1)  \\ \text{---} &=& h(X_T\mid O_1, O_2) \end{eqnarray*}",
            # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.text(
            -0.11,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.9, 0.0),
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()

        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"Unique Spectators per Exp")
        # plt.title(
        #     r"\begin{eqnarray*}O_1 &=X_T + X_{S_1}\\O_2 &=X_T + X_{S_2}\end{eqnarray*}",
        #     y=1.1,
        # )
        for idx, dir in enumerate(folder):

            # print(dir)
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            plt.title(
                get_title(data, numTargets),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )
            # lam = str(ls_experiments[0]).split("_")
            # lam = float(lam[-1])
            (l2,) = plt.plot(
                data[:, 9],
                [a - b for a, b in zip(data[:, 15], data[:, 13])],
                # data[:, 13],
                marker="",
                color=color_arr[idx],
                linestyle="-",
                label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
            )

        ax.text(
            1.11,
            1.3,
            r"\begin{eqnarray*} h(X_{T} \mid O_1) - h(X_{T} \mid O_1, O_2)\end{eqnarray*}",
            # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.text(
            -0.11,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.9, 0.0),
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()

        plt.ylabel(r"Percent Decrease")
        plt.xlabel(r"Unique Spectators per Exp")
        # plt.title(
        #     r"\begin{eqnarray*}O_1 &=X_T + X_{S_1}\\O_2 &=X_T + X_{S_2}\end{eqnarray*}",
        #     y=1.1,
        # )
        for idx, dir in enumerate(folder):

            # print(dir)
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            plt.title(
                get_title(data, numTargets),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )
            # lam = str(ls_experiments[0]).split("_")
            # lam = float(lam[-1])
            (l2,) = plt.plot(
                data[0:50, 9],
                [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 14], data[0:50, 15])],
                # data[:, 13],
                marker="",
                color=color_arr[idx],
                linestyle="-",
                label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
            )

            (l2,) = plt.plot(
                data[0:50, 9],
                [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 15], data[0:50, 13])],
                # data[:, 13],
                marker="",
                color=color_arr[idx],
                linestyle="--",
                # label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$" % (data[1, 1], data[1, 3] , data[1, 5] , data[1, 7]),
                # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
            )
            (l2,) = plt.plot(
                data[0:50, 9],
                [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 14], data[0:50, 13])],
                # data[:, 13],
                marker="",
                color=color_arr[idx],
                linestyle="dotted",
                # label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$" % (data[1, 1], data[1, 3] , data[1, 5] , data[1, 7]),
                # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
            )

        ax.text(
            1.11,
            1.55,
            r"\begin{eqnarray*} \text{---} &=& \frac{h(X_T) -  h(X_T\mid O_1)}{h(X_T)} \\ - - &=&  \frac{h(X_T\mid O_1) - h(X_T \mid O_1, O_2)}{h(X_T\mid O_1)} \\  \dots &=&  \frac{h(X_T) - h(X_T \mid O_1, O_2)}{h(X_T)}\end{eqnarray*}",
            # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )

        ax.text(
            -0.11,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.9, 0.0),
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()
        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"Unique Spectators per Exp")

        dir = folder[0]
        result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
        # print(result_path)
        # print(result_path)
        # print(result_path)
        data = genfromtxt(
            result_path,
            delimiter=",",
        )
        plt.title(
            get_title(data, numTargets),
            # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
            y=1.1,
        )
        (l2,) = plt.plot(
            data[:, 9],
            data[:, 13],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"$h(X_T \mid O_1, O_2)$",
        )

        (l2,) = plt.plot(
            data[:, 9],
            data[:, 15],
            marker="",
            color=color_arr[2],
            linestyle="dotted",
            label=r"$h(X_T \mid O_1)$",
        )
        (l2,) = plt.plot(
            data[:, 9],
            data[:, 14],
            marker="",
            color=color_arr[1],
            linestyle="--",
            # label=r"$\sigma^2 = %s, h(X_T)$" % data[1, 1],
            label=r"$h(X_T)$",
        )
        ax.text(
            -0.11,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.6, 0.0),
        )

        ax.text(
            0.3,
            0.07,
            r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
            % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
            # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()

        plt.ylabel(r"Percent Decrease")
        plt.xlabel(r"Unique Spectators per Exp")
        # plt.title(
        #     r"\begin{eqnarray*}O_1 &=X_T + X_{S_1}\\O_2 &=X_T + X_{S_2}\end{eqnarray*}",
        #     y=1.1,
        # )

        # print(dir)
        result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
        data = genfromtxt(
            result_path,
            delimiter=",",
        )
        plt.title(
            get_title(data, numTargets),
            # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
            y=1.1,
        )
        # lam = str(ls_experiments[0]).split("_")
        # lam = float(lam[-1])
        (l2,) = plt.plot(
            data[0:50, 9],
            [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 14], data[0:50, 15])],
            # data[:, 13],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"$ \frac{h(X_T) -  h(X_T\mid O_1)}{h(X_T)}$"
            # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
        )

        (l2,) = plt.plot(
            data[0:50, 9],
            [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 15], data[0:50, 13])],
            # data[:, 13],
            marker="",
            color=color_arr[1],
            linestyle="--",
            label=r"$ \frac{h(X_T\mid O_1) - h(X_T \mid O_1, O_2)}{h(X_T\mid O_1)}$"
            # label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$" % (data[1, 1], data[1, 3] , data[1, 5] , data[1, 7]),
            # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
        )
        (l2,) = plt.plot(
            data[0:50, 9],
            [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 14], data[0:50, 13])],
            # data[:, 13],
            marker="",
            color=color_arr[2],
            linestyle="dotted",
            label=r"$ \frac{h(X_T) - h(X_T \mid O_1, O_2)}{h(X_T)}$"
            # label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$" % (data[1, 1], data[1, 3] , data[1, 5] , data[1, 7]),
            # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
        )

        # ax.text(
        #     1.11,
        #     1.1,
        #     r"\begin{eqnarray*} \text{---} &=& \frac{h(X_T) -  h(X_T\mid O_1)}{h(X_T)} \\ - - &=&  \frac{h(X_T\mid O_1) - h(X_T \mid O_1, O_2)}{h(X_T\mid O_1)} \\  \dots &=&  \frac{h(X_T) - h(X_T \mid O_1, O_2)}{h(X_T)}\end{eqnarray*}",
        #     # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
        #     transform=ax.transAxes,
        #     fontsize=14,
        #     verticalalignment="top",
        #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        # )

        ax.text(
            -0.11,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.6, 0.0),
        )

        ax.text(
            0.35,
            0.95,
            r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
            % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
            # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()
        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"Unique Spectators per Exp")

        dir = folder[-1]
        result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
        # print(result_path)
        # print(result_path)
        # print(result_path)
        data = genfromtxt(
            result_path,
            delimiter=",",
        )
        plt.title(
            get_title(data, numTargets),
            # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
            y=1.1,
        )
        (l2,) = plt.plot(
            data[:, 9],
            data[:, 13],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"$h(X_T \mid O_1, O_2)$",
        )

        (l2,) = plt.plot(
            data[:, 9],
            data[:, 15],
            marker="",
            color=color_arr[2],
            linestyle="dotted",
            label=r"$h(X_T \mid O_1)$",
        )
        (l2,) = plt.plot(
            data[:, 9],
            data[:, 14],
            marker="",
            color=color_arr[1],
            linestyle="--",
            # label=r"$\sigma^2 = %s, h(X_T)$" % data[1, 1],
            label=r"$h(X_T)$",
        )
        ax.text(
            -0.11,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.6, 0.0),
        )
        ax.text(
            0.3,
            0.07,
            r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
            % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
            # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        fig, ax = plt.subplots()

        plt.ylabel(r"Percent Decrease")
        plt.xlabel(r"Unique Spectators per Exp")
        # plt.title(
        #     r"\begin{eqnarray*}O_1 &=X_T + X_{S_1}\\O_2 &=X_T + X_{S_2}\end{eqnarray*}",
        #     y=1.1,
        # )

        # print(dir)
        result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
        data = genfromtxt(
            result_path,
            delimiter=",",
        )
        plt.title(
            get_title(data, numTargets),
            # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
            y=1.1,
        )
        # lam = str(ls_experiments[0]).split("_")
        # lam = float(lam[-1])
        (l2,) = plt.plot(
            data[0:50, 9],
            [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 14], data[0:50, 15])],
            # data[:, 13],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"$ \frac{h(X_T) -  h(X_T\mid O_1)}{h(X_T)}$"
            # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
        )

        (l2,) = plt.plot(
            data[0:50, 9],
            [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 15], data[0:50, 13])],
            # data[:, 13],
            marker="",
            color=color_arr[1],
            linestyle="--",
            label=r"$ \frac{h(X_T\mid O_1) - h(X_T \mid O_1, O_2)}{h(X_T\mid O_1)}$"
            # label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$" % (data[1, 1], data[1, 3] , data[1, 5] , data[1, 7]),
            # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
        )
        (l2,) = plt.plot(
            data[0:50, 9],
            [((a - b) / a) * 100.0 for a, b in zip(data[0:50, 14], data[0:50, 13])],
            # data[:, 13],
            marker="",
            color=color_arr[2],
            linestyle="dotted",
            label=r"$ \frac{h(X_T) - h(X_T \mid O_1, O_2)}{h(X_T)}$"
            # label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$" % (data[1, 1], data[1, 3] , data[1, 5] , data[1, 7]),
            # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
        )

        # ax.text(
        #     1.11,
        #     1.1,
        #     r"\begin{eqnarray*} \text{---} &=& \frac{h(X_T) -  h(X_T\mid O_1)}{h(X_T)} \\ - - &=&  \frac{h(X_T\mid O_1) - h(X_T \mid O_1, O_2)}{h(X_T\mid O_1)} \\  \dots &=&  \frac{h(X_T) - h(X_T \mid O_1, O_2)}{h(X_T)}\end{eqnarray*}",
        #     # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
        #     transform=ax.transAxes,
        #     fontsize=14,
        #     verticalalignment="top",
        #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        # )

        ax.text(
            -0.11,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.6, 0.0),
        )

        ax.text(
            0.35,
            0.95,
            r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
            % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
            # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        plt.clf()

    plt.close("all")

    return 0


def main2(all_folders, experiment, sizes, experimentName, numTargets):

    dir_path = "../output/" + experimentName + "_plots/"
    out_path = dir_path + experiment + "_normal.pdf"

    print(out_path)
    # print(out_path)
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")
    # print(out_path)

    if numTargets == 1 or numTargets == -1:
        target_string = r"X_T"
    elif numTargets == 2:
        target_string = r"X_{T_1}"

    with PdfPages(out_path) as pdf:
        for sz in sizes:

            folder = [k for k in all_folders if sz + experiment in k]

            # print(folder)
            # folder.sort(key=lambda x: (float(x), len(x)))
            # l = sorted(folder, key=lambda k: [*map(int, k.split('.'))])
            folder = natsorted(folder)
            folder.reverse()
            # print(l)

            fig, ax = plt.subplots()

            plt.ylabel(r"Entropy (bits)")
            plt.xlabel(r"\% Shared spectators")

            for idx, dir in enumerate(folder):

                # print(dir)
                result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
                # print(result_path)
                # print(result_path)
                # print(result_path)
                data = genfromtxt(
                    result_path,
                    delimiter=",",
                )
                plt.title(
                    get_title(data, numTargets),
                    # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                    y=1.1,
                )
                # lam = str(ls_experiments[0]).split("_")
                # lam = float(lam[-1])
                (l2,) = plt.plot(
                    data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                    data[:, 13],
                    marker="",
                    color=color_arr[idx],
                    linestyle="-",
                    label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                    % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                )

                (l2,) = plt.plot(
                    data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                    data[:, 15],
                    marker="",
                    color=color_arr[idx],
                    linestyle="dotted",
                    # label=r"$h(X_{T} \mid O_1)$",
                )
                (l2,) = plt.plot(
                    data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                    data[:, 14],
                    marker="",
                    color=color_arr[idx],
                    linestyle="--",
                    # label=r"$h(X_{T} )$",
                )
            ax.text(
                -0.11,
                1.1,
                get_dist(experimentName),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            )
            ax.legend(
                loc="lower right",
                borderpad=0.5,
                handlelength=1.0,
                bbox_to_anchor=(1.9, 0.0),
            )
            ax.text(
                1.1,
                -0.1,
                r"No. Spectators per Exp = %s" % (data[1, 8] + data[1, 9]),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="bottom",
            )

            # ax.set_ylim(bottom=0, top=None)
            ax.set_xlim(left=0, right=None)
            pdf.savefig(fig, bbox_inches="tight")

            fig, ax = plt.subplots()

            plt.ylabel(r"Entropy (bits)")
            plt.xlabel(r"\% Shared spectators")
            # plt.title(
            #     r"\begin{eqnarray*}O_1 &=X_T + X_{S_1}\\O_2 &=X_T + X_{S_2}\end{eqnarray*}",
            #     y=1.1,
            # )
            for idx, dir in enumerate(folder):

                # print(dir)
                result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
                # print(result_path)
                # print(result_path)
                # print(result_path)
                data = genfromtxt(
                    result_path,
                    delimiter=",",
                )
                plt.title(
                    get_title(data, numTargets),
                    # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                    y=1.1,
                )
                (l2,) = plt.plot(
                    data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                    [a - b for a, b in zip(data[:, 15], data[:, 13])],
                    # data[:, 13],
                    marker="",
                    color=color_arr[idx],
                    linestyle="-",
                    label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                    % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                    # label=r"\begin{eqnarray*}&\sigma^2 = \{%s, %s\},\\&\end{eqnarray*}"%( data[1, 1], data[1, 5])
                )

            ax.text(
                1.11,
                1.3,
                r"\begin{eqnarray*} h(%s \mid O_1) - h(%s \mid O_1, O_2)\end{eqnarray*}"
                % (target_string, target_string),
                # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            )
            ax.text(
                -0.11,
                1.1,
                get_dist(experimentName),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            )
            ax.legend(
                loc="lower right",
                borderpad=0.5,
                handlelength=1.0,
                bbox_to_anchor=(1.9, 0.0),
            )
            ax.text(
                1.1,
                -0.1,
                r"No. Spectators per Exp = %s" % (data[1, 8] + data[1, 9]),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="bottom",
            )

            # ax.set_ylim(bottom=0, top=None)
            ax.set_xlim(left=0, right=None)
            pdf.savefig(fig, bbox_inches="tight")

            fig, ax = plt.subplots()
            plt.ylabel(r"Entropy (bits)")
            plt.xlabel(r"\% Shared spectators")
            plt.title(
                get_title(data, numTargets),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )

            dir = folder[-1]
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            # print(result_path)
            # print(result_path)
            # print(result_path)
            data = genfromtxt(
                result_path,
                delimiter=",",
            )

            # lam = str(ls_experiments[0]).split("_")
            # lam = float(lam[-1])
            (l2,) = plt.plot(
                data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                data[:, 13],
                marker="",
                color=color_arr[0],
                linestyle="-",
                label=r"$h(%s \mid O_1, O_2)$" % target_string,
            )

            (l2,) = plt.plot(
                data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                data[:, 15],
                marker="",
                color=color_arr[2],
                linestyle="dotted",
                label=r"$h(%s \mid O_1)$" % target_string,
            )
            (l2,) = plt.plot(
                data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                data[:, 14],
                marker="",
                color=color_arr[1],
                linestyle="--",
                label=r"$h(%s )$" % target_string,
            )
            ax.text(
                -0.11,
                1.1,
                get_dist(experimentName),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            )
            ax.legend(
                loc="lower right",
                borderpad=0.5,
                handlelength=1.0,
                bbox_to_anchor=(1.6, 0.0),
            )
            ax.text(
                1.1,
                -0.1,
                r"No. Spectators per Exp = %s" % (data[1, 8] + data[1, 9]),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="bottom",
            )
            ax.text(
                0.3,
                0.07,
                r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
            )

            # ax.text(
            #     1.11,
            #     1.1,
            #     r"\begin{eqnarray*} h(X_{T} \mid O_1) - h(X_{T} \mid O_1, O_2)\end{eqnarray*}",
            #     # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            #     transform=ax.transAxes,
            #     fontsize=14,
            #     verticalalignment="top",
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            # )

            # ax.set_ylim(bottom=0, top=None)
            ax.set_xlim(left=0, right=None)
            pdf.savefig(fig, bbox_inches="tight")

            fig, ax = plt.subplots()
            plt.ylabel(r"Entropy (bits)")
            plt.xlabel(r"\% Shared spectators")
            plt.title(
                get_title(data, numTargets),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )

            dir = folder[0]
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            # print(result_path)
            # print(result_path)
            # print(result_path)
            data = genfromtxt(
                result_path,
                delimiter=",",
            )

            # lam = str(ls_experiments[0]).split("_")
            # lam = float(lam[-1])
            (l2,) = plt.plot(
                data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                data[:, 13],
                marker="",
                color=color_arr[0],
                linestyle="-",
                label=r"$h(%s \mid O_1, O_2)$" % target_string,
            )

            (l2,) = plt.plot(
                data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                data[:, 15],
                marker="",
                color=color_arr[2],
                linestyle="dotted",
                label=r"$h(%s \mid O_1)$" % target_string,
            )
            (l2,) = plt.plot(
                data[:, 8] / (data[:, 8] + data[:, 9] + data[:, 10]),
                data[:, 14],
                marker="",
                color=color_arr[1],
                linestyle="--",
                label=r"$h(%s )$" % target_string,
            )
            ax.text(
                -0.11,
                1.1,
                get_dist(experimentName),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            )
            ax.legend(
                loc="lower right",
                borderpad=0.5,
                handlelength=1.0,
                bbox_to_anchor=(1.6, 0.0),
            )
            ax.text(
                1.1,
                -0.1,
                r"No. Spectators per Exp = %s" % (data[1, 8] + data[1, 9]),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="bottom",
            )
            ax.text(
                0.3,
                0.07,
                r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
            )

            # ax.text(
            #     1.11,
            #     1.1,
            #     r"\begin{eqnarray*} h(X_{T} \mid O_1) - h(X_{T} \mid O_1, O_2)\end{eqnarray*}",
            #     # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            #     transform=ax.transAxes,
            #     fontsize=14,
            #     verticalalignment="top",
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            # )

            # ax.set_ylim(bottom=0, top=None)
            ax.set_xlim(left=0, right=None)
            pdf.savefig(fig, bbox_inches="tight")

            plt.clf()

    plt.close("all")

    return 0


def plot_1_target():
    experimentName = "joint_sum_normal"

    cmd = "ls  ../output/" + experimentName + "/"
    files = os.listdir("../output/" + experimentName + "/")
    files = [item for item in files if ".pdf" not in item]
    # print(files, experimentName)

    ctr = 0

    exps = [
        "no_common_spec",
        "one_common_spec",
        "high_target_sigma",
        "high_target_nonlinear_sigma2",
        "high_target_nonlinear_sigma3",
        "low_target_sigma",
    ]

    for ex in exps:
        filtered = [k for k in files if ex in k]
        main1(filtered, ex, experimentName, 1)

    spec_sizes = ["10_", "50_", "100_"]
    exnames = [
        "vary_spec_percent_same_target",
        "vary_spec_percent_high_target",
        "vary_spec_percent_low_target",
    ]

    for ex in exnames:
        # filtered = [k for k in files if sz + ex in k]
        main2(files, ex, spec_sizes, experimentName, 1)

    # filtered_0 = [k for k in files if exps[0] in k]
    # filtered_1 = [k for k in files if exps[1] in k]
    # plot_0_vs_1_shared_spec(filtered_0, filtered_1, )
    # for ex in exps:
    #     filtered = [k for k in files if ex in k]
    #     main2(filtered, ex, experimentName)


if __name__ == "__main__":
    plot_1_target()
