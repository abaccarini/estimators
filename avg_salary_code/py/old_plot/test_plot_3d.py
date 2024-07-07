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
from plot_joint_normal import get_title
from plot_joint_normal import get_dist
from test import H_T
from natsort import natsorted
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.interpolate import griddata


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

mpl.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb,xcolor}",
    }
)


logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager


def get_dist(exp_name):
    exp_name = exp_name.split("_")
    if "normal" in exp_name:
        return r"$X\sim  \mathcal{N} (\mu, \sigma^2)$"
    elif "poisson" in exp_name:
        return r"$X\sim  \text{Pois} (\lambda = N/2)$"
    else:
        print("DISTRIBUTION NOT FOUND", exp_name)
        return 0


def get_title(data):

    # exam_name = 'num_shared'
    # exam_index = np.where(data[0] == exam_name)
    # print(data[1])
    # print(exam_index)
    # print(int(data[8, 1]))
    # dist = ""
    # if data[0, 0] == "target.mu":
    dist = r"$X\sim  \mathcal{N} (\mu, \sigma^2)$"

    if data[1, 8] == 0 and data[2, 8] == 0:
        return (
            r"\begin{align*}O_1 = X_{T} + X_{S_1} \\ O_2 =X_{T} + X_{S_2}\end{align*}"
        )
    else:
        return r"\begin{align*}O_1 &=X_{T} +X_S+ X_{S_1}\\O_2 &=X_{T}+X_S + X_{S_2}\end{align*}"


def main_3d_plot(folder, experiment, experimentName):
    dir_path = "../output/" + experimentName + "_plots/"
    out_path = dir_path + experiment + "_normal.pdf"
    print(out_path)
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_path)

    dir_list = [-1, 0]  # two samples

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")
    with PdfPages(out_path) as pdf:

        print(folder)
        # folder.sort(key=lambda x: (float(x), len(x)))
        # l = sorted(folder, key=lambda k: [*map(int, k.split('.'))])
        max_shared = 0
        max_unique = 0
        folder = natsorted(folder)
        folder.reverse()
        # print(l)
        for dir_l in dir_list:

            dir = folder[dir_l]
            # print(dir)
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            data = data[1:]  # removing
            x = np.array(data[:, 8])
            y = np.array(data[:, 9])
            z = np.array(data[:, 13])

            max_shared = int(x[-1] - x[0] + 1)
            max_unique = int(y[-1] - y[0] + 1)
            np.set_printoptions(linewidth=np.inf)

            # print(x[0:max_shared])
            # print(x[max_shared:2*max_shared])
            # print("max_shared", max_shared)
            # print("max_unique", max_unique)

            # print(data_0_shared)
            # print(data_1_shared)

            trim_data = []

            upper_bound_3d = 25
            for dat in data:
                if dat[8] < upper_bound_3d and dat[9] < upper_bound_3d:
                    # print(dat)
                    # trim_data = np.concatenate(trim_data, dat)
                    trim_data.append(dat)
            trim_data = np.array(trim_data)
            # print(trim_data)

            x = np.array(trim_data[:, 8])
            y = np.array(trim_data[:, 9])
            z = np.array(trim_data[:, 13])

            # matplotlib
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            ax = fig.add_subplot(111, projection="3d")
            fig.add_axes(ax)

            surf = ax.plot_trisurf(
                x,
                y,
                z,
                linewidth=0.2,
                color="r",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$h(X_T \mid O_1, O_2)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            surf = ax.plot_trisurf(
                x,
                y,
                np.array(trim_data[:, 15]),
                linewidth=0.2,
                color="g",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$h(X_T \mid O_1)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            surf = ax.plot_trisurf(
                x,
                y,
                np.array(trim_data[:, 14]),
                linewidth=0.2,
                color="b",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$h(X_T)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            plt.title(
                get_title(data),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )

            # ax.set_title("Shared vs. Unique Spectators")

            ax.set_title(
                r"\begin{align*}O_1 &= X_{T}  +X_S + X_{S_1},\\O_2 &= X_{T}  +X_S+ X_{S_2}\end{align*}",
                y=1.1,
            )
            ax.set_xlabel(r"Shared Spectators per Exp")
            ax.set_ylabel(r"Unique Spectators per Exp")
            ax.set_zlabel(r"Entropy (bits)")
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.7, 1.1),
            )
            ax.text2D(
                1.1,
                0.07,
                r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )

            # ax.text(
            #     -0.15,
            #     1.1,
            #     get_dist(experimentName),
            #     transform=ax.transAxes,
            #     fontsize=14,
            #     verticalalignment="top",
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            # )

            # plt.show()

            pdf.savefig(fig, bbox_inches="tight")

            # NEW FIGURE

            dir = folder[dir_l]
            # print(dir)
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            data = data[1:]  # removing titles
            x = np.array(data[:, 8])
            y = np.array(data[:, 9])
            z = np.array(data[:, 13])

            max_shared = int(x[-1] - x[0] + 1)
            max_unique = int(y[-1] - y[0] + 1)
            np.set_printoptions(linewidth=np.inf)

            # print(x[0:max_shared])
            # print(x[max_shared:2*max_shared])
            # print("max_shared", max_shared)
            # print("max_unique", max_unique)

            # print(data_0_shared)
            # print(data_1_shared)

            trim_data = []

            upper_bound_3d = 25
            for dat in data:
                if dat[8] < upper_bound_3d and dat[9] < upper_bound_3d:
                    # print(dat)
                    # trim_data = np.concatenate(trim_data, dat)
                    trim_data.append(dat)
            trim_data = np.array(trim_data)
            # print(trim_data)

            x = np.array(trim_data[:, 8])
            y = np.array(trim_data[:, 9])
            z = np.array(trim_data[:, 13])

            # matplotlib
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            ax = fig.add_subplot(111, projection="3d")
            fig.add_axes(ax)

            surf = ax.plot_trisurf(
                x,
                y,
                [a - b for a, b in zip(trim_data[:, 15], trim_data[:, 13])],
                linewidth=0.2,
                color="r",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$ h(X_{T} \mid O_1) - h(X_{T} \mid O_1, O_2)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            surf = ax.plot_trisurf(
                x,
                y,
                [a - b for a, b in zip(trim_data[:, 14], trim_data[:, 13])],
                linewidth=0.2,
                color="g",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$ h(X_{T}) - h(X_{T} \mid O_1, O_2)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            surf = ax.plot_trisurf(
                x,
                y,
                [a - b for a, b in zip(trim_data[:, 14], trim_data[:, 15])],
                linewidth=0.2,
                color="b",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$ h(X_{T} ) - h(X_{T} \mid O_1)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            # surf = ax.plot_trisurf(
            #     x,
            #     y,
            #     np.array(trim_data[:, 15]),
            #     linewidth=0.2,
            #     color="g",
            #     antialiased=True,
            #     shade=False,
            #     alpha=0.4,
            #     label=r"$h(X_T \mid O_1)$",
            # )
            # surf._facecolors2d = surf._facecolor3d
            # surf._edgecolors2d = surf._edgecolor3d
            # surf = ax.plot_trisurf(
            #     x,
            #     y,
            #     np.array(trim_data[:, 14]),
            #     linewidth=0.2,
            #     color="b",
            #     antialiased=True,
            #     shade=False,
            #     alpha=0.4,
            #     label=r"$h(X_T)$",
            # )
            # surf._facecolors2d = surf._facecolor3d
            # surf._edgecolors2d = surf._edgecolor3d
            plt.title(
                get_title(data),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )

            # ax.set_title("Shared vs. Unique Spectators")

            ax.set_title(
                r"\begin{align*}O_1 &= X_{T}  +X_S + X_{S_1},\\O_2 &= X_{T}  +X_S+ X_{S_2}\end{align*}",
                y=1.1,
            )
            ax.set_xlabel(r"Shared Spectators per Exp")
            ax.set_ylabel(r"Unique Spectators per Exp")
            ax.set_zlabel(r"Entropy (bits)")
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.7, 1.1),
            )
            ax.text2D(
                1.1,
                0.07,
                r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )
            # ax.text2D(
            #     1.11,
            #     1.1,
            # r"\begin{eqnarray*} h(X_{T} \mid O_1) - h(X_{T} \mid O_1, O_2)\end{eqnarray*}",
            #     # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            #     transform=ax.transAxes,
            #     fontsize=14,
            #     verticalalignment="top",
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            # )

            # ax.text(
            #     -0.15,
            #     1.1,
            #     get_dist(experimentName),
            #     transform=ax.transAxes,
            #     fontsize=14,
            #     verticalalignment="top",
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
            # )

            # plt.show()

            pdf.savefig(fig, bbox_inches="tight")

            # NEW FIGURE

            dir = folder[dir_l]
            # print(dir)
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            data = data[1:]  # removing titles
            x = np.array(data[:, 8])
            y = np.array(data[:, 9])
            z = np.array(data[:, 13])

            max_shared = int(x[-1] - x[0] + 1)
            max_unique = int(y[-1] - y[0] + 1)
            np.set_printoptions(linewidth=np.inf)

            # print(x[0:max_shared])
            # print(x[max_shared:2*max_shared])
            # print("max_shared", max_shared)
            # print("max_unique", max_unique)

            # print(data_0_shared)
            # print(data_1_shared)

            trim_data = []

            upper_bound_3d = 25
            for dat in data:
                if dat[8] < upper_bound_3d and dat[9] < upper_bound_3d:
                    # print(dat)
                    # trim_data = np.concatenate(trim_data, dat)
                    trim_data.append(dat)
            trim_data = np.array(trim_data)
            # print(trim_data)

            x = np.array(trim_data[:, 8])
            y = np.array(trim_data[:, 9])
            z = np.array(trim_data[:, 13])

            # matplotlib
            fig = plt.figure()
            ax = Axes3D(fig, auto_add_to_figure=False)
            ax = fig.add_subplot(111, projection="3d")
            fig.add_axes(ax)

            surf = ax.plot_trisurf(
                x,
                y,
                [
                    ((a - b) / a) * 100.0
                    for a, b in zip(trim_data[:, 15], trim_data[:, 13])
                ],
                linewidth=0.2,
                color="r",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$ h(X_{T} \mid O_1) - h(X_{T} \mid O_1, O_2)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            surf = ax.plot_trisurf(
                x,
                y,
                [
                    ((a - b) / a) * 100.0
                    for a, b in zip(trim_data[:, 14], trim_data[:, 13])
                ],
                linewidth=0.2,
                color="g",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$ h(X_{T}) - h(X_{T} \mid O_1, O_2)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            surf = ax.plot_trisurf(
                x,
                y,
                [
                    ((a - b) / a) * 100.0
                    for a, b in zip(trim_data[:, 14], trim_data[:, 15])
                ],
                linewidth=0.2,
                color="b",
                antialiased=True,
                shade=False,
                alpha=0.4,
                label=r"$ h(X_{T} ) - h(X_{T} \mid O_1)$",
            )
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            plt.title(
                get_title(data),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
                y=1.1,
            )

            # ax.set_title("Shared vs. Unique Spectators")

            ax.set_title(
                r"\begin{align*}O_1 &= X_{T}  +X_S + X_{S_1},\\O_2 &= X_{T}  +X_S+ X_{S_2}\end{align*}",
                y=1.1,
            )
            ax.set_xlabel(r"Shared Spectators per Exp")
            ax.set_ylabel(r"Unique Spectators per Exp")
            ax.set_zlabel(r"Percent Decrese")
            ax.legend(
                loc="upper right",
                bbox_to_anchor=(1.7, 1.1),
            )
            ax.text2D(
                1.1,
                0.07,
                r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
                # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
            )

            pdf.savefig(fig, bbox_inches="tight")

            # NEW FIGURE

        for dir_l in dir_list:

            fig, ax = plt.subplots()
            plt.ylabel(r"Entropy (bits)")
            plt.xlabel(r"Unique Spectators per Exp")
            dir = folder[dir_l]

            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            data = data[1:]  # removing 1st row
            data_0_shared = data[:max_unique]
            data_1_shared = data[max_unique : 2 * max_unique]

            plt.title(
                "Zero vs. One Shared Spectator"
                # r"\begin{align*}O_1 =X_{T} + X_{S_1},\quad O_1' =X_{T} +X_S+ X_{S_1}\\O_2 =X_{T}+ X_{S_2},\quad O_2' =X_{T}+X_S + X_{S_2}\end{align*}",
                # y=1.1,
            )

            ax.text(
                1.1,
                1.0,
                r"\begin{align*}O_1 &=X_{T} + X_{S_1}\\ O_1' &=X_{T} +X_S+ X_{S_1}\\O_2 &=X_{T}+ X_{S_2}\\ O_2' &=X_{T}+X_S + X_{S_2}\end{align*}",
                transform=ax.transAxes,
                # fontsize=14,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
            )

            (l2,) = plt.plot(
                data_0_shared[:, 9],
                data_0_shared[:, 13],
                marker="",
                color=color_arr[0],
                linestyle="-",
                label=r"$h(X_T \mid O_1, O_2)$",
            )

            (l2,) = plt.plot(
                data_0_shared[:, 9],
                data_0_shared[:, 15],
                marker="",
                color=color_arr[0],
                linestyle="dotted",
                label=r"$h(X_T \mid O_1)$",
            )

            (l2,) = plt.plot(
                data_1_shared[:, 9],
                data_1_shared[:, 13],
                marker="",
                color=color_arr[1],
                linestyle="-",
                label=r"$h(X_T \mid O_1', O_2')$",
            )

            (l2,) = plt.plot(
                data_1_shared[:, 9],
                data_1_shared[:, 15],
                marker="",
                color=color_arr[1],
                linestyle="dotted",
                label=r"$h(X_T \mid O_1')$",
            )
            (l2,) = plt.plot(
                data_1_shared[:, 9],
                data_1_shared[:, 14],
                marker="",
                color=color_arr[2],
                linestyle="-",
                # label=r"$\sigma^2 = %s, h(X_T)$" % data[1, 1],
                label=r"$h(X_T)$",
            )

            ax.text(
                -0.15,
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
                bbox_to_anchor=(1.4, 0.0),
            )

            ax.text(
                0.2,
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
        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"Unique Spectators per Exp")
        plt.title(
            r"\begin{align*}O_1 =X_{T} + X_{S_1},\quad O_1' =X_{T} +X_S+ X_{S_1}\\O_2 =X_{T}+ X_{S_2},\quad O_2' =X_{T}+X_S + X_{S_2}\end{align*}",
            y=1.1,
        )
        for idx, dir in enumerate(folder[2:]):  # only doing half
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            # print(result_path)
            # print(result_path)
            # print(result_path)
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            data = data[1:]  # removing 1st row
            data_0_shared = data[:max_unique]
            data_1_shared = data[max_unique : 2 * max_unique]

            (l2,) = plt.plot(
                data_0_shared[:50, 9],
                data_0_shared[:50, 13],
                marker="",
                color=color_arr[idx],
                linestyle="-",
                label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
            )

            (l2,) = plt.plot(
                data_0_shared[:50, 9],
                data_0_shared[:50, 15],
                marker="",
                color=color_arr[idx],
                linestyle="dotted",
            )

            (l2,) = plt.plot(
                data_1_shared[:50, 9],
                data_1_shared[:50, 13],
                marker="",
                alpha=0.5,
                color=color_arr[idx],
                linestyle="-",
            )

            (l2,) = plt.plot(
                data_1_shared[:50, 9],
                data_1_shared[:50, 15],
                marker="",
                alpha=0.5,
                color=color_arr[idx],
                linestyle="dotted",
            )
            (l2,) = plt.plot(
                data_1_shared[:50, 9],
                data_1_shared[:50, 14],
                marker="",
                color=color_arr[idx],
                linestyle="--",
            )
        ax.text(
            1.11,
            1.1,
            r"\begin{eqnarray*} - - &=& h(X_T) \\ \cdots &=& h(X_T\mid O_1)  \\ \text{---} &=& h(X_T\mid O_1, O_2) \end{eqnarray*}",
            # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )

        ax.text(
            1.11,
            0.8,
            r"{\begin{eqnarray*}- - &=& h(X_T) \\ \cdots &=& h(X_T\mid O_1')  \\ \text{---} &=& h(X_T\mid O_1', O_2') \end{eqnarray*}}",
            # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
            transform=ax.transAxes,
            fontsize=14,
            alpha=0.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.61, -0.2),
        )
        ax.text(
            -0.15,
            1.1,
            get_dist(experimentName),
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
        )

        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)

        pdf.savefig(fig, bbox_inches="tight")
        plt.clf()
    plt.close("all")


def view_3d(folder, experiment, experimentName, exp_num):
    folder = natsorted(folder)
    folder.reverse()
    # print(l)
    dir = folder[exp_num]
    print(dir)
    result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
    data = genfromtxt(
        result_path,
        delimiter=",",
    )
    data = data[1:]  # removing
    x = np.array(data[:, 8])
    y = np.array(data[:, 9])
    z = np.array(data[:, 13])

    max_shared = int(x[-1] - x[0] + 1)
    max_unique = int(y[-1] - y[0] + 1)
    np.set_printoptions(linewidth=np.inf)

    # print(x[0:max_shared])
    # print(x[max_shared:2*max_shared])
    # print("max_shared", max_shared)
    # print("max_unique", max_unique)

    # print(data_0_shared)
    # print(data_1_shared)

    trim_data = []

    upper_bound_3d = 25
    for dat in data:
        if dat[8] < upper_bound_3d and dat[9] < upper_bound_3d:
            # print(dat)
            # trim_data = np.concatenate(trim_data, dat)
            trim_data.append(dat)
    trim_data = np.array(trim_data)
    # print(trim_data)

    x = np.array(trim_data[:, 8])
    y = np.array(trim_data[:, 9])
    z = np.array(trim_data[:, 13])

    # matplotlib
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)

    surf = ax.plot_trisurf(
        x,
        y,
        z,
        linewidth=0.2,
        color="r",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$h(X_T \mid O_1, O_2)$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_trisurf(
        x,
        y,
        np.array(trim_data[:, 15]),
        linewidth=0.2,
        color="g",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$h(X_T \mid O_1)$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    surf = ax.plot_trisurf(
        x,
        y,
        np.array(trim_data[:, 14]),
        linewidth=0.2,
        color="b",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$h(X_T)$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    plt.title(
        get_title(data),
        # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
        y=1.1,
    )

    # ax.set_title("Shared vs. Unique Spectators")

    ax.set_title(
        r"\begin{align*}O_1 &= X_{T}  +X_S + X_{S_1},\\O_2 &= X_{T}  +X_S+ X_{S_2}\end{align*}",
        y=1.1,
    )
    ax.set_xlabel(r"Shared Spectators per Exp")
    ax.set_ylabel(r"Unique Spectators per Exp")
    ax.set_zlabel(r"Entropy (bits)")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.7, 1.1),
    )
    ax.text2D(
        1.1,
        0.07,
        r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
        % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
        # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    # plt.show()

    dir = folder[exp_num]
    print(dir)
    result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
    data = genfromtxt(
        result_path,
        delimiter=",",
    )
    data = data[1:]  # removing titles
    x = np.array(data[:, 8])
    y = np.array(data[:, 9])
    z = np.array(data[:, 13])

    max_shared = int(x[-1] - x[0] + 1)
    max_unique = int(y[-1] - y[0] + 1)
    np.set_printoptions(linewidth=np.inf)

    # print(x[0:max_shared])
    # print(x[max_shared:2*max_shared])
    # print("max_shared", max_shared)
    # print("max_unique", max_unique)

    # print(data_0_shared)
    # print(data_1_shared)

    trim_data = []

    upper_bound_3d = 25
    for dat in data:
        if dat[8] < upper_bound_3d and dat[9] < upper_bound_3d:
            # print(dat)
            # trim_data = np.concatenate(trim_data, dat)
            trim_data.append(dat)
    trim_data = np.array(trim_data)
    # print(trim_data)

    x = np.array(trim_data[:, 8])
    y = np.array(trim_data[:, 9])
    z = np.array(trim_data[:, 13])

    # matplotlib
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)

    surf = ax.plot_trisurf(
        x,
        y,
        [a - b for a, b in zip(trim_data[:, 15], trim_data[:, 13])],
        linewidth=0.2,
        color="r",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$ h(X_{T} \mid O_1) - h(X_{T} \mid O_1, O_2)$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_trisurf(
        x,
        y,
        [a - b for a, b in zip(trim_data[:, 14], trim_data[:, 13])],
        linewidth=0.2,
        color="g",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$ h(X_{T}) - h(X_{T} \mid O_1, O_2)$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_trisurf(
        x,
        y,
        [a - b for a, b in zip(trim_data[:, 14], trim_data[:, 15])],
        linewidth=0.2,
        color="b",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$ h(X_{T} ) - h(X_{T} \mid O_1)$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    plt.title(
        get_title(data),
        # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
        y=1.1,
    )

    # ax.set_title("Shared vs. Unique Spectators")

    ax.set_title(
        r"\begin{align*}O_1 &= X_{T}  +X_S + X_{S_1},\\O_2 &= X_{T}  +X_S+ X_{S_2}\end{align*}",
        y=1.1,
    )
    ax.set_xlabel(r"Shared Spectators per Exp")
    ax.set_ylabel(r"Unique Spectators per Exp")
    ax.set_zlabel(r"Entropy (bits)")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.7, 1.1),
    )
    ax.text2D(
        1.1,
        0.07,
        r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
        % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
        # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    dir = folder[exp_num]
    print(dir)
    result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
    data = genfromtxt(
        result_path,
        delimiter=",",
    )
    data = data[1:]  # removing titles
    x = np.array(data[:, 8])
    y = np.array(data[:, 9])
    z = np.array(data[:, 13])

    max_shared = int(x[-1] - x[0] + 1)
    max_unique = int(y[-1] - y[0] + 1)
    np.set_printoptions(linewidth=np.inf)

    # print(x[0:max_shared])
    # print(x[max_shared:2*max_shared])
    # print("max_shared", max_shared)
    # print("max_unique", max_unique)

    # print(data_0_shared)
    # print(data_1_shared)

    trim_data = []

    upper_bound_3d = 25
    for dat in data:
        if dat[8] < upper_bound_3d and dat[9] < upper_bound_3d:
            # print(dat)
            # trim_data = np.concatenate(trim_data, dat)
            trim_data.append(dat)
    trim_data = np.array(trim_data)
    # print(trim_data)

    x = np.array(trim_data[:, 8])
    y = np.array(trim_data[:, 9])
    z = np.array(trim_data[:, 13])

    # matplotlib
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)

    surf = ax.plot_trisurf(
        x,
        y,
        [((a - b) / a) * 100.0 for a, b in zip(trim_data[:, 15], trim_data[:, 13])],
        linewidth=0.2,
        color="r",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$  \frac{h(X_T\mid O_1) - h(X_T \mid O_1, O_2)}{h(X_T\mid O_1)} $",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_trisurf(
        x,
        y,
        [((a - b) / a) * 100.0 for a, b in zip(trim_data[:, 14], trim_data[:, 13])],
        linewidth=0.2,
        color="g",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$\frac{h(X_T) - h(X_T \mid O_1, O_2)}{h(X_T)}$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    surf = ax.plot_trisurf(
        x,
        y,
        [((a - b) / a) * 100.0 for a, b in zip(trim_data[:, 14], trim_data[:, 15])],
        linewidth=0.2,
        color="b",
        antialiased=True,
        shade=False,
        alpha=0.4,
        label=r"$\frac{h(X_T) -  h(X_T\mid O_1)}{h(X_T)}$",
    )
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d

    plt.title(
        get_title(data),
        # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
        y=1.1,
    )

    # ax.set_title("Shared vs. Unique Spectators")

    ax.set_title(
        r"\begin{align*}O_1 &= X_{T}  +X_S + X_{S_1},\\O_2 &= X_{T}  +X_S+ X_{S_2}\end{align*}",
        y=1.1,
    )
    ax.set_xlabel(r"Shared Spectators per Exp")
    ax.set_ylabel(r"Unique Spectators per Exp")
    ax.set_zlabel(r"Percent Decrease")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.7, 1.1),
    )
    ax.text2D(
        1.1,
        0.07,
        r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
        % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
        # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    plt.show()

    plt.clf()
    plt.close("all")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        experimentName = "joint_sum_normal"

        cmd = "ls  ../output/" + experimentName + "/"
        files = os.listdir("../output/" + experimentName + "/")
        files = [item for item in files if ".pdf" not in item]
        # print(files, experimentName)

        ctr = 0

        exps = [
            "vary_shared_and_unique",
            "vary_low_target_shared_and_unique",
            "vary_high_target_shared_and_unique",
        ]

        for ex in exps:
            filtered = [k for k in files if ex in k]
            main_3d_plot(filtered, ex, experimentName)

        ex_name = "vary_high_target_shared_and_unique"
        filtered = [k for k in files if ex_name in k]

        # view_3d(filtered, ex_name, experimentName, -1)
    else:
        exit(0)
