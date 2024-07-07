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

from plot_joint_normal import main2
from plot_joint_normal import get_dist

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
        "font.size": 8,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb}",
    }
)

rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 16})
rc("text", usetex=True)


def get_title(data):

    if data[1, 10] == 0 and data[2, 10] == 0:
        return r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T} + X_{S_2}\\O_3 &=X_{T}  + X_{S_3}\end{eqnarray*}"
    else:
        return (
            r"\begin{eqnarray*}O_1 &=X_{T_1} +X_S+ X_{S_1}\\O_2 &=X_{T} + X_S+ X_{S_2}\\O_3 &=X_{T} +X_S + X_{S_3}\end{eqnarray*}",
        )


def main1(folder, experiment, experimentName, numTargets):
    # for file in files:
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
        folder = natsorted(folder)
        folder.reverse()
        # print(l)

        # plt.figure(figsize=(1,15))
        # plt.rcParams["figure.figsize"] = [6, 6]
        fig, ax = plt.subplots()

        plt.ylabel(r"Entropy (bits)")
        plt.xlabel(r"Unique Spectators per Exp")

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
            plt.title(
                get_title(data),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} +X_S+ X_{S_1}\\O_2 &=X_{T} + X_S+ X_{S_2}\\O_3 &=X_{T} +X_s + X_{S_3}\end{eqnarray*}",
                # y=1.1,
            )

            # lam = str(ls_experiments[0]).split("_")
            # lam = float(lam[-1])
            (l2,) = plt.plot(
                data[:, 11],
                data[:, 14],
                marker="",
                color=color_arr[idx],
                linestyle="-",
                label=r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
            )

            (l2,) = plt.plot(
                data[:, 11],
                data[:, 15],
                marker="",
                color=color_arr[idx],
                linestyle="--",
                # label=r"$h(X_T)$" ,
            )
            (l2,) = plt.plot(
                data[:, 11],
                data[:, 16],
                marker="",
                color=color_arr[idx],
                linestyle="dotted",
                # label=r"$h(X_T)$" ,
            )
            (l2,) = plt.plot(
                data[:, 11],
                data[:, 17],
                marker="",
                color=color_arr[idx],
                linestyle="dashdot",
                # label=r"$h(X_T)$" ,
            )

        #  \begin{eqnarray*}
        # - - = h(X_{T_1}) \\ \cdots = h(X_{T_1}\mid O_1)  \\ -- = h(X_{T_1}\mid O_1, O_2)
        #  \end{eqnarray*}

        ax.text(
            1.11,
            1.4,
            r"\begin{eqnarray*} \text{---}  &=& h(X_{T_1}) \\ - - &=& h(X_{T_1}\mid O_1)  \\ \dots &=& h(X_{T_1}\mid O_1, O_2)  \\ {\cdot\text{-}\cdot} &=& h(X_{T_1}\mid O_1, O_2, O_3)\end{eqnarray*}",
            # - - $= h(X_{T_1})$ \\ $\cdots$ $= h(X_{T_1}\mid O_1)$  \\ -- $= h(X_{T_1}\mid O_1, O_2)$'
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
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
        upper_bound = 8
        # ax.set_ylim(bottom=0, top=None)
        ax.set_xlim(left=0, right=None)
        pdf.savefig(fig, bbox_inches="tight")

        for dir_l in dir_list:

            fig, ax = plt.subplots()
            plt.ylabel(r"Entropy (bits)")
            plt.xlabel(r"Unique Spectators per Exp")

            dir = folder[dir_l]
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            # print(result_path)
            # print(result_path)
            # print(result_path)
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            plt.title(
                get_title(data),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
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
            (l2,) = plt.plot(
                data[:, 11],
                data[:, 14],
                marker="",
                color=color_arr[0],
                linestyle="-",
                label=r"$h(X_{T} )$",
            )
            (l2,) = plt.plot(
                data[:, 11],
                data[:, 15],
                marker="",
                color=color_arr[1],
                linestyle="-",
                label=r"$h(X_{T} \mid O_1)$",
            )

            (l2,) = plt.plot(
                data[:, 11],
                data[:, 16],
                marker="",
                color=color_arr[2],
                linestyle="dotted",
                label=r"$h(X_{T} \mid O_1, O_2)$",
            )
            (l2,) = plt.plot(
                data[:, 11],
                data[:, 17],
                marker="",
                color=color_arr[3],
                linestyle="--",
                label=r"$h(X_{T} \mid O_1, O_2, O_3)$",
            )

            ax.legend(
                loc="lower right",
                borderpad=0.5,
                handlelength=1.0,
                bbox_to_anchor=(1.6, 0.0),
            )

            ax.text(
                0.2,
                0.07,
                r"$\sigma_{T, S, S_1, S_2, S_3}^2 = \{%s, %s, %s, %s , %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7], data[1, 9]),
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

            dir = folder[dir_l]
            result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
            # print(result_path)
            # print(result_path)
            # print(result_path)
            data = genfromtxt(
                result_path,
                delimiter=",",
            )
            plt.title(
                get_title(data),
                # r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
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
            (l2,) = plt.plot(
                data[:, 11],
                [a - b for a, b in zip(data[:, 14], data[:, 15])],
                marker="",
                color=color_arr[0],
                linestyle="--",
                label=r"$h(X_{T}) - h(X_{T} \mid O_1)$",
            )
            (l2,) = plt.plot(
                data[:, 11],
                [a - b for a, b in zip(data[:, 14], data[:, 16])],
                marker="",
                color=color_arr[1],
                linestyle="-",
                label=r"$h(X_{T}) - h(X_{T} \mid O_1, O_2)$",
            )

            (l2,) = plt.plot(
                data[:, 11],
                [a - b for a, b in zip(data[:, 14], data[:, 17])],
                marker="",
                color=color_arr[2],
                linestyle="dotted",
                label=r"$h(X_{T}) - h(X_{T} \mid O_1, O_2, O_3)$",
            )

            (l2,) = plt.plot(
                data[:, 11],
                [a - b for a, b in zip(data[:, 15], data[:, 16])],
                marker="",
                color=color_arr[3],
                linestyle="--",
                label=r"$h(X_{T} \mid O_1)- h(X_{T} \mid O_1, O_2)$",
            )
            (l2,) = plt.plot(
                data[:, 11],
                [a - b for a, b in zip(data[:, 16], data[:, 17])],
                marker="",
                color=color_arr[4],
                linestyle="-",
                label=r"$h(X_{T} \mid O_1, O_2)- h(X_{T} \mid O_1, O_2, O_3)$",
            )

            ax.legend(
                loc="lower right",
                borderpad=0.5,
                handlelength=1.0,
                bbox_to_anchor=(1.9, 0.0),
            )

            ax.text(
                0.2,
                0.97,
                r"$\sigma_{T, S, S_1, S_2, S_3}^2 = \{%s, %s, %s, %s , %s \}$"
                % (data[1, 1], data[1, 3], data[1, 5], data[1, 7], data[1, 9]),
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


def plot_one_vs_two_vs_three_exp(
    all_folders, spec_sizes, experiment_1, experiment_2, experiment_3, experimentName
):
    dir_path = "../../output/" + experimentName + "_plots/"
    out_path = dir_path + experiment_1 + "_1_3_vs_2_3_vs_3_3.pdf"
    print(out_path)
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_path)
    dir_list = [4]  # two samples

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")

    with PdfPages(out_path) as pdf:
        for sz in spec_sizes:
            folder_1 = [k for k in all_folders if sz + experiment_1 in k]
            folder_2 = [k for k in all_folders if sz + experiment_2 in k]
            folder_3 = [k for k in all_folders if sz + experiment_3 in k]
            folder_1 = natsorted(folder_1)
            # folder_1.reverse()
            print(folder_1)

            folder_2 = natsorted(folder_2)
            # folder_2.reverse()

            folder_3 = natsorted(folder_3)
            # folder_3.reverse()

            for dir_l in dir_list:
                fig, ax = plt.subplots()
                plt.ylabel(r"Entropy (bits)")
                plt.xlabel(r"Fraction of shared spectators")

                dir_1 = folder_1[dir_l]
                result_path_1 = (
                    "../../output/" + experimentName + "/" + dir_1 + "/results.csv"
                )

                dir_2 = folder_2[dir_l]
                result_path_2 = (
                    "../../output/" + experimentName + "/" + dir_2 + "/results.csv"
                )
                dir_3 = folder_3[dir_l]
                result_path_3 = (
                    "../../output/" + experimentName + "/" + dir_3 + "/results.csv"
                )

                data_1 = genfromtxt(
                    result_path_1,
                    delimiter=",",
                )

                data_2 = genfromtxt(
                    result_path_2,
                    delimiter=",",
                )
                data_3 = genfromtxt(
                    result_path_3,
                    delimiter=",",
                )

                plt.title(
                    r"\begin{gather*}O_1 =X_{T} +X_S + X_{S_1}\\O_2 =X_{T}+ X_S+ X_{S_2}, \quad O_2' =X_S + X_{S_2}\\O_3 =X_{T}+ X_S+ X_{S_3}, \quad O_3' =X_S + X_{S_3}\end{gather*}",
                )

                ax.text(
                    -0.25,
                    1.1,
                    get_dist(experimentName),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.2),
                )

                color_iter = iter(color_arr)

                (l2,) = plt.plot(
                    data_1[:, 10] / (data_1[:, 10] + data_1[:, 11]),
                    data_1[:, 14],
                    # marker="",
                    color=next(color_iter),
                    linestyle="--",
                    # label=r"$\sigma^2 = %s, h(X_{T})$" % data[1, 1],
                    label=r"$h(X_{T})$",
                )

                (l2,) = plt.plot(
                    data_1[:, 10] / (data_1[:, 10] + data_1[:, 11]),
                    data_1[:, 15],
                    # marker="x",
                    color=next(color_iter),
                    linestyle="-",
                    alpha=0.5,
                    label=r"$h(X_{T} \mid O_1)$",
                )
                (l2,) = plt.plot(
                    data_1[:-1, 10] / (data_1[:-1, 10] + data_1[:-1, 11]),
                    data_1[:-1, 16],
                    # marker="o",
                    color=next(color_iter),
                    alpha=0.5,
                    linestyle="-",
                    label=r"$h(X_{T} \mid O_1, O_2')$",
                )

                (l2,) = plt.plot(
                    data_1[:-1, 10] / (data_1[:-1, 10] + data_1[:-1, 11]),
                    data_1[:-1, 17],
                    # marker="o",
                    color=next(color_iter),
                    alpha=0.5,
                    linestyle="-",
                    label=r"$h(X_{T} \mid O_1, O_2', O_3')$",
                )

                (l2,) = plt.plot(
                    data_1[:, 10] / (data_1[:, 10] + data_1[:, 11]),
                    data_2[:, 16],
                    # marker="o",
                    color=next(color_iter),
                    alpha=0.5,
                    linestyle="-",
                    label=r"$h(X_{T} \mid O_1, O_2)$",
                )

                (l2,) = plt.plot(
                    data_1[:-1, 10] / (data_1[:-1, 10] + data_1[:-1, 11]),
                    data_2[:-1, 17],
                    # marker="o",
                    color=next(color_iter),
                    alpha=0.5,
                    linestyle="-",
                    label=r"$h(X_{T} \mid O_1, O_2, O_3')$",
                )

                (l2,) = plt.plot(
                    data_1[:, 10] / (data_1[:, 10] + data_1[:, 11]),
                    data_3[:, 17],
                    # marker="o",
                    color=next(color_iter),
                    alpha=0.5,
                    linestyle="-",
                    label=r"$h(X_{T} \mid O_1, O_2, O_3)$",
                )

                ax.text(
                    0.2,
                    0.07,
                    r"$\sigma_{T, S, S_1, S_2, S_3}^2 = \{%s, %s, %s, %s , %s \}$"
                    % (
                        data_1[1, 1],
                        data_1[1, 3],
                        data_1[1, 5],
                        data_1[1, 7],
                        data_1[1, 9],
                    ),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                )
                ax.text(
                    1.1,
                    -0.1,
                    r"No. Spectators per Exp = %s" % (data_1[1, 10] + data_1[1, 11]),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="bottom",
                )

                ax.legend(
                    loc="lower right",
                    borderpad=0.5,
                    handlelength=1.0,
                    bbox_to_anchor=(1.6, 0.0),
                )
                # ax.set_ylim(bottom=0, top=None)
                ax.set_xlim(left=0, right=None)
                pdf.savefig(fig, bbox_inches="tight")

            plt.clf()

    plt.close("all")

    return 0


def plot_one_target_three_exps():
    experimentName = "joint_sum_normal_three_exp"

    cmd = "ls  ../../output/" + experimentName + "/"
    files = os.listdir("../../output/" + experimentName + "/")
    files = [item for item in files if ".pdf" not in item]
    # print(files, experimentName)
    spec_sizes = ["6_", "10_", "24_"]

    ctr = 0

    # exps = [
    #     "one_target_three_out_of_three_exps_no_common_spec",
    # ]

    # for ex in exps:

    #     filtered = [k for k in files if ex in k]
    #     main1(filtered, ex, experimentName, 2)

    exnames_1 = [
        "one_T_1_3_vsp",
        # "one_T_high_T_1_3_vsp",
        # "one_T_low_T_1_3_vsp",
    ]
    exnames_2 = [
        "one_T_2_3_vsp",
        # "one_T_high_T_2_3_vsp",
        # "one_T_low_T_2_3_vsp",
    ]
    exnames_3 = [
        "one_T_3_3_vsp",
        # "one_T_high_T_3_3_vsp",
        # "one_T_low_T_3_3_vsp",
    ]

    for ex_1, ex_2, ex_3 in zip(exnames_1, exnames_2, exnames_3):

        plot_one_vs_two_vs_three_exp(
            files, spec_sizes, ex_1, ex_2, ex_3, experimentName
        )


if __name__ == "__main__":
    plot_one_target_three_exps()
