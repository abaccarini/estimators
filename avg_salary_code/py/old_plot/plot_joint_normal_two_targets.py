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
from plot_joint_normal import get_dist, get_title

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

# def get_title(data, numTargets):

#     # exam_name = 'num_shared'
#     # exam_index = np.where(data[0] == exam_name)
#     # print(data[1])
#     # print(exam_index)
#     # print(int(data[8, 1]))

#     if (data[1, 8]) == 0:
#         return r"\begin{eqnarray*}O_1 &=X_{T_1} + X_{S_1}\\O_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}"
#     else:
#         return r"\begin{eqnarray*}O_1 &=X_{T_1} +X_S+ X_{S_1}\\O_2 &=X_{T_2}+X_S + X_{S_2}\end{eqnarray*}"


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
        # - - = h(X_{T_1}) \\ \cdots = h(X_{T_1}\mid O_1)  \\ -- = h(X_{T_1}\mid O_1, O_2)
        #  \end{eqnarray*}

        ax.text(
            1.11,
            1.4,
            r"\begin{eqnarray*} - - &=& h(X_{T_1}) \\ \cdots &=& h(X_{T_1}\mid O_1)  \\ \text{---} &=& h(X_{T_1}\mid O_1, O_2) \end{eqnarray*}",
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
            data[:, 9],
            data[:, 13],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"$h(X_{T_1} \mid O_1, O_2)$",
        )

        (l2,) = plt.plot(
            data[:, 9],
            data[:, 15],
            marker="",
            color=color_arr[2],
            linestyle="dotted",
            label=r"$h(X_{T_1} \mid O_1)$",
        )
        (l2,) = plt.plot(
            data[:, 9],
            data[:, 14],
            marker="",
            color=color_arr[1],
            linestyle="--",
            # label=r"$\sigma^2 = %s, h(X_{T_1})$" % data[1, 1],
            label=r"$h(X_{T_1})$",
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
            data[0:upper_bound, 9],
            [a - b for a, b in zip(data[0:upper_bound, 15], data[0:upper_bound, 13])],
            # data[0:upper_bound, 13] - data[0:upper_bound, 15],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"\begin{eqnarray*}h(X_{T_1} \mid O_1) - h(X_{T_1} \mid O_1, O_2)\end{eqnarray*}",
        )

        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.9, 0.0),
        )

        ax.text(
            0.3,
            0.07,
            r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
            % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
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
            data[:, 9],
            data[:, 13],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"$h(X_{T_1} \mid O_1, O_2)$",
        )

        (l2,) = plt.plot(
            data[:, 9],
            data[:, 15],
            marker="",
            color=color_arr[2],
            linestyle="dotted",
            label=r"$h(X_{T_1} \mid O_1)$",
        )
        (l2,) = plt.plot(
            data[:, 9],
            data[:, 14],
            marker="",
            color=color_arr[1],
            linestyle="--",
            # label=r"$\sigma^2 = %s, h(X_{T_1})$" % data[1, 1],
            label=r"$h(X_{T_1})$",
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
            data[0:upper_bound, 9],
            [a - b for a, b in zip(data[0:upper_bound, 15], data[0:upper_bound, 13])],
            # data[0:upper_bound, 13] - data[0:upper_bound, 15],
            marker="",
            color=color_arr[0],
            linestyle="-",
            label=r"\begin{eqnarray*}h(X_{T_1} \mid O_1) - h(X_{T_1} \mid O_1, O_2)\end{eqnarray*}",
        )

        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1.9, 0.0),
        )

        ax.text(
            0.3,
            0.07,
            r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
            % (data[1, 1], data[1, 3], data[1, 5], data[1, 7]),
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


def plot_one_vs_two_exp(
    all_folders, spec_sizes, experiment_1, experiment_2, experimentName
):
    # for file in files:
    dir_path = "../output/" + experimentName + "_plots/"
    out_path = dir_path + experiment_2 + "_vs_two_normal.pdf"
    print(out_path)
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_path)
    dir_list = [-1, 0]  # two samples

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")

    with PdfPages(out_path) as pdf:
        for sz in spec_sizes:
            folder_1 = [k for k in all_folders if sz + experiment_1 in k]
            folder_2 = [k for k in all_folders if sz + experiment_2 in k]

            folder_1 = natsorted(folder_1)
            folder_1.reverse()

            folder_2 = natsorted(folder_2)
            folder_2.reverse()

            for dir_l in dir_list:
                fig, ax = plt.subplots()
                plt.ylabel(r"Entropy (bits)")
                plt.xlabel(r"Unique Spectators per Exp")

                dir_1 = folder_1[dir_l]
                result_path_1 = (
                    "../output/" + experimentName + "/" + dir_1 + "/results.csv"
                )
                # print("dir_1 = ",dir_1)

                dir_2 = folder_2[dir_l]
                # print("dir_2 = ",dir_2)
                result_path_2 = (
                    "../output/" + experimentName + "/" + dir_2 + "/results.csv"
                )

                data_1 = genfromtxt(
                    result_path_1,
                    delimiter=",",
                )

                data_2 = genfromtxt(
                    result_path_2,
                    delimiter=",",
                )
                plt.title(
                    "Participating in one vs. two experiment(s)"
                    # r"\begin{align*}O_1 &= X_{T} +X_S + X_{S_1}\\O_2 &= X_{T}+ X_S+ X_{S_2}\\ O_2' &= X_S + X_{S_2}\end{align*}",
                    # y=1.1,
                )
                ax.text(
                    1.13,
                    1.0,
                    r"\begin{align*}O_1 &= X_{T} +X_S + X_{S_1}\\O_2 &= X_{T}+ X_S+ X_{S_2}\\ O_2' &= X_S + X_{S_2}\end{align*}",
                    transform=ax.transAxes,
                    # fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
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
                    data_1[:, 8] / (data_1[:, 8] + data_1[:, 9]),
                    data_1[:, 13],
                    marker="x",
                    color=color_arr[0],
                    linestyle="-",
                    alpha=0.5,
                    label=r"$h(X_{T} \mid O_1, O_2)$",
                )

                (l2,) = plt.plot(
                    data_2[:, 8] / (data_2[:, 8] + data_2[:, 9]),
                    data_2[:, 13],
                    marker="o",
                    color=color_arr[3],
                    alpha=0.5,
                    linestyle="-",
                    label=r"$h(X_{T} \mid O_1, O_2')$",
                )

                (l2,) = plt.plot(
                    data_1[:, 8] / (data_1[:, 8] + data_1[:, 9]),
                    data_1[:, 15],
                    marker="",
                    color=color_arr[2],
                    linestyle="dotted",
                    label=r"$h(X_{T} \mid O_1)$",
                )

                (l2,) = plt.plot(
                    data_1[:, 8] / (data_1[:, 8] + data_1[:, 9]),
                    data_1[:, 14],
                    marker="",
                    color=color_arr[1],
                    linestyle="--",
                    # label=r"$\sigma^2 = %s, h(X_{T})$" % data[1, 1],
                    label=r"$h(X_{T})$",
                )

                # ax.set_xlim(xmin = 0)

                # plt.autoscale(enable=True, axis='x', tight=True)

                # plt.ylim(ymin=0.0)
                ax.text(
                    0.1,
                    0.07,
                    r"$\sigma_{T, S, S_1, S_2}^2 = \{%s, %s, %s, %s \}$"
                    % (data_1[1, 1], data_1[1, 3], data_1[1, 5], data_1[1, 7]),
                    # r"$\sigma^2 = \{%s, %s\}$" % (data[1, 1], data[1, 5]),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                )
                # (l2,) = plt.plot(
                #     data_2[:, 8] / (data_2[:, 8] + data_2[:, 9]),
                #     data_2[:, 15],
                #     marker="",
                #     color=next(color_iter),
                #     linestyle="dotted",
                #     label=r"$h(X_{T_1} \mid O_1)$",
                # )
                # (l2,) = plt.plot(
                #     data_2[:, 8] / (data_2[:, 8] + data_2[:, 9]),
                #     data_2[:, 14],
                #     marker="",
                #     color=next(color_iter),
                #     linestyle="--",
                #     # label=r"$\sigma^2 = %s, h(X_{T_1})$" % data[1, 1],
                #     label=r"$h(X_{T_1})$",
                # )

                ax.text(
                    1.1,
                    -0.1,
                    r"No. Spectators per Exp = %s" % (data_1[1, 8] + data_1[1, 9]),
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


def plot_2_targets():
    experimentName = "joint_sum_normal_two_targets"

    cmd = "ls  ../output/" + experimentName + "/"
    files = os.listdir("../output/" + experimentName + "/")
    files = [item for item in files if ".pdf" not in item]
    # print(files, experimentName)

    ctr = 0

    exps = [
        "two_targets_ncs",
        "two_targets_one_cs",
        "two_targets_one_high_target_sigma_cs",
        "two_targets_one_low_target_sigma_cs",
    ]

    for ex in exps:

        filtered = [k for k in files if ex in k]
        main1(filtered, ex, experimentName, 2)

    spec_sizes = ["10_", "50_", "100_"]
    exnames = [
        "two_targets_vary_spec_percent",
        "two_targets_low_target_sigma_vary_spec_percent",
        "two_targets_high_target_sigma_vary_spec_percent",
    ]

    for ex in exnames:
        main2(files, ex, spec_sizes, experimentName, 2)

    exnames_1 = [
        "one_target_two_exp_vary_spec_percent",
        "one_target_low_two_exp_vary_spec_percent",
        "one_target_high_two_exp_vary_spec_percent",
    ]
    exnames_2 = [
        "one_target_one_exp_vary_spec_percent",
        "one_target_low_one_exp_vary_spec_percent",
        "one_target_high_one_exp_vary_spec_percent",
    ]

    for ex_1, ex_2 in zip(exnames_1, exnames_2):

        main2(files, ex_1, spec_sizes, experimentName, 1)
        main2(files, ex_2, spec_sizes, experimentName, -1)
        plot_one_vs_two_exp(files, spec_sizes, ex_1, ex_2, experimentName)

 

if __name__ == "__main__":
    plot_2_targets()
