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

logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager

color_arr = ['red', 'blue','darkgreen','magenta','darkorange','teal','saddlebrown']


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


def main1(folder, experiment, experimentName):
    dir_path = "../output/" + experimentName + "_plots/"
    out_path = dir_path + experiment + "_poisson.pdf"
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
        plt.rcParams["figure.figsize"] = [6,6]
        fig, ax = plt.subplots()
        
        # plt.ylabel(r"$h(X_T \mid X_1, X_2)$ (bits)")
        # plt.xlabel(r"No. Spectators per Exp")
        # plt.title(
        #     r"\begin{eqnarray*}X_1 &=X_T + X_S+ X_{S_1}\\X_2 &=X_T + X_S+ X_{S_2}\end{eqnarray*}",
        #     y=1.1,
        # )
        # for idx, dir in enumerate(folder):

        #     print(dir)
        #     result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
        #     # print(result_path)
        #     # print(result_path)
        #     # print(result_path)
        #     data = genfromtxt(
        #         result_path,
        #         delimiter=",",
        #     )

        #     # lam = str(ls_experiments[0]).split("_")
        #     # lam = float(lam[-1])
        #     (l2,) = plt.plot(
        #         data[:, 9],
        #         data[:, 13],
        #         marker="",
        #         color=color_arr[idx],
        #         linestyle="-",
        #         label=r"$\sigma^2 = \{%s, %s\}$" %( data[1, 1], data[1, 5])
        #     )
            
        #     (l2,) = plt.plot(
        #         data[:, 9],
        #         data[:, 14],
        #         marker="",
        #         color=color_arr[idx],
        #         linestyle="--",
        #         # label=r"$h(X_T)$" ,
        #     )
        #     (l2,) = plt.plot(
        #         data[:, 9],
        #         data[:, 15],
        #         marker="",
        #         color=color_arr[idx],
        #         linestyle="dotted",
        #         # label=r"$h(X_T)$" ,
        #     )
            
        # #  \begin{eqnarray*}
        # # - - = h(X_T) \\ \cdots = h(X_T\mid O_1)  \\ -- = h(X_T\mid O_1, O_2)
        # #  \end{eqnarray*}   
            
        # ax.text(1.11, 1, r'\begin{eqnarray*} - - &=& h(X_T) \\ \cdots &=& h(X_T\mid O_1)  \\ \text{---} &=& h(X_T\mid O_1, O_2) \end{eqnarray*}',
        #         # - - $= h(X_T)$ \\ $\cdots$ $= h(X_T\mid O_1)$  \\ -- $= h(X_T\mid O_1, O_2)$'
        #      transform=ax.transAxes,fontsize=14, verticalalignment='top', bbox=dict(boxstyle='square', facecolor='white', alpha=0.5)
        #         )

        # ax.legend(
        #     loc="lower right",
        #     borderpad=0.5,
        #     handlelength=1.0,
        #     bbox_to_anchor=(1.6, 0.2),
        # )
        
        # pdf.savefig(fig, bbox_inches="tight")
        fig, ax = plt.subplots()
        plt.ylabel(r"$h(X_T \mid X_1, X_2)$ (bits)")
        plt.xlabel(r"No. Spectators per Exp")

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
            get_title(data, 1),
            # r"\begin{eqnarray*}X_1 &=X_{T_1} + X_{S_1}\\X_2 &=X_{T_2} + X_{S_2}\end{eqnarray*}",
            y=1.1,
        )

        (l2,) = plt.plot(
            data[:, 9],
            data[:, 13],
            marker="",
            color=color_arr[0],
            linestyle="-",
        label=r"$\lambda = \{%s, %s\}, h(X_T \mid O_1, O_2)$" %( data[1, 1], data[1, 5])
        )
        


        (l2,) = plt.plot(
            data[:, 9],
            data[:, 15],
            marker="",
            color=color_arr[2],
            linestyle="dotted",
            label=r"$\lambda = \{%s, %s\}, h(X_T \mid O_1)$" %( data[1, 1], data[1, 5])
        )       
        (l2,) = plt.plot(
            data[:, 9],
            data[:, 14],
            marker="",
            color=color_arr[1],
            linestyle="--",
            # label=r"$\lambda = %s, h(X_T)$" % data[1, 1],
            label=r"$\lambda = \{%s, %s\}, h(X_T)$" %( data[1, 1], data[1, 5])
        )
        
        ax.legend(
            loc="lower right",
            borderpad=0.5,
            handlelength=1.0,
            bbox_to_anchor=(1, 0.2),
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
        
        
        pdf.savefig(fig, bbox_inches="tight")
        
        # fig, ax = plt.subplots()
        # plt.ylabel(r"$h(X_T \mid X_1, X_2)$ (bits)")
        # plt.xlabel(r"No. Spectators per Exp")
        # plt.title(
        #     r"\begin{eqnarray*}X_1 &=X_T + X_S+ X_{S_1}\\X_2 &=X_T + X_S+ X_{S_2}\end{eqnarray*}",
        #     y=1.1,
        # )
        # dir = folder[-1]
        # result_path = "../output/" + experimentName + "/" + dir + "/results.csv"
        # # print(result_path)
        # # print(result_path)
        # # print(result_path)
        # data = genfromtxt(
        #     result_path,
        #     delimiter=",",
        # )


        # (l2,) = plt.plot(
        #     data[:, 9],
        #     data[:, 13],
        #     marker="",
        #     color=color_arr[0],
        #     linestyle="-",
        # label=r"$\lambda = \{%s, %s\}, h(X_T \mid O_1, O_2)$" %( data[1, 1], data[1, 5])
        # )
        


        # (l2,) = plt.plot(
        #     data[:, 9],
        #     data[:, 15],
        #     marker="",
        #     color=color_arr[2],
        #     linestyle="dotted",
        #     label=r"$\lambda = \{%s, %s\}, h(X_T \mid O_1)$" %( data[1, 1], data[1, 5])
        # )       
        # (l2,) = plt.plot(
        #     data[:, 9],
        #     data[:, 14],
        #     marker="",
        #     color=color_arr[1],
        #     linestyle="--",
        #     # label=r"$\lambda = %s, h(X_T)$" % data[1, 1],
        #     label=r"$\lambda = \{%s, %s\}, h(X_T)$" %( data[1, 1], data[1, 5])
        # )
        
        # ax.legend(
        #     loc="lower right",
        #     borderpad=0.5,
        #     handlelength=1.0,
        #     bbox_to_anchor=(1, 0.2),
        # )
        
        
        # pdf.savefig(fig, bbox_inches="tight")



        plt.clf()

    plt.close("all")

    return 0



if __name__ == "__main__":
    if len(sys.argv) == 1:
        experimentName = "joint_sum_poisson"
        
        
        cmd = "ls  ../output/" + experimentName + "/"
        files = os.listdir("../output/" + experimentName + "/")
        files = [item for item in files if ".pdf" not in item]
        # print(files, experimentName)

        ctr = 0
        exps = ["one_common_spec", "no_common_spec"]
        
        filtered = [k for k in files if exps[ctr] in k]
        main1(filtered, exps[ctr], experimentName)
        ctr+=1 
    
        filtered = [k for k in files if exps[ctr] in k]
        main1(filtered, exps[ctr], experimentName)
        ctr+=1     
        
        # for exp in exps:
        #     filtered = [k for k in files if exp in k]
        #     # print(filtered)
        #     main(filtered, exp,experimentName)

    else:
        exit(0)
