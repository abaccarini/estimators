# import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
import numpy as np
from matplotlib import cm
from matplotlib import rc
from matplotlib import figure
from numpy import genfromtxt
import logging
import math
from natsort import natsorted
import itertools

from plot_joint_normal import main2
from plot_joint_normal import get_dist

logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager

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

sigmas = {
    0.25: "0",
    0.5: "1",
    1.0: "2",
    2.0: "3",
    4.0: "4",
    8.0: "5",
    16.0: "6",
    32.0: "7",
    64.0: "8",
    128.0: "9",
}

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


def plot_sh_vs_diff(experimentName, data_dirs, listof_N_exps, N_vals, sigma_vals, epsilon, integreation_type):
    dir_path = "../output/" + experimentName + "_plots/"
    out_path = dir_path + "sh_vs_diff_" + epsilon + "_" + integreation_type+".pdf"
    print(out_path)
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_path)
    # sigma_id = sigmas[sigma_val]
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir_path)
        print("The new directory is created!")

    all_data = {}

    for d_dir in data_dirs:
        path = "../output/" + experimentName + "/" + d_dir + "/results.csv"
        # all_data[N_exp] = { d_dir: genfromtxt( path, dtype=float, delimiter=",",names=True )}
        all_data.update(
            {d_dir: genfromtxt(path, dtype=float, delimiter=",", names=True)}
        )

    with PdfPages(out_path) as pdf:
        for N_val in N_vals:

            for sig in sigma_vals:
                # print(all_data[str(N_val)+'_' + sigmas[sig]])
                t_mu = all_data[str(N_val) + "_" + sigmas[sig]]["target_params_mu"][0]
                t_sigma = all_data[str(N_val) + "_" + sigmas[sig]][
                    "target_params_sigma"
                ][0]
                spec_mu = all_data[str(N_val) + "_" + sigmas[sig]]["spec_params_mu"][0]
                spec_sigma = all_data[str(N_val) + "_" + sigmas[sig]][
                    "spec_params_sigma"
                ][0]

                num_spec = all_data[str(N_val) + "_" + sigmas[sig]]["num_spec"]

                H_T = all_data[str(N_val) + "_" + sigmas[sig]]["H_T"]
                H_S = all_data[str(N_val) + "_" + sigmas[sig]]["H_S"]
                H_T_S = all_data[str(N_val) + "_" + sigmas[sig]]["H_T_S"]
                delta_T = all_data[str(N_val) + "_" + sigmas[sig]]["delta_T"]
                delta_S = all_data[str(N_val) + "_" + sigmas[sig]]["delta_S"]
                delta_T_S = all_data[str(N_val) + "_" + sigmas[sig]]["delta_T_S"]
                awae_shannon = all_data[str(N_val) + "_" + sigmas[sig]]["awae_shannon"]
                h_T = all_data[str(N_val) + "_" + sigmas[sig]]["h_T"]
                h_S = all_data[str(N_val) + "_" + sigmas[sig]]["h_S"]
                h_T_S = all_data[str(N_val) + "_" + sigmas[sig]]["h_T_S"]
                awae_differential = all_data[str(N_val) + "_" + sigmas[sig]][
                    "awae_differential"
                ]

                H_T_delta = np.add(H_T, delta_T)
                H_S_delta = np.add(H_S, delta_S)
                H_T_S_delta = np.add(H_T_S, delta_T_S)
                awae_delta = np.subtract(np.add(H_T_delta, H_S_delta), H_T_S_delta)
                
                fig, ax = plt.subplots()

                plt.ylabel(r"Entropy (bits)")
                plt.xlabel(r"No. Spectators")

                cc = itertools.cycle(colors)
                plot_lines = []

                c = next(cc)
                (l1,) = plt.plot(
                    num_spec,
                    awae_shannon,
                    marker="",
                    color='r',
                    linestyle="-",
                    label=r"$H(X_T^\Delta \mid O^\Delta)$",
                )
                c = next(cc)

                (l2,) = plt.plot(
                    num_spec,
                    H_T,
                    marker="",
                    color='b',
                    linestyle="--",
                    label=r"$H(X^\Delta_T)$",
                )
                ax.legend(
                    loc="best",
                    borderpad=0.5,
                    handlelength=1.0,
                    # bbox_to_anchor=(1.6, 0.0),
                )
                # plot_lines.append([l1, l2])
                plt.title(
                    r"Shannon Approx., $X \sim \mathcal{N}(%s, %s)$" % (t_mu, t_sigma)
                )
                plt.text(
                    -0.2,
                    1.2,
                    r"\begin{align*}N &= %s\\\varepsilon &= 10^{-%s}\end{align*}" % (int(N_val), epsilon),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
                )
                pdf.savefig(fig, bbox_inches="tight")

                plt.cla()
                plt.ylabel(r"Entropy (bits)")
                plt.xlabel(r"No. Spectators")
                # c = next(cc)
                (l1,) = plt.plot(
                    num_spec,
                    awae_differential,
                    marker="",
                    color='r',
                    linestyle="-",
                    label=r"$h(X_T \mid O)$",
                )
                # c = next(cc)

                (l2,) = plt.plot(
                    num_spec,
                    h_T,
                    marker="",
                    color='b',
                    linestyle="--",
                    label=r"$h(X_T)$",
                )
                ax.legend(
                    loc="best",
                    borderpad=0.5,
                    handlelength=1.0,
                    # bbox_to_anchor=(1.6, 0.0),
                )
                plot_lines.append([l1, l2])
                plt.title(
                    r"Differential, $X\sim \mathcal{N}(%s, %s)$"
                    % (t_mu, t_sigma)
                )
                plt.text(
                    -0.2,
                    1.2,
                    r"\begin{align*}N &= %s\\\varepsilon &= 10^{-%s}\end{align*}" % (int(N_val), epsilon),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
                )

                pdf.savefig(fig, bbox_inches="tight")
                plt.cla()

                plt.ylabel(r"Percent Decrease")
                plt.xlabel(r"No. Spectators")
                c = next(cc)
                (l1,) = plt.plot(
                    num_spec,
                    [((a - b) / a) * 100.0 for a, b in zip(H_T, awae_shannon)],
                    marker="",
                    color='r',
                    linestyle="-",
                    label=r"$ \frac{H(X^\Delta_T) -  H(X_T^\Delta\mid O)}{H(X_T^\Delta)}$",
                )

                (l2,) = plt.plot(
                    num_spec,
                    [((a - b) / a) * 100.0 for a, b in zip(h_T, awae_differential)],
                    marker="",
                    color='b',
                    linestyle="--",
                    label=r"$ \frac{h(X_T) -  h(X_T\mid O)}{h(X_T)}$",
                )
                ax.legend(
                    loc="best",
                    borderpad=0.5,
                    handlelength=1.0,
                    # bbox_to_anchor=(1.6, 0.0),
                )
                plt.text(
                    -0.2,
                    1.2,
                    r"\begin{align*}N &= %s\\\varepsilon &= 10^{-%s}\end{align*}" % (int(N_val), epsilon),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
                )

                plot_lines.append([l1, l2])
                plt.title(
                    r"Percentage Loss, $X\sim \mathcal{N}(%s, %s)$"
                    % (t_mu, t_sigma)
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.cla()

                plt.ylabel(r"Entropy (bits)")
                plt.xlabel(r"No. Spectators")
                c = next(cc)
                (l1,) = plt.plot(
                    num_spec,
                    awae_differential,
                    marker="",
                    color='r',
                    linestyle="-",
                    label=r"$h(X_T \mid O)$",
                )
                (l2,) = plt.plot(
                    num_spec,
                    h_T,
                    marker="",
                    color='b',
                    linestyle="--",
                    label=r"$h(X_T)$",
                )
                
                (l3,) = plt.plot(
                    num_spec,
                    awae_delta,
                    marker="",
                    color='green',
                    linestyle="-",
                    label=r"$H(X_T \mid O) + \log \Delta_T \Delta_S \Delta_{T,S} $",
                )
                (l4,) = plt.plot(
                    num_spec,
                    H_T_delta,
                    # [a + math.log(b) for a,b in zip(H_T, delta_T)],
                    marker="",
                    color='purple',
                    linestyle="--",
                    label=r"$H(X_T) + \log \Delta_T$",
                )                
                
                
                ax.legend(
                    loc="best",
                    borderpad=0.5,
                    handlelength=1.0,
                    # bbox_to_anchor=(1.6, 0.0),
                )
                plt.text(
                    -0.1,
                    1.1,
                    r"$N = %s$" % int(N_val),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
                )

                plot_lines.append([l1, l2])
                plt.title(
                    r"Approx vs. Differential, $X\sim \mathcal{N}(%s, %s)$"
                    % (t_mu, t_sigma)
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.cla()

                plt.ylabel(r"Percent Decrease")
                plt.xlabel(r"No. Spectators")
                # c = next(cc)
                

                (l1,) = plt.plot(
                    num_spec,
                    [((a - b) / a) * 100.0 for a, b in zip( H_T_delta , awae_delta )],
                    marker="",
                    color='r',
                    linestyle="-",
                    label=r"$ \frac{H(X^\Delta_T)  + \log \Delta_T -  (H(X_T^\Delta\mid O) +  \log \Delta_T \Delta_S \Delta_{T,S} )}{H(X_T^\Delta)  + \log \Delta_T }$",
                )

                (l2,) = plt.plot(
                    num_spec,
                    [((a - b) / a) * 100.0 for a, b in zip(h_T, awae_differential)],
                    marker="",
                    color='b',
                    linestyle="--",
                    label=r"$ \frac{h(X_T) -  h(X_T\mid O)}{h(X_T)}$",
                )
                
                ax.legend(
                    loc="best",
                    borderpad=0.5,
                    handlelength=1.0,
                    # bbox_to_anchor=(1.6, 0.0),
                )
                plt.text(
                    -0.2,
                    1.2,
                    r"\begin{align*}N &= %s\\\varepsilon &= 10^{-%s}\end{align*}" % (int(N_val), epsilon),
                    transform=ax.transAxes,
                    fontsize=14,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.3),
                )

                plot_lines.append([l1, l2])
                plt.title(
                    r"Approx vs. Differential, $X\sim \mathcal{N}(%s, %s)$"
                    % (t_mu, t_sigma)
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.cla()                
    plt.close("all")

    return 0


def plot_main():
    precisions = ["5", "10", "20"]
    integ_tyeps = ["gsl", "trap"]
    
    for prec in precisions:
        for integ in integ_tyeps:
        
            experimentName = "normal_v2/single_experiment_shannon_vs_diff_" + integ + "_"+ prec
            epsilon = experimentName.split('_')[-1]
            print(epsilon)
            cmd = "ls  ../output/" + experimentName + "/"
            files = os.listdir("../output/" + experimentName + "/")
            files = [item for item in files if ".pdf" not in item]
            # print(files)

            N_vals = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            N_data = []
            for N_val in N_vals:
                N_data.append([x for x in files if str(N_val) == x.split("_")[0]])
            # print(N_data)
            plot_sh_vs_diff(experimentName, files, N_data, [8, 128], [0.25, 0.5, 2.0], epsilon,integ)

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_main()
