from scipy.linalg import null_space
from plot_ksg_est import *


def plot_mu_var(dist, param_str):
    full_fname = "mu_var_comp"
    # verify_args(fname, dist)

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
            [mean - stdev * i for i in (range(1, tick_upper))]
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
        plt.xticks(fontsize=18)

    if dist == "uniform_int":
        tick_upper = 2
        a, b = getParams(dist, param_str)
        # print(a, b)
        b = b - 1.0
        mean = (a + b) / 2.0
        stdev = np.sqrt(((b - a + 1.0) * (b - a + 1.0) - 1.0) / 12.0)
        xtick = (
            [mean - stdev * i for i in (range(1, tick_upper))]
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
        plt.xticks(fontsize=18)

    if dist == "normal":
        mu, sigma = getParams(dist, param_str)
        tick_upper = 4

        xtick = (
            [mu - sigma * i for i in (range(1, tick_upper))]
            + [mu]
            + [mu + sigma * i for i in (range(1, tick_upper))]
        )
        xlabels = (
            [
                r"$\mu {-} %s \sigma$" % (i if i > 1 else "")
                for i in (range(1, tick_upper))
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
        mu, sigma = getParams(dist, param_str)
        mean = np.exp(mu + (sigma * sigma) / 2.0)
        sig = np.sqrt(
            (np.exp((sigma * sigma)) - 1.0) * (np.exp(2.0 * mu + sigma * sigma))
        )
        tick_upper = 4
        xtick = (
            [mean - sig * i for i in (range(1, tick_upper))]
            + [mean]
            + [mean + sig * i for i in (range(1, tick_upper))]
        )
        # print(mean)
        # print(sig)
        # print(xtick)
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
        plt.xticks(fontsize=18, rotation=45)

    lower_bound, upper_bound = getBounds(dist, param_str)
    cc = itertools.cycle(["blue", "orange"])
    alph = 0.5

    spec_vals = [2, 5]
    linestys = ["o", "x"]
    msty = ["-", "--"]
    # spec_vals = [5]
    fnames = ["mean", "var", "var_mu"]

    fname = "mean"
    path = data_dir + fname + "/" + dist
    files = os.listdir(path)
    file = [f for f in files if Path(f).stem == param_str][0]

    with open(path + "/" + file) as json_file:
        mean_json = json.load(json_file)

    t_init = mean_json["target_init_entropy"]

    mean_data = []
    for s in spec_vals:
        mean_data.append(mean_json["awae_data"][str(s)])

    fname = "var"
    path = data_dir + fname + "/" + dist
    files = os.listdir(path)
    file = [f for f in files if Path(f).stem == param_str][0]

    with open(path + "/" + file) as json_file:
        var_json = json.load(json_file)
    var_data = []
    for s in spec_vals:
        var_data.append(var_json["awae_data"][str(s)])

    fname = "var_mu"
    path = data_dir + fname + "/" + dist
    files = os.listdir(path)
    file = [f for f in files if Path(f).stem == param_str][0]
    with open(path + "/" + file) as json_file:
        mu_var_json = json.load(json_file)
    mu_var_data = []
    for s in spec_vals:
        mu_var_data.append(mu_var_json["awae_data"][str(s)])
    
    for md, vd, mvd, numSpec in zip(mean_data, var_data, mu_var_data, spec_vals):

        plot_lines = []


        lsty = itertools.cycle(linestys)
        mst = itertools.cycle(msty)
        c = next(cc)
        def plotfn_cont(ubound):
            x_A = list(map(float, list( md.keys() )))
            awae_mu = list(md.values())
            awae_var =list( vd.values())
            awae_mu_var =list( mvd.values())
            awae_mu_plus_var = [x + y for x,y in zip(awae_mu, awae_var)]


            if max(x_A) > ubound:
                ubound = max(x_A)
                global upper_bound
                upper_bound = ubound
            ls = next(lsty)
            ms = next(mst)

            awae_mu_var  = savgol_filter(awae_mu_var , 51, 3)  # window size 51, polynomial order 3
            awae_mu_plus_var = savgol_filter(awae_mu_plus_var, 51, 3)  # window size 51, polynomial order 3
            label = r"$\lvert S\rvert = %s$" % numSpec
            (l2,) = plt.plot(
                x_A,
                awae_mu_var,
                marker="",
                color=c,
                alpha=alph,
                linestyle="--",
                label=label,
            )
            (l2,) = plt.plot(
                x_A,
                awae_mu_plus_var,
                marker="",
                color=c,
                alpha=alph,
                linestyle="-",
                label=label,
            )
            plot_lines.append(l2)

        def plotfn_disc():
            x_A = list( md.keys() )
            awae_mu = list(md.values())
            awae_var =list( vd.values())
            awae_mu_var =list( mvd.values())
            awae_mu_plus_var = [x + y for x,y in zip(awae_mu, awae_var)]

            label = r"$\lvert S\rvert = %s$" % numSpec

            (l2,) = plt.plot(
                x_A[lower_bound:upper_bound],
                awae_mu_var  [lower_bound:upper_bound],
                marker="x",
                color=c,
                alpha=alph,
                linestyle="--",
                label=label,
            )
            (l2,) = plt.plot(
                x_A[lower_bound:upper_bound],
                awae_mu_plus_var [lower_bound:upper_bound],
                marker="o",
                color=c,
                alpha=alph,
                linestyle="-",
                label=label,
            )
            plot_lines.append(l2)

        if dist in ["normal", "lognormal"]:
            plotfn_cont(upper_bound)
        elif dist in ["uniform_int", "poisson"]:
            plotfn_disc()

        plt.xticks(
            np.arange(lower_bound, upper_bound, 1), minor=True
        )  # set minor ticks on x-axis
        # plt.yticks(
        #     np.arange(lower_bound, upper_bound, 1), minor=True
        # )  # set minor ticks on y-axis
        plt.tick_params(which="minor", length=0)  # remove minor tick lines

    out_path = fig_dir + full_fname + "/" + dist
    Path(out_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        # out_path + "/" + param_str + "_" +oe_str[oe_key]+"_discrete_leakage.pdf",
        out_path + "/" + param_str + "_leakage.pdf",
        bbox_inches="tight",
    )
    plt.close("all")


def generateLegend():
    figl, axl = plt.subplots()
    
    fn_names = [r"$H_{f_{\mu}} + H_{f_{\sigma^{2}}}$", r"$H_{f_{(\mu,\sigma^{2})}}$"]
    marks = ["o", "x"]
    lsts = ["-", "--"]
    fn_leg = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=mk,
            alpha=0.7,
            linestyle=ls,
            label=r"$%s$" % fname,
        )
        for fname, mk, ls in zip(fn_names,marks, lsts)
    ]
    spec_leg = [
        Line2D(
            [0],
            [0],
            color="blue",
            alpha=0.9,
            linestyle="-",
            label=r"$\lvert S\rvert = 2$",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            alpha=0.9,
            linestyle="-",
            label=r"$\lvert S\rvert = 5$",
        ),
    ]
    # no_a_legend = [

    # ]
    # # figl2, axl2 = plt.subplots(figsize=(0, 0))
    plt.axis(False)
    axl.margins(0, 0)
    figl.subplots_adjust(left=0, right=0.8, bottom=0, top=0.3)
    leg1 = plt.legend(
        # handles=reorder(all_specs_legend, 5),
        handles=fn_leg,
        loc="center",
        bbox_to_anchor=(0.63, 0.5),
        # columnspacing=0.5,
        borderpad=0.5,
        fontsize="small",
        ncol=3,
        # handlelength=1,
    )
    plt.gca().add_artist(leg1)
    leg3 = plt.legend(
        handles=spec_leg,
        loc="center",
        bbox_to_anchor=(0.1, 0.5),
        borderpad=0.5,
        fontsize="small",
        ncol=1,
        # handlelength=1,
    )

    plt.savefig(
        fig_dir + "/mu_var_comp/legend_text_only" + ".pdf",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close("all")


def main():
    generateLegend()
    # return
    json_fname = data_dir + "p_strs.json"
    fname = open(json_fname)
    param_strs_dict = json.load(fname)

    disc_dists = [
        "uniform_int",
        "poisson",
    ]

    cont_dists = [
        "normal",
        "lognormal",
    ]

    for dname in disc_dists:
        for p_str in param_strs_dict[dname]:
            print(dname, p_str)
            plot_mu_var(dname, p_str)

    for dname in cont_dists:
        for p_str in param_strs_dict[dname]:
            print(dname, p_str)
            plot_mu_var(dname, p_str)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
