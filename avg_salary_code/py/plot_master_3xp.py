import logging
from math import dist
import os
import re
import subprocess
import sys
import itertools
from scipy.interpolate import make_interp_spline

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, figure, rc, pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from natsort import natsorted
from numpy import genfromtxt

from plot_classes import *
# from plot_3xp_funcs import *

logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager



mpl.rcParams["text.usetex"] = True
plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts,amsmath,amssymb}",
    }
)
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 12})
rc("text", usetex=True)




output_dir = "../output/"
dir_path = "../../proof/figs/3d_plots/"

target_configs = ["111","110","101","011","100","010","001"]

def h_X_T(sigma):
    return np.log(np.sqrt(sigma * 2.0 * np.pi * np.e)) / np.log(2.0);

def plot_main():
    main_exp="joint_normal_3xp_full"
    total_num_spec = 24
    # vary_S12 = three_exp_continuous(output_dir, main_exp, "vary_S12", continuous_3xp_cols, num_spec=total_num_spec)
    # vary_S12_40 = three_exp_continuous(output_dir, main_exp, "vary_S12_40", continuous_3xp_cols, num_spec=total_num_spec)
    # vary_S12_60 = three_exp_continuous(output_dir, main_exp, "vary_S12_60", continuous_3xp_cols, num_spec=total_num_spec)
    # vary_S13 = three_exp_continuous(output_dir, main_exp, "vary_S13", continuous_3xp_cols, num_spec=total_num_spec)
    # vary_S23 = three_exp_continuous(output_dir, main_exp, "vary_S23", continuous_3xp_cols, num_spec=total_num_spec)
    # vary_S123 = three_exp_continuous(output_dir, main_exp, "vary_S123", continuous_3xp_cols, num_spec=total_num_spec)
    
    # total_num_spec_SPECIAL = 36
    # vary_S123_S12_S13_S23 = three_exp_continuous(output_dir, main_exp, "vary_S123_S12_S13_S23", continuous_3xp_cols, num_spec=total_num_spec_SPECIAL)
    
    # plot_3xp(vary_S12,total_num_spec, 4.0, dir_path)
    # plot_3xp(vary_S12_40,total_num_spec, 4.0, dir_path)
    # plot_3xp(vary_S12_60,total_num_spec, 4.0, dir_path)
    # plot_3xp(vary_S13,total_num_spec, 4.0, dir_path)
    # plot_3xp(vary_S23,total_num_spec, 4.0, dir_path)
    # plot_3xp(vary_S123,total_num_spec, 4.0, dir_path)
    
    # plot_3xp(vary_S123_S12_S13_S23,total_num_spec_SPECIAL, 4.0, dir_path)
    
    
    brute_force_all = three_exp_continuous_brute_force(output_dir, main_exp, "brute_force_all", continuous_3xp_cols_brute_force, num_spec=24)
    
    # print(brute_force_all.class_data["H_X_T_O_123_t110"])
    
    # for d in brute_force_all.class_data:
    #     if all(math.isfinite(y) for y in list(d)[-7:]):
    #         print(d)
    cleanedData = [x for x in brute_force_all.class_data if all(math.isfinite(y) for y in list(x)[-7:])]
    
    # print(len(cleanedData[0]))
    # print(type(cleanedData))
    # print(type(cleanedData[0]))
    cleanedData = np.array(cleanedData)
    # print(type(cleanedData))
    length = len(cleanedData[0])
    min_arr = []
    t_confs = []
    for id, d in enumerate(list(cleanedData)):
        t_confs.append([])
        # val, idx = min((val, idx) for (idx, val) in enumerate((list(d)[-7:])))
        val, idx = min((val, idx) for (idx, val) in enumerate((list(d)[-7:])))
        # print(list(t_list))
        for index, item in enumerate(list(d)[-7:]):
            # print(item, val)
            if str(item) == str(val):
                # print(index)
                # print(target_configs[index])
                t_confs[id].append(target_configs[index])
                
        min_arr.append([ (val)])
        val = length + (idx - 7)
        # cleanedData[id] = np.append(cleanedData[id], val )
        # print(cleanedData[id])
        # print(d, " -- ", val, idx, d[length + (idx - 3)])
    # print(min_arr)
    # min_arr = np.array(cleanedData)
    t_strings = []
    for t in t_confs:
        t_strings.append(["-".join(t)]) 
    # cleanedData = np.concatenate((cleanedData, min_arr), axis=1)
    # zipped_data = zip(list(cleanedData),  t_confs, min_arr)
    zipped_data = zip(list(cleanedData),  t_strings, min_arr)
    # print(zipped_data)
    concat_list = []
    for d in zipped_data:
        concat_list.append(list(d[0]) + list(d[1]) + list(d[2]))  
    
    # for d in concat_list:
    #     print(d)
    sorted_list = sorted(concat_list, key=lambda x: x[-1], reverse=True)   
    
    filtered = []
    # for i,l in enumerate(sorted_list[:]):
    for i,l in enumerate(sorted_list[:500]):
        # print(l, "\t", target_configs[l[-2]])
        # print(l[:9], l[-4:])
        myFilteredList = [x for i, x in enumerate(l) if (( i not in range(9,20) )and (i != 0) and (i!= 8)) ]
        # print(l)
        filtered.append(myFilteredList)
        print(i, myFilteredList)
        # print(l)
    # print(filtered)
    ind_start = 0
    ind_list = [0]
    num_uniques = 0;
    # first_val = filtered[0][-1]
    for i, l in enumerate(filtered):
        if i < len(filtered) - 1:
            if str(filtered[i][-1]) == str(filtered[i+1][-1]):
                ind_start+=1
            else:
                # ind_list[num_uniques][1] = ind_start
                ind_start+=1
                ind_list.append(ind_start)
                # num_uniques+=1
    print(ind_list)
    print(len(ind_list) - 1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # filtered = np.array([np.array(xi) for xi in filtered])
    # filtered = np.array(filtered)
    marks = ['x', 'o', 's']
    mm = itertools.cycle(marks)
    
    cm = plt.cm.get_cmap('RdYlBu')
    s12 = []
    s13 = []
    s23 = []
    entropy = []
    for i in range(len(ind_list)):
        if i == len(ind_list) - 1:
            for f in filtered[ind_list[i]:]:
                s12.append(float(f[3]) + float(f[6]))
                s13.append(float(f[4]) + float(f[6]))
                s23.append(float(f[5]) + float(f[6]))
                entropy.append(float(f[-1]))
        else:
            for f in filtered[ind_list[i]:ind_list[i+1]]:
                s12.append(float(f[3]) + float(f[6]))
                s13.append(float(f[4]) + float(f[6]))
                s23.append(float(f[5]) + float(f[6]))
                entropy.append(float(f[-1]))
     
    for (i,j,k,l) in zip(s12, s13, s23, entropy ):
        print( i,j,k,l)
                
    # print(len(s12))
    # print(len(s13))
    # print(len(s23))
    # print(len(entropy))
    
    m = next(mm)
    c = np.abs(entropy)
    
    cmhot = plt.get_cmap("hot")
    p = ax.scatter(s12, s13, s23, entropy, c = c, edgecolor='black', marker='o', alpha=0.8,linewidth=0.1, cmap="hot")
    
    
    # p = ax.scatter(s12, s13, entropy, edgecolor='black', marker='o', alpha=0.8,linewidth=0.1, cmap="hot",c=entropy)
    
    ax.set_xlabel(r'$s_{12} + s_{123}$')
    ax.set_ylabel(r'$s_{13} + s_{123}$')
    ax.set_zlabel(r'$s_{23} + s_{123}$')
    # ax.set_zlabel(r'$h\left(\vec{X}_T \mid O_1^{(\tau_1)}, O_2^{(\tau_2)}, O_3^{(\tau_3)}\right)$')
    
    # plt.title(r"Total no. spectators per exp. = %s" % total_num_spec)

    # ax.plot(s12, s23, 'r+', zdir='y', zs=13)
    # ax.plot(s13, s23, 'g+', zdir='x', zs=4)
    # ax.plot(s12, s13, 'b+', zdir='z', zs=4)

    # ax.set_xlim([4, 13])
    # ax.set_ylim([4, 13])
    # ax.set_zlim([4, 13])

    interval = 10
    # plt.show()
    
    # for ii in range(0,360 + interval,interval):
    #     ax.view_init(elev=10., azim=ii)
    #     # plt.savefig(dir_path + "2d_entropy_3eval_%d.pdf" % ii)
    #     plt.savefig(dir_path + "3d_entropy_3eval_%d.png" % ii, bbox_inches='tight', dpi=450)
    #     plt.savefig(dir_path + "3d_entropy_3eval_%d.pdf" % ii, bbox_inches='tight')
    

    ax.set_yticks(np.arange(6, 11+1, 2.0))
    ax.set_yticklabels([])
    # ax.set_yticks(np.arange(6, 11+1, 3.0))
    azimuth = 265
    # ax.set_ylabel(r'$s_{13} + s_{123}$',fontsize=8)
    ax.view_init(elev=0, azim=azimuth)
    # ax.tight_layout()
    # fig.subplots_adjust(left=0.0,right=0.5)
    # fig.subplots_adjust(top=0.8, bottom=-0.5, left=-0.1, right=1)
    ax.view_init(elev=0, azim=azimuth)
    # plt.savefig(dir_path + "2d_entropy_3eval_%d.pdf" % ii)
    # plt.savefig(dir_path + "3d_entropy_3eval_%d_0.png" % azimuth, bbox_inches='tight', dpi=450)
    # plt.savefig(dir_path + "3d_entropy_3eval_%d_0.pdf" % azimuth, bbox_inches='tight')

    
    fig,axx = plt.subplots()
    # cbar = plt.colorbar(p,shrink=0.8, aspect=15, locatiaon = 'left')
    cbar = plt.colorbar(p,ax=axx,aspect=40,orientation="horizontal")
    cbar.set_label(r'$\min_{\tau_1,\tau_2,\tau_3}   h\left(\vec{X}_T \mid O_1^{(\tau_1)}, O_2^{(\tau_2)}, O_3^{(\tau_3)}\right)$',labelpad=-53, y=0.0, rotation=0)
    axx.remove()

    # save the same figure with some approximate autocropping
    plt.savefig(dir_path+'plot_onlycbar_tight.pdf',bbox_inches='tight')
    plt.savefig(dir_path+'plot_onlycbar_tight.png',bbox_inches='tight', dpi=450)
    
    plt.close()
    plt.clf()

    # ax.set_zlim([0, 24])

    # fig = plt.figure(figsize=(3,3))
    # # ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot(111)
    # ax.scatter(s12, s23,   entropy, c = c,marker='o',alpha=0.8, cmap="hot")
    # ax.set_xlabel(r'$s_{12} + s_{123}$')
    # ax.set_ylabel(r'$s_{23} + s_{123}$')
    # ax.set_xlim([0, 24])
    # ax.set_ylim([0, 24])
    
    # plt.savefig(dir_path+'2d_projection_s12_23.pdf',bbox_inches='tight')
    
    
    fig = plt.figure(figsize=(2.5,2.5))
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(111)
    ax.scatter(s12, s13,   entropy, c = c,marker='o',alpha=0.8, cmap="hot")
    ax.set_xlabel(r'$s_{12} + s_{123}$')
    ax.set_ylabel(r'$s_{13} + s_{123}$')
    ax.set_xlim([0, 24])
    ax.set_ylim([0, 24])
    ax.set_yticks(np.arange(0, 25, 4.0))
    ax.set_xticks(np.arange(0, 25, 4.0))
    
    plt.savefig(dir_path+'2d_projection_s12_13.pdf',bbox_inches='tight')
    
    
    # fig = plt.figure(figsize=(3,3))
    # # ax = fig.add_subplot(projection='3d')
    # ax = fig.add_subplot(111)
    # ax.scatter(s13, s23,   entropy, c = c,marker='o',alpha=0.8, cmap="hot")
    # ax.set_xlabel(r'$s_{13} + s_{123}$')
    # ax.set_ylabel(r'$s_{23} + s_{123}$')
    # ax.set_xlim([0, 24])
    # ax.set_ylim([0, 24])
    # plt.savefig(dir_path+'2d_projection_s13_23.pdf',bbox_inches='tight')
    
    # plt.show()

    return 0
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    s12_trimmed = []
    s13_trimmed = []
    # s23_trimmed = []
    entropy_trimmed = []
    for x in range(0,24):
        for y in range(0,24):
            current_max = 0
            for (i,j,k) in zip(s12, s13, entropy):
                if (i == x) and (j == y):
                    if k > current_max:
                        current_max = k
            s12_trimmed.append(x)
            s13_trimmed.append(y)
            entropy_trimmed.append(current_max)
            
    # s12_trimmed, s13_trimmed = np.meshgrid(s12_trimmed, s13_trimmed)
    # entropy_trimmed = np.array(entropy_trimmed)
    # p = ax.scatter(s12_trimmed, s13_trimmed, entropy_trimmed, edgecolor='black', marker='o', alpha=0.8,linewidth=0.1, cmap="hot",c=entropy_trimmed)
    # p = ax.plot_trisurf(s12_trimmed, s13_trimmed, entropy_trimmed, edgecolor='none', linewidth=0, cmap="afmhot", antialiased = False)
    p = ax.scatter(s12, s13, entropy, edgecolor='black', marker='o', alpha=0.8,linewidth=0.1, cmap="hot",c=entropy)
    
    ax.set_xlabel(r'$s_{12} + s_{123}$')
    ax.set_ylabel(r'$s_{13} + s_{123}$')
    # ax.set_zlabel(r'$s_{23} + s_{123}$')
    ax.set_zlabel(r'$h\left(\vec{X}_T \mid O_1^{(\tau_1)}, O_2^{(\tau_2)}, O_3^{(\tau_3)}\right)$')
    
    plt.title(r"Total no. spectators per exp. = %s" % total_num_spec)

    plt.savefig("2d_entropy_3eval.pdf", )
    # plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    p = ax.plot_trisurf(s12_trimmed, s13_trimmed, entropy_trimmed, edgecolor='none', linewidth=0, cmap="afmhot", antialiased = False)
    
    ax.set_xlabel(r'$s_{12} + s_{123}$')
    ax.set_ylabel(r'$s_{13} + s_{123}$')
    # ax.set_zlabel(r'$s_{23} + s_{123}$')
    ax.set_zlabel(r'$h\left(\vec{X}_T \mid O_1^{(\tau_1)}, O_2^{(\tau_2)}, O_3^{(\tau_3)}\right)$')
    
    plt.title(r"Total no. spectators per exp. = %s" % total_num_spec)

    plt.savefig("2d_entropy_3eval_surf.pdf", )

    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

     
    m = next(mm)
    c = np.abs(s23)
    
    cmhot = plt.get_cmap("hot")
    p = ax.scatter(s12, s13,  entropy, s23, c = c, edgecolor='black', marker='o', alpha=0.8,linewidth=0.1, cmap="hot")
    cbar = plt.colorbar(p,shrink=0.8, aspect=15, location = 'left')
    # cbar.set_label(r'$h\left(\vec{X}_T \mid O_1^{(\tau_1)}, O_2^{(\tau_2)}, O_3^{(\tau_3)}\right)$',labelpad=-40, y=1.05, rotation=0)
    cbar.set_label(r'$s_{23} + s_{123}$',labelpad=-40, y=1.05, rotation=0)
    
    # p = ax.scatter(s12, s13, entropy, edgecolor='black', marker='o', alpha=0.8,linewidth=0.1, cmap="hot",c=entropy)
    
    ax.set_xlabel(r'$s_{12} + s_{123}$')
    ax.set_ylabel(r'$s_{13} + s_{123}$')
    ax.set_zlabel(r'$h\left(\vec{X}_T \mid O_1^{(\tau_1)}, O_2^{(\tau_2)}, O_3^{(\tau_3)}\right)$')
    # ax.set_zlabel(r'$h\left(\vec{X}_T \mid O_1^{(\tau_1)}, O_2^{(\tau_2)}, O_3^{(\tau_3)}\right)$')
    
    plt.title(r"Total no. spectators per exp. = %s" % total_num_spec)
    # plt.show()
    plt.savefig("2d_entropy_s23_cmap.pdf", )

    
    plt.close()
    
    # print(h_X_T(4.0))
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_main()
