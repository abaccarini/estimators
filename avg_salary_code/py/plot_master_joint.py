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
from plot_vary_spec import *

logging.getLogger("matplotlib.font_manager").disabled = True
# import matplotlib.font_manager


output_dir = "../output/"

def plot_main():
    main_exp="joint_sum_normal"
    vary_shared_unique_data = joint_computation_continuous(output_dir, main_exp, "vary_shared_and_unique", joint_col_names, num_spec=None)
    
    num_total_spec = 6
    one_exp_vary_shared_spec = joint_computation_continuous(output_dir, main_exp+"_two_targets", "one_target_one_exp_vary_spec_percent", joint_col_names, num_spec=num_total_spec)
    two_exp_vary_shared_spec = joint_computation_continuous(output_dir, main_exp+"_two_targets", "one_target_two_exp_vary_spec_percent", joint_col_names, num_spec=num_total_spec)
    two_exp_vary_shared_spec_second = joint_computation_continuous(output_dir, main_exp+"_two_targets", "one_target_one_exp_second_not_first_vary_spec_percent", joint_col_names, num_spec=num_total_spec)
    
    
    
    test_sigma = 64.0
    # print(vary_shared_unique_data.get_all_data(test_sigma).single_data["target.sigma"])
    vary_spec_percent_same_target = joint_computation_continuous(output_dir, main_exp, "vary_spec_percent_same_target", joint_col_names, num_spec=10)
    # print(vary_spec_percent_same_target.get_all_data(test_sigma).single_data["diff_ent_joint_exps"])
    # spec_intervals = [0, 1, 5, 10]
    spec_intervals = [0, 1, 5]
    
    # spec_intervals = [0, 1]
    sigma_vals = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0] 

    dir_path = "../../proof/figs/two_computations/"
    
    plot_vary_spec(vary_shared_unique_data, sigma_vals, 4.0, spec_intervals, 25, dir_path)
    
    # for s in spec_intervals:
    # plot_one_vs_two_exp(one_exp_vary_shared_spec, two_exp_vary_shared_spec_second, two_exp_vary_shared_spec, num_total_spec, 4.0, dir_path    )
    
    spec_intervals = [0, 1, 5]
    plot_vary_loss(vary_shared_unique_data, sigma_vals, 4.0, spec_intervals, 25, dir_path)
    # return 0
    # spec_arr =[5, 6, 7, 8, 9, 10, 20, 30, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42 ] 
    spec_intervals = [6, 10, 24]
    for s in spec_intervals:
        
        num_total_spec = s
        one_exp_vary_shared_spec = joint_computation_continuous(output_dir, main_exp+"_two_targets", "one_target_one_exp_vary_spec_percent", joint_col_names, num_spec=num_total_spec)
        two_exp_vary_shared_spec = joint_computation_continuous(output_dir, main_exp+"_two_targets", "one_target_two_exp_vary_spec_percent", joint_col_names, num_spec=num_total_spec)
        two_exp_vary_shared_spec_second = joint_computation_continuous(output_dir, main_exp+"_two_targets", "one_target_one_exp_second_not_first_vary_spec_percent", joint_col_names, num_spec=num_total_spec)
        
        plot_one_vs_two_exp(one_exp_vary_shared_spec, two_exp_vary_shared_spec_second, two_exp_vary_shared_spec, num_total_spec, 4.0, dir_path)
    
        # plot_one_vs_two_exp(one_exp_vary_shared_spec, two_exp_vary_shared_spec, s, 4.0, dir_path)
    
    # plot_one_vs_two_exp_percent([6,10,24], 4.0, dir_path) 
    # plot_one_vs_two_exp_percent([6,10,24], 4.0, dir_path) 
    # ratio([6,10,24], 4.0, dir_path) 
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        plot_main()
