# -*- coding: utf-8 -*-
"""
This script is designed to generate and visualize performance comparisons of 
several models using the MMPMR (Mated Morph Presentatoion Match Rate) and FMR (False Match Rate) metrics.
Specifically, the script reads precomputed performance data for various models and plots the results.

Arguments:

| Argument                 | Description                                                                                   | Default Value                                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `-p`, `--models_path`     | Path to the directory containing the model data.                                              | `./models/`                                                                                       |
| `-m`, `--models_names`    | List of model names to compare.                                                               | `['ArcFace_R50_ms1mv2', 'AdaFace_R50_ms1mv2', 'MagFace_R50_ms1mv2', 'GhostFaceNet_ms1mv2', 'test_model']` |
| `-d`, `--display_names`   | Names to display in the plot legend for each model.                                           | `['ArcFace', 'AdaFace', 'MagFace', 'GhostFaceNet', 'Test_Model']`                                 |
| `-l`, `--protocol_label`  | Protocol to be used for comparison. Options: `diffae`, `opencv`, `opencv_sm`, `stylegan`, `stylegan_r`. | `opencv_sm`                                                                                        |
| `-r`, `--mmpmr_rate`      | Type of MMPMR metric to use. Options: `MMPMR`, `MinMax_MMPMR`, `Prod_MMPMR`.                 | `MinMax_MMPMR`                                                                                    |
| `-f`, `--x_axis_rate`     | Metric for the x-axis. Options: `FMR`, `FNMR`.                                               | `FNMR`                                                                                            |

# Examples

## Example 1: Default Usage - compare default models

python compare_models_mmpmr.py

## Example 2: Compare Specific Models with a Different Protocol

python compare_models_mmpmr.py -m ArcFace_R50_ms1mv2 MagFace_R50_ms1mv2 -d "ArcFace" "MagFace" -l stylegan -r MMPMR -f FMR

This command will compare only `ArcFace_R50_ms1mv2` and `MagFace_R50_ms1mv2` models on `stylegan` protocol (`MMPMR`  against `FMR`) 
and save the output as `MMPMR_vs_FMR_stylegan.pdf`.

The plots are saved in PDF format in the `./combined_results/` 

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

mmpmr_choices=['MMPMR', 'MinMax_MMPMR','Prod_MMPMR']
fmr_choices=['FMR', 'FNMR']

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-p', '--models_path', default = "./models/", help='Path to models')           
    parser.add_argument('-m', '--models_names', nargs='+', default=['ArcFace_R50_ms1mv2', 'AdaFace_R50_ms1mv2', 'MagFace_R50_ms1mv2', 'GhostFaceNet_ms1mv2', 'test_model'], help='List of the models to compare')
    parser.add_argument('-d', '--display_names', nargs='+', default=['ArcFace', 'AdaFace', 'MagFace', 'GhostFaceNet', 'Test_Model'], help='Display model names')
    parser.add_argument('-l', '--protocol_label', default = "opencv_sm" , help='Prolocol labels', choices=["diffae", "opencv", "opencv_sm", "stylegan", "stylegan_r"])     
    parser.add_argument('-r', '--mmpmr_rate', default='MinMax_MMPMR', choices=mmpmr_choices, help='type of MMPMR')
    parser.add_argument('-f', '--x_axis_rate', default='FNMR', choices=fmr_choices, help='Rate on the x axis')
    args = parser.parse_args()
    return args


def plot_combined_MMPMR(protocol_name, 
                        path, 
                        colors, 
                        save_name,
                        args):

    x_axis_rates = []
    y_axis_rates = []
    
    print("Models proceeded:")
    for model in args.models_names:
        print(model)

        x_rate = np.loadtxt("%s/%s/%s/%s%s" %(path, model, protocol_name, args.x_axis_rate,"s.txt"), dtype=np.float32)
        y_rate = np.loadtxt("%s/%s/%s/%s%s" %(path, model, protocol_name, args.mmpmr_rate,".txt"), dtype=np.float32)
        
        x_axis_rates.append(x_rate)
        y_axis_rates.append(y_rate)      
    
    
    
    fig = plt.figure(figsize=(8,5))
    lw = 1
    plot_title = args.mmpmr_rate+" "+protocol_name

    for idx, nm in enumerate(args.models_names):
        plt.plot(x_axis_rates[idx], y_axis_rates[idx], lw=lw, label = nm,color = colors[idx])
        
     
        
    x_lim_min, x_lim_max = 0.00001, 0.1
    y_lim_min, y_lim_max = 0.00001, 0.1
    plt.xlim([x_lim_min, x_lim_max])
    #plt.ylim([y_lim_min, y_lim_max])
    plt.xscale('log')
    plt.xlabel(args.x_axis_rate,fontsize = 14)
    plt.ylabel(args.mmpmr_rate,fontsize = 14)
    plt.title(plot_title,fontsize = 14)
    
    #changing lengend lace depending on x_rate choice
    if args.x_axis_rate == fmr_choices[0]:
        plt.legend(loc="lower right",fontsize=14)
    else:
        plt.legend(loc="lower left",fontsize=14) 
    
    fig.savefig(save_name) 



if __name__ == "__main__":
    
    args = parse_args()
     
    colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan'
    ]

    save_name = "./combined_results/"+args.mmpmr_rate+"_vs_"+args.x_axis_rate+"_"+args.protocol_label+".pdf"
    plot_combined_MMPMR("protocol_mmpmr_Facing2_"+ args.protocol_label, 
                        './models', 
                        colors, 
                        save_name,
                        args)




    
    
    
    
    