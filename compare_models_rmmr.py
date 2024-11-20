# -*- coding: utf-8 -*-
"""

This script generates and visualizes RMMR (Relative Mated Morph Rate), 
which is a combination of the `MMPMR` (Mated Morph Presentation Match Rate) and 
`FNMR` (False Non Match Rate) metrics. 


## Arguments

Below are the command-line options you can use to customize the script:

| Argument                 | Description                                                                                   | Default Value                                                                                      |
|--------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `-p`, `--models_path`     | Path to the directory containing model data.                                                  | `./models/`                                                                                       |
| `-m`, `--models_names`    | List of model names to compare.                                                               | `['ArcFace_R50_ms1mv2', 'AdaFace_R50_ms1mv2', 'MagFace_R50_ms1mv2', 'GhostFaceNet_ms1mv2', 'test_model']` |
| `-d`, `--display_names`   | Names to display in the plot legend for each model.                                           | `['ArcFace', 'AdaFace', 'MagFace', 'GhostFaceNet', 'Test_Model']`                                 |
| `-l`, `--protocol_label`  | Protocol to be used for comparison. Options: `diffae`, `opencv`, `opencv_sm`, `stylegan`, `stylegan_r`. | `opencv_sm`                                                                                        |
| `-r`, `--mmpmr_rate`      | Type of MMPMR metric to use. Options: `MMPMR`, `MinMax_MMPMR`, `Prod_MMPMR`.                 | `MinMax_MMPMR`                                                                                    |

## Examples

### Example 1: Default Usage

python compare_models_rmmr.py

### Example 2: Compare Specific Models with Different Protocol

python compare_models_rmmr.py -m ArcFace_R50_ms1mv2 MagFace_R50_ms1mv2 -d "ArcFace" "MagFace" -l stylegan -r MMPMR

This command will:
- Compare only `ArcFace_R50_ms1mv2` and `MagFace_R50_ms1mv2` on the `stylegan` 
protocol and save the output as `RMMR_MMPMR_vs_FNMR_stylegan.pdf`.

The plots are saved as a PDF in the `./combined_results/` directory.

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

mmpmr_choices=['MMPMR', 'MinMax_MMPMR','Prod_MMPMR']

def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-p', '--models_path', default = "./models/", help='Path to models')           
    parser.add_argument('-m', '--models_names', nargs='+', default=['ArcFace_R50_ms1mv2', 'AdaFace_R50_ms1mv2', 'MagFace_R50_ms1mv2', 'GhostFaceNet_ms1mv2', 'test_model'], help='List of the models to compare')
    parser.add_argument('-d', '--display_names', nargs='+', default=['ArcFace', 'AdaFace', 'MagFace', 'GhostFaceNet', 'Test_Model'], help='Display model names')
    parser.add_argument('-l', '--protocol_label', default = "opencv_sm" , help='Prolocol labels', choices=["diffae", "opencv", "opencv_sm", "stylegan", "stylegan_r"])     
    parser.add_argument('-r', '--mmpmr_rate', default='MinMax_MMPMR', choices=mmpmr_choices, help='type of MMPMR')
    args = parser.parse_args()
    return args

def plot_combined_RMMR(protocol_name, 
                       path, 
                       colors, 
                       save_name,
                       args):

    array_thresholds = []
    
    array_FR = []
    array_MMPMR = []
    
    #If no particular models are chosen, plot all models:
    for model in args.models_names:
        print(model)
        
        
        thresholds = np.loadtxt("%s/%s/%s/%s" %(path, model, protocol_name, "Thresholds.txt"), dtype=np.float32)
        FR = np.loadtxt("%s/%s/%s/%s%s" %(path, model, protocol_name, args.x_axis_rate,"s.txt"), dtype=np.float32)
        MMPMR = np.loadtxt("%s/%s/%s/%s%s" %(path, model, protocol_name, args.mmpmr_rate,".txt"), dtype=np.float32)

        array_thresholds.append(thresholds)
        array_FR.append(FR)
        array_MMPMR.append(MMPMR)
        
    
    
    
    fig = plt.figure(figsize=(8,5))
    lw = 1
    plot_title = 'RMMR('+args.mmpmr_rate+") "+protocol_name
    

    for idx, nm in enumerate(args.models_names):

        threshs = array_thresholds[idx]
        rates = (array_MMPMR[idx]+array_FR[idx])
        plt.plot(threshs, rates, lw=lw, label = nm,color = colors[idx])
        

    x_lim_min, x_lim_max = 0.0, 1.0
    y_lim_min, y_lim_max = 0.0, 1.0
    plt.xlim([x_lim_min, x_lim_max])
    plt.ylim([y_lim_min, y_lim_max])
    
    plt.xlabel('Threshold',fontsize = 14)
    plt.ylabel('RMMR('+args.mmpmr_rate+')',fontsize = 14)
    plt.title(plot_title,fontsize = 14)
    
    plt.legend(loc="lower right",fontsize=14)

    fig.savefig(save_name) 

if __name__ == "__main__":
       
    args = parse_args()
    
    args.x_axis_rate = 'FNMR'
    
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

    save_name = "./combined_results/"+"RMMR_"+args.mmpmr_rate+"_vs_"+args.x_axis_rate+"_"+args.protocol_label+".pdf"
    plot_combined_RMMR("protocol_mmpmr_Facing2_"+ args.protocol_label, 
                        './models', 
                        colors, 
                        save_name,
                        args)
    
    
    
    
    