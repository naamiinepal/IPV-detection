# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:20:45 2022

@author: Sagun Shakya

Description:
    Creates visual plots for the learning curves.
"""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from warnings import filterwarnings
filterwarnings(action = 'ignore')

# Local Module(s).
from utils import visualize_learning

def parse_args():
    parser = argparse.ArgumentParser(description = "Visualization of Learning curves.")
    
    parser.add_argument('-i', '--source', 
                        type = str, metavar='PATH', default = r'cache_dir/lstm',
                        help = 'Path to the folder containing cache files.')
    parser.add_argument('-o', '--target', 
                        type = str, metavar='PATH', default = r'./images/lstm_trial',
                        help = 'Path to the folder to store the images.')
    parser.add_argument('-t', '--plot_bar', action = 'store_true', 
                        help = 'Whether to display results in a bar chart.')
    args = parser.parse_args()
    return args

def main(args):
    '''
    Plots the learning and the metrics curves for each fold.
    Also, the test results for each folds are displayed in a bar plot (horizontal).
    The aggregated test results are stored in a CSV file in the source directory.
    
    Parameters
    -----------
    source -- Path to the directory containing the cache files. This includes the loss/metrics value as well as test results.
    target -- Path to the directory to store the PNG files of the plots. If the directory doesn't exist, a new one with the same name will be created.
    '''
    

    target = args.target
    source = args.source

    # If the target folder doesn't exist, create one.
    if not os.path.exists(target):
        os.mkdir(target)

    # Accumulate every filename that starts with 'cache'.
    filenames = [os.path.join(source, file) for file in os.listdir(source) if file.startswith('cache')]

    # Each result of the run is stored in a dictionary. Call cache_df_dict['fold3'], for example.
    cache_df_dict = {'fold' + str(filename[-5]) : pd.read_csv(filename) for filename in filenames}

    plt.style.use('ggplot')
    
    # Plot the learning curves for each fold and save it in the target directory.
    for ii in range(1, len(cache_df_dict) + 1):
        fold = 'fold' + str(ii)
        df = cache_df_dict[fold]
        df.drop(df[df['training loss'] < 1e-6].index, inplace = True)
        visualize_learning(df, save_loc = target, suffix = fold)
    
    if args.plot_test:
        # Test results.
        # Test filenames.
        test_filenames = [os.path.join(source, file) for file in os.listdir(source) if file.startswith('test')]
        assert len(test_filenames) > 0, "No test results in this directory."
        
        # Column names : [testing accuracy, testing precision, testing recall, testing f1 score]
        # Each result of the run is stored in a dictionary. Call test_df_dict['fold3'], for example.
        test_df_dict = {'fold' + str(t_file[-5]) : pd.read_csv(t_file) for t_file in test_filenames}
        
        # Concat all the dataframes into one.
        df = pd.concat(test_df_dict.values()).reset_index(drop = True)
        
        # Change the index names.
        df.index = ['Fold ' + str(ii + 1) for ii in range(len(df))]
        
        # Color pallette.
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        colors = colors[:df.shape[0]]
        
        # Plot horizontal barplot.
        to_plot = df[["testing precision", "testing recall", "testing f1 score"]].T.round(3)
        plot = to_plot.plot.barh(figsize = (15,8), width = 0.8, color = colors, xticks = np.arange(0,1.1,0.1), edgecolor = 'none')
        
        # Add grids on x-axis.
        plot.xaxis.grid(True, linestyle = '-', linewidth = 0.25)
        
        # Add Y-tick labels.
        y_tick_labels = plot.set_yticklabels(['Precision', 'Recall', 'F1 Score'], fontsize = 16)
        
        # Add annotation.
        for container in plot.containers:
            plot.bar_label(container)
        
        # Adjust legend.
        lgd = plot.legend(loc = 'best', prop = {'size' : 10})
            
        # Filename of the plot generated.
        out_filename = os.path.join(target, 'test_results.png')
        
        # Save file.
        fig = plot.get_figure()
        fig.savefig(out_filename)
        
        # Save aggregate test_results.
        df = df.append(df.mean(), ignore_index = True)
        df = df.round(3)
        df.index = ['Fold ' + str(ii + 1) for ii in range(len(df) - 1)] + ['Average']
        df.to_csv(os.path.join(source, 'Aggregated Test Results.csv'))
    
    
    

    
    
# Driver code.
if __name__ == "__main__":
    # Parse the arguments.
    args = parse_args()
    main(args)
