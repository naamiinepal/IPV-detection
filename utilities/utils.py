# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:58:03 2022

@author: Sagun Shakya

Description:
    Helper Functions used in the project for doing minor tasks.
"""
# Importing necessary Libraries.

import io
import os
import logging
from os.path import exists, join
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import yaml
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Helper Functions.
def parse_args_yaml(config_file = 'config.yaml'):
    '''
    Parses the arguments from config file (in YAML format).

    Parameters
    ----------
    config_file : str, optional
        Path to the config file. The default is 'config.yaml'.

    Returns
    -------
    args : DotDict object
        Arguments to be used in the project. The values can be accessed by using notation.

    '''
    # Processing the hyperparameters.
    configuration = load_config(config_file)
    
    return configuration

def log_object(args): 
    '''
    Generates a logger object.

    Parameters
    ----------
    args : DotDict object.
        Arguments for the project.

    Returns
    -------
    logger : logger object on which log information can be written.
    '''      
    
    # If the log directory does not exist, we'll create it.
    if not exists(args.log_dir):
        os.mkdir(args.log_dir)
    
    name_ = f'{args.model}.log'
    log_file = join(args.log_dir, name_)

    # Intialize Logger.
    logger = get_logger(log_file)
    return logger

def write_csv(df: DataFrame, target_filename: str):
    '''
    Writes a csv file for the dataframe.

    Parameters
    ----------
    df : DataFrame
        Source DataFrame.
    target_filename : str
        File path of the target filename.

    Returns
    -------
    None.

    '''
    df.to_csv(target_filename, encoding = 'utf-8', header = None, index = None)


def get_logger(filepath):
    '''
    Gets a logger instance to write the program info and errors to.

    Parameters
    ----------
    filepath : str
        File path to the log output.

    Returns
    -------
    logger : object
        Instance of a logger.
    '''

    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    handler = logging.FileHandler(filepath)
    handler.setFormatter(logging.Formatter(
        "%(levelname)s:%(message)s"
    ))

    logging.getLogger().addHandler(handler)

    with io.open(filepath, "a", encoding="utf-8") as lf:
        lf.write("\n=========================================================================\n")
        lf.write(current_timestamp() + "\n")
        lf.write("=========================================================================\n")

    return logger


def epoch_time(start_time, end_time):
    '''
    Time taken for the epochs to complete.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def load_config(config_path):
    '''
    Loads the YAML config file.

    Parameters
    ----------
    config_path : str
        path to the config file.

    Raises
    ------
    FileNotFoundError
        If file does't exist.

    Returns
    -------
    config : dict
        config dict to be accessed for project.

    '''
    if exists(config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)

    else:
        raise FileNotFoundError()
    return config  

def current_timestamp():
    '''
    Current date and time.

    Returns
    -------
    str
        date and time.

    '''
    dateTimeObj = datetime.now()
    date = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day)
    time = str(dateTimeObj.hour) + ':' + str(dateTimeObj.minute) + ':' + str(dateTimeObj.second)
    
    return f'{date}---{time}'

def timestamp():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d-%m-%Y")  
    return dt_string  

def compute_prec_rec_f1(gold_list, pred_list, average = 'weighted'):
    '''
    Calculates the Precision, Recall, F1-score and ROC-AUC Score.

    Parameters
    ----------
    gold_list : list
        true labels.
    pred_list : list
        predicted labels.

    Returns
    -------
    float.
    Precision, Recall, F1-score and ROC-AUC Score.

    '''
    prec, rec, f1, _ = precision_recall_fscore_support(gold_list, pred_list, average = average)
    gold_list = np.array(gold_list)
    pred_list = np.array(pred_list)
    
    n_values = np.max(gold_list) + 1
    
    # create one hot encoding for auc calculation
    gold_list = np.eye(n_values)[gold_list]
    pred_list = np.eye(n_values)[pred_list]
    auc = roc_auc_score(gold_list, pred_list, average = average)
    return prec, rec, f1, auc

def reset_weights(m):
    '''
    Resets the model weights to avoid weight leakage when we go from one run to the next.

    Parameters
    ----------
    m : pytorch model.

    Returns
    -------
    None.

    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Resetting trainable parameters of layer = {layer}')
            layer.reset_parameters()
            print('Successful!\n')

def count_parameters(model) -> str:
    '''
    Counts the number of trainable parameters.
    Parameters
    ----------
    model : torch model
    Returns
    -------
    str
        Verbose.
    '''
    num_par = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    return f'\nThe model has {num_par:,} trainable parameters.'

def visualize_learning(cache_df, save_loc = './images', suffix = 'fold1'):
    # Loss and Accuracy.
    plt.figure(figsize = (18,6))
    plt.style.use('classic')

    plt.subplot(1,2,1)
    plt.plot(cache_df['training loss'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation loss'], color = 'steelblue', label = 'validation')
    
    legend = plt.legend(loc = 'best', prop = {"size" : 8})
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))
    
    plt.xlabel('Epochs')
    plt.title('Loss curves')

    plt.subplot(1,2,2)
    plt.plot(cache_df['training accuracy'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation accuracy'], color = 'steelblue', label = 'validation')

    legend = plt.legend(loc = 'best', prop = {"size" : 8})
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 1, 0.1))

    plt.xlabel('Epochs')
    plt.title('Accuracy')

    ## Save file.
    file1 = os.path.join(save_loc, f'loss_accuracy_{suffix}.png')
    plt.savefig(file1)
    plt.show()

    # Metrics.
    plt.figure(figsize = (18,6))
    plt.style.use('classic')

    plt.subplot(1,3,1)
    plt.plot(cache_df['training precision'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation precision'], color = 'steelblue', label = 'validation')
    plt.legend(loc = 'lower right', prop = {"size" : 8})
    plt.xlabel('Epochs')
    plt.title('Precision')

    plt.subplot(1,3,2)
    plt.plot(cache_df['training recall'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation recall'], color = 'steelblue', label = 'validation')
    plt.legend(loc = 'lower right', prop = {"size" : 8})
    plt.xlabel('Epochs')
    plt.title('Recall')

    plt.subplot(1,3,3)
    plt.plot(cache_df['training f1 score'], color = 'tomato', label = 'train')
    plt.plot(cache_df['validation f1 score'], color = 'steelblue', label = 'validation')
    plt.legend(loc = 'lower right', prop = {"size" : 8})
    plt.xlabel('Epochs')
    plt.title('F1 Score')
    
    ## Save file.
    file2 = os.path.join(save_loc, f'metrics_{suffix}.png')
    plt.savefig(file2)
    plt.show()