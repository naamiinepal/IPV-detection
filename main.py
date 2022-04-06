# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:44:18 2022

@author: Sagun Shakya
"""
import os
import argparse

from models.ml_models import svm, naive_bayes, random_forest, logistic_regression, adaboost
from utilities import utils
from utilities.read_configuration import DotDict
from models.ml_models.new_tool import InstantiateModel

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
    configuration = utils.load_config(config_file)
    
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
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    if args.model == "svm":
        name_ = f'svm_log_{args.svm.kernel}_{str(args.svm.C)}_{str(args.vectorizer.mode)}.log'
        
    elif args.model == "nb":
        name_ = f'nb_log_{args.nb.alpha}_{str(args.vectorizer.mode)}.log'

    elif args.model == "logistic_regression":
        name_ = f'logistic_regression_log_{args.logistic_regression.C}_{args.logistic_regression.max_iter}_{str(args.vectorizer.mode)}.log'
    
    else: 
        name_ = 'experiment_log.log'
    
    log_file = os.path.join(args.log_dir, name_)

    # Intialize Logger.
    logger = utils.get_logger(log_file)
    return logger

def parse_arguments():
    '''
    Argument parser for IPV project.

    Returns
    -------
    args : TYPE
        Arguments for the run.

    '''
    CHOICES = ['svm', 'nb', 'random_forest', 'adaboost', 'logistic_regression']
    parser = argparse.ArgumentParser(description="Online IPVDetection argument parser.")
    parser.add_argument('-m', '--model', choices=CHOICES,  default = 'adaboost',
                        help = 'Type of model to run.')
    args = parser.parse_args()
    
    return args
    
def main():
    '''
    Main function that is called in the driver code.
    '''
    # From CLI.
    arguments = parse_arguments()
    
    # From YAML.
    config = parse_args_yaml(config_file = 'config.yaml')
    
    # Append to a single dictionary.
    configuration = dict(vars(arguments), **config)
    
    # Access the configuration values using dot notation.
    args = DotDict(configuration)
    
    # Instantiate logger object.
    logger = log_object(args)
    
    mymodel = InstantiateModel(args, logger).__get__()
    print(mymodel.__dict__)


if __name__=='__main__':
    '''
    Driver Code.
    '''
    main()
