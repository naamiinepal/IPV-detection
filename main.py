# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:44:18 2022

@author: Sagun Shakya
"""
import os
import argparse
import random
from numpy import random as rdm
import torch

# Local Modules.
from dataloader.dl_dataloader import Dataloader
from trainer.ml_trainer import MLTrainer
from utilities import utils
from utilities.read_configuration import DotDict

# Determinism.
SEED = 1234
random.seed(SEED)
rdm.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def parse_arguments():
    '''
    Argument parser for IPV project.

    Returns
    -------
    args : TYPE
        Arguments for the run.

    '''
    MODEL_CHOICES = ['svm', 'nb', 'random_forest', 'adaboost', 'logistic_regression'] + ['lstm', 'gru', 'cnn', 'mbert']
    parser = argparse.ArgumentParser(description="Online IPV Detection argument parser.")

    parser.add_argument('-m', '--model', choices = MODEL_CHOICES,  default = 'adaboost',
                        help = 'Type of model to run.')
    parser.add_argument('-t', '--train_type', choices = ['text', 'atsa', 'acsa', 'concat'], default = 'text',
                        help = 'Type of training: Should be one of {"text : text only", "atsa : text + at", "acsa : text + ac", "concat : text + at + ac"}.')
    
    args = parser.parse_args()
    
    return args
    
def main():
    '''
    Main function called in the driver code.
    '''

    # From CLI.
    arguments = parse_arguments()
    
    # From YAML.
    config = utils.parse_args_yaml(config_file = 'config.yaml')
    
    # Append to a single dictionary.
    configuration = dict(vars(arguments), **config)
    
    # Access the configuration values using dot notation.
    args = DotDict(configuration)
    
    # Instantiate logger object.
    logger = utils.log_object(args)
    
    # Whether to use CUDA.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f'\nRunning on CUDA : {use_cuda}\n')   

    # Train ML Classifier.
    #ml_trainer = MLTrainer(args, logger)
    #ml_trainer.train()

    data_loader = Dataloader(args, 1, device)
    train_dl, val_dl = data_loader.load_data(args.batch_size)
    temp = next(iter(train_dl))
    print(temp.TEXT)

if __name__=='__main__':
    '''
    Driver Code.
    '''
    main()
