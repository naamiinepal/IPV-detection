import argparse
import os
from utilities import utils
from utilities.kfold_split_asp_extraction import kfold_split_aspect
#from utilities.read_configuration import DotDict

def argument_parser():
    parser = argparse.ArgumentParser(description = "Argument Parser for Creating K fold splits fr aspect term extraction.")
    parser.add_argument('-i', '--input_dir', default = r'data/aspect_extraction', type = str, metavar = 'PATH',
                        help = 'Path to raw data directory.')
    parser.add_argument('-o', '--save_dir', default = r'data/aspect_extraction/kfold', type = str, metavar='PATH',
                        help = 'Path to the data directory to store K fold splits.')
    parser.add_argument('--log_dir', default = r'logs/splits', type = str, metavar = 'PATH',
                        help = 'Directory to save logs.')
    parser.add_argument('-k', '--kfolds', default=5, type=int, 
                        help = 'Number of folds to generate.')                    
    parser.add_argument('-r', '--random_seed', default=1234, type=int, 
                        help = 'Random seed.')
    parser.add_argument('-v', '--verbose', action = 'store_true', 
                        help = 'Whether to display verbose.')
    args = parser.parse_args()
    return args

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
    os.makedirs(args.log_dir, exist_ok = True)

    name_ = f"asp_extraction_{args.kfolds}_fold_split.log"
    log_file = os.path.join(args.log_dir, name_)

    # Intialize Logger.
    logger = utils.get_logger(log_file)
    return logger
    
if __name__ == "__main__":
    args = argument_parser()
    logger = log_object(args)
    kfold_split_aspect(args, logger)