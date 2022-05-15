from doctest import OutputChecker
import numpy as np
import os
from os.path import join
from pandas import DataFrame
import argparse

from yaml import parse

# Local Modules.
from annotation.read_annotation import *
from annotation.agreement import PairwiseAgreement

# Constants.
#root = r'D:\ML_projects\IPV-Project\annotation\data\first_lot\non-ipv'
root = r'D:\ML_projects\IPV-Project\annotation\data\first_lot\common_annotation\Exported'

def parse_arguments():
    '''
    Argument parser for inter-annotator agreement.

    Returns
    -------
    args : TYPE
        Arguments for the run.

    '''
    parser = argparse.ArgumentParser(description="Online IPV Detection argument parser.")

    parser.add_argument('-i', '--input_dir', type = str, default = r'D:\ML_projects\IPV-Project\annotation\data\first_lot\common_annotation\Exported',
                        help = 'Directory where the exported files for annotator 1 and 2 and located (in separate folders).')   
    parser.add_argument('-o', '--output_dir', type = str, default = r'annotation/results/inter_annotator_agreement',
                        help = 'Path to the directory to store the results of the IA agreement.')
    parser.add_argument('-f', '--save_filename', type = str, default = 'ia_agreement.csv', 
                        help = 'Name of the file to be created to store the IA agreement for each aspect category.')
    args = parser.parse_args()
    
    return args


def get_agreement(df_shr, df_krn, filepath):
    shr_agreement = PairwiseAgreement(df_shr, df_krn, 'token', 1.0)
    agreement_dict = shr_agreement.calculate_agreement()
    agreement_df = pd.DataFrame(agreement_dict, index=[0]).T.round(3)

    # Save the results
    agreement_df.to_csv(filepath)
    print(agreement_dict)
    print(f'Results saved as {filepath}.')

if __name__ == "__main__":
    '''
    Driver Code.
    '''
    # Parse Arguments.
    args = parse_arguments()
    
    # Create directory for output file (if not already exists).
    os.makedirs(args.output_dir, exist_ok = True)

    # Done by Sharmila.
    shr_root = join(args.input_dir, 'shr')
    assert exists(shr_root), f'The path {shr_root} does not exist.'
    
    # Done by Kiran.
    krn_root = join(args.input_dir, 'krn')
    assert exists(krn_root), f'The path {krn_root} does not exist.'

    # Get dataframes of both A1 and A2.
    df_shr, df_krn = get_processed_data(shr_root, krn_root, get_common=True)

    # Get and store results.
    get_agreement(df_shr, df_krn, filename = args.save_filename)