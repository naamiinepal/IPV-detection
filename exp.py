import numpy as np
import os
from os.path import join

from pandas import DataFrame
from annotation.read_annotation import *
from annotation.agreement import PairwiseAgreement

# Constants.
root = r'D:\ML_projects\IPV-Project\annotation\data\first_lot\non-ipv'

# Done by Sharmila.
shr_root = join(root, 'shr')
# Done by Kiran.
krn_root = join(root, 'krn')

def get_processed_data(shr_root: str, krn_root: str) -> DataFrame:

    # Get the filenames of the exports in both directories.
    shr_files = os.listdir(shr_root)
    krn_files = os.listdir(krn_root)

    # Make sure that the directory is not empty.
    assert len(shr_files) > 1, "The directory shr is empty."
    assert len(krn_files) > 1, "The directory krn is empty."
    
    # For agreement calculation, we need common files.
    target = np.intersect1d(shr_files, krn_files)
    print(f'\nNumber of files to inspect : {len(target)}\n')

    # Set the target filenames.
    shr_target_filenames = [os.path.join(shr_root, file) for file in target]
    krn_target_filenames = [os.path.join(krn_root, file) for file in target]

    # For getting dataframes.
    shr_filenames = [os.path.join(shr_root, file) for file in shr_files]
    krn_filenames = [os.path.join(krn_root, file) for file in krn_files]

    df_shr = merge_annotations(shr_filenames)
    df_krn = merge_annotations(krn_filenames)

    return df_shr, df_krn
    

def get_agreement(df_shr, df_krn):
    shr_agreement = PairwiseAgreement(df_shr, df_krn, 'token', 1.0)
    agreement_dict = shr_agreement.calculate_agreement()
    df = pd.DataFrame(agreement_dict, index=[0]).T.round(3)
    df.to_csv('agreement1.csv')
    print(agreement_dict)

if __name__ == "__main__":
    df_shr, df_krn = get_processed_data()
    df_shr.to_csv(os.path.join(r"annotation\data\aspect_extraction_sample", 'shr_asp.csv'), index = None, encoding='utf-8')
    df_krn.to_csv(os.path.join(r"annotation\data\aspect_extraction_sample", 'krn_asp.csv'), index = None, encoding='utf-8')
