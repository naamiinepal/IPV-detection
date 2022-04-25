import numpy as np
import os
from os.path import join
from annotation.read_annotation import *
from annotation.agreement import PairwiseAgreement

def get_processed_data():
    # Done by Sharmila.
    shr_root = r'D:\ML_projects\IPV-Project\annotation\data\ipv\Sharmila'
    #krn_verified = r'D:\ML_projects\IPV-Project\annotation\data\CROSS\Kiran Cross checked files'
    
    # Done by Kiran.
    krn_root = r'D:\ML_projects\IPV-Project\annotation\data\ipv\kiran'
    #shr_verified = r'D:\ML_projects\IPV-Project\annotation\data\CROSS\Sharmila Cross checked files'

    shr_files = [file[:-4] for file in os.listdir(shr_root)]
    krn_files = [file[:-4] for file in os.listdir(krn_root)]

    # For agreement calculation.
    target = np.intersect1d(shr_files, krn_files)
    print(f'Number of files to inspect : {len(target)}\n')

    shr_target_filenames = [os.path.join(shr_root, file + '.tsv') for file in target]
    krn_target_filenames = [os.path.join(krn_root, file + '.tsv') for file in target]

    # For getting dataframes.
    shr_filenames = [os.path.join(shr_root, file) for file in os.listdir(shr_root)]
    krn_filenames = [os.path.join(krn_root, file) for file in os.listdir(krn_root)]

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
