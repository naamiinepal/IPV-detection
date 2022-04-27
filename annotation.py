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

def get_agreement(df_shr, df_krn):
    shr_agreement = PairwiseAgreement(df_shr, df_krn, 'token', 1.0)
    agreement_dict = shr_agreement.calculate_agreement()
    df = pd.DataFrame(agreement_dict, index=[0]).T.round(3)
    df.to_csv('agreement-non-ipv.csv')
    print(agreement_dict)

if __name__ == "__main__":
    df_shr, df_krn = get_processed_data(shr_root, krn_root, get_common=True)
    get_agreement(df_shr, df_krn)