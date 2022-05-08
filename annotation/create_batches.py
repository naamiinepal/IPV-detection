# -*- coding: utf-8 -*-
"""
Created on Wed May 05 10:26:12 2021

@author: Sagun Shakya
"""

# Importing necessary libraries.

import re
from string import punctuation
import pandas as pd
import numpy as np
import os
from os.path import join, exists
from warnings import filterwarnings

from tqdm import tqdm
filterwarnings('ignore')

# For Twitter Scraped data.
def create_batches_twitter(root_dir, filename, save_dir):
    # Load data.
    filepath = join(root_dir, filename)

    if filename.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filepath, encoding = 'utf-8', skip_blank_lines = True)
    
    # Make sure the tweets are not already included in the master data file "overall.xlsx".
    df_overall = pd.read_excel(r'D:\ML_projects\IPV-Project\annotation\data\overall.xlsx')

    # Accumulate already annotated tweets links.
    df_overall_link = df_overall[df_overall['source'] == 'twitter']['links']
    df_link = df['link']
    common_tweets = np.intersect1d(df_overall_link, df_link)

    # Verbose.
    print(f"Found {len(common_tweets)} tweets which have already been annotated.")
    print('Excluding these tweets for creating batches...')
    print(common_tweets)
    print()

    # Delete rows.
    df.drop(df[df['link'].isin(common_tweets)].index, inplace = True)
    df.reset_index(drop = True, inplace = True)

    # Clean Text.
    text = df['text']
    
    ## Remove punctuations.
    other_punctuations = '।‘’' + '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~' + chr(8211)
    to_remove = punctuation + other_punctuations + chr(8226)     
    text = text.str.translate(str.maketrans('', '', to_remove))

    ## Remove multiple whitespaces + strip whitespaces at the extremes.
    text = text.apply(lambda x: re.sub(' +', ' ', x))
    text = text.str.strip()

    ## Add sentence enders.
    enders = ['|', '?', '।', '.', '!']
    text = text.apply(lambda x : x + " |" if x[-1] not in enders else x)

    # Save batches.
    save_dir_shr = join(save_dir, 'shr')
    save_dir_krn = join(save_dir, 'krn')

    os.makedirs(save_dir_shr, exist_ok = True)
    os.makedirs(save_dir_krn, exist_ok = True)

    for k, ii in tqdm(enumerate(range(0, len(text), 10), 1)):
        start = ii
        end = ii + 10
        batch = text.loc[start:end].str.strip()

        save_filename = f'batch_twitter2_{start}-{end}.txt'

        # Distribute batches to annotators.
        if k%2 == 0:
            batch.to_csv(join(save_dir_shr, save_filename), header = None, index = None, encoding = 'utf-8')
        else:
            batch.to_csv(join(save_dir_krn, save_filename), header = None, index = None, encoding = 'utf-8')

if __name__ == "__main__":
    create_batches_twitter(root_dir=r'D:\ML_projects\IPV-Scraper\results\second_lot',
                            filename='df_merged_second_lot_search_terms_cleaned_05-May-2021_nep.xlsx',
                            save_dir=r'D:\ML_projects\IPV-Scraper\results\second_lot\sample_folder1')
