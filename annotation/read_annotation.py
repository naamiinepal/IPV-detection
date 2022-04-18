# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:44:35 2022

@author: Sagun Shakya
"""

# Necessary libraries.
import pandas as pd
import copy
import os
from os.path import join, exists

# Local modules.
from .annotation_utils import get_file, convert_to_bio

# Path to data.
root = r'annotation\data'
annot1 = r'annotation\data\krn'
annot2 = r'annotation\data\shr'

example = 'batch_ipv_600-610 exported.tsv'

def parse_file(filename: str) -> pd.DataFrame:
    """
    Parses a WebAnno TSV v3 file to a DataFrame.
    Preprocessing involved:
        - Converts the aspect category annotations to BIO format.
        - Removes the sentence numbers associated with the annotations in square brackets. e.g. 1[8] --> 1.
        - Removes escape characters like "\".

    Args:
        filename (str): Path to the TSV file.

    Returns:
        pd.DataFrame: DataFrame with columns -- ['s_no, str_id, token, ac, ap, conf, ipv']
    """
    # Getting the exported TSV file.
    text = get_file(filename)
    text = [tt for tt in text if tt != '\n']
    text = text[3:]
    #print(text[:6])

    # Save.
    storage = []
    count = 0    # Sentence counter.
    for ii, line in enumerate(text, 1):
        if line.startswith('#Text'):
            count += 1
            continue
        else:
            storage.append(line.split())
    
    # Define DataFrame.
    df = pd.DataFrame(storage, columns = 's_no, str_id, token, ac, ap, conf, ipv'.split(", "))

    # Convert to BIO format.
    df['ac'] = convert_to_bio(df['ac'].to_list())

    # Replace "\\_" with "_" in ac.
    df['ac'] = df['ac'].apply(lambda x: x.replace('\\_', '_'))
    df['ac'] = df['ac'].apply(lambda x: '_' if (x =='B-*') or (x == 'I-*') else x)
    # Fixing aspect polarity.
    df['ap'] = df['ap'].apply(lambda x: x[0] if x != '_' else x)
    df['ap'] = df['ap'].apply(lambda x: '_' if x == '*' else x)

    return df

def merge_annotations(filename_list: list) -> pd.DataFrame:
    """
    Merges the dataframes generated from parse_file function.

    Args:
        filename_list (list): List of filepaths to be parsed.

    Returns:
        pd.DataFrame: DataFrame contaning all the annotations from the files provided.
    """    
    storage = [parse_file(file) for file in filename_list]
    res = pd.concat(storage, axis = 0).reset_index(drop = True)
    return res
