# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:44:35 2022

@author: Sagun Shakya
"""

import os
from os.path import exists, join

import pandas as pd

# Necessary libraries.
from numpy import intersect1d

# Local modules.
from .annotation_utils import convert_to_bio, get_file


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
    text = [tt for tt in text if tt != "\n"]
    text = text[3:]
    # print(text[:6])

    # Save.
    storage = []
    count = 0  # Sentence counter.
    for ii, line in enumerate(text, 1):
        if line.startswith("#Text"):
            count += 1
            continue
        else:
            splits = line.split()
            if len(splits) == 5:
                splits = splits[:3] + ["_", "_"] + splits[3:]
            storage.append(splits)

    # Define DataFrame.
    df = pd.DataFrame(
        storage, columns="s_no, str_id, token, ac, ap, conf, ipv".split(", ")
    )

    # Convert to BIO format.
    df["ac"] = convert_to_bio(df["ac"].to_list())

    # Replace "\\_" with "_" in ac.
    df["ac"] = df["ac"].apply(lambda x: x.replace("\\_", "_"))
    df["ac"] = df["ac"].apply(lambda x: "_" if (x == "B-*") or (x == "I-*") else x)
    # Fixing aspect polarity.
    df["ap"] = df["ap"].apply(lambda x: x[0] if x != "_" else x)
    df["ap"] = df["ap"].apply(lambda x: "_" if x == "*" else x)

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
    res = pd.concat(storage, axis=0, ignore_index=True)
    return res


def get_processed_data(
    shr_root: str, krn_root: str, get_common: bool = True
) -> pd.DataFrame:
    """
    Process all the exported data by both annotators into one.

    Args:
        shr_root (str): Directory holding the exported files by annotator 1.
        krn_root (str): Directory holding the exported files by annotator 2.
        get_common (bool, optional): If set to True, return the common file for both Annotators. Used for computing Inter-rated agreement. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing seven fields, namely, {s_no, str_id, token, ac, ap, conf, ipv}.
    """

    # Get the filenames of the exports in both directories.
    shr_files = os.listdir(shr_root)
    krn_files = os.listdir(krn_root)

    # Make sure that the directory is not empty.
    assert len(shr_files) > 1, "The directory shr is empty."
    assert len(krn_files) > 1, "The directory krn is empty."

    # For interannotator agreement.
    if get_common:
        # For agreement calculation, we need common files.
        target = intersect1d(shr_files, krn_files)
        print(f"\nNumber of files to inspect : {len(target)}\n")

        # Set the target filenames.
        shr_target_filenames = [os.path.join(shr_root, file) for file in target]
        krn_target_filenames = [os.path.join(krn_root, file) for file in target]

        df_shr_common = merge_annotations(shr_target_filenames)
        df_krn_common = merge_annotations(krn_target_filenames)

        return df_shr_common, df_krn_common

    # For getting dataframes.
    shr_filenames = [os.path.join(shr_root, file) for file in shr_files]
    krn_filenames = [os.path.join(krn_root, file) for file in krn_files]

    # Merge annotations for both.
    df_shr = merge_annotations(shr_filenames)
    df_krn = merge_annotations(krn_filenames)

    return df_shr, df_krn
