# -*- coding: utf-8 -*-
"""
Created on Wednesday Apr 13 09:44:35 2022

@author: Sagun Shakya
"""
import pandas as pd
import numpy as np

def preprocess_token_based(df: pd.DataFrame):
    """
    Takes in a dataframe containing all the annotations from an annotator and preprocess them.
    Steps:
        - Remove rows having "_" in their aspect category column.
        - Remove B- and I- prefixes from the category.

    Args:
        df (pd.DataFrame): Dataframe containing all the annotations from an annotator

    Returns:
        tuple: DataFrame containing the tokens with their ac values & the unique ac values.
    """    

    annot = df[['token', 'ac']]

    # Remove rows having "_" in their aspect category column.
    annot['ac'] = annot['ac'].apply(lambda x: np.nan if x == "_" else x)
    annot.dropna(inplace = True)
    annot.reset_index(drop=True, inplace=True)

    # Removing B and I prefix.
    annot['ac'] = annot['ac'].apply(lambda x: x[2:])   
    return annot, annot['ac'].unique()

def preprocess_instance_based(df: pd.DataFrame):
    """
    Takes in a dataframe containing all the annotations from an annotator and preprocess them.
    Steps:
        - Remove rows having "_" in their aspect category column.

    Args:
        df (pd.DataFrame): Dataframe containing all the annotations from an annotator

    Returns:
        tuple: DataFrame containing the tokens with their ac values & the unique ac values.
    """    

    annot = df[['token', 'ac']]

    # Remove rows having "_" in their aspect category column.
    annot['ac'] = annot['ac'].apply(lambda x: np.nan if x == "_" else x)
    annot.dropna(inplace = True)
    annot.reset_index(drop=True, inplace=True)

    # Preprocess.
    t = list(annot[annot['ac'].str.startswith('B-')].index)
    store = []
    for ii, id in enumerate(t):
        if ii <= len(t) - 2:
            ac = annot['ac'].iloc[id][2:]
            index = list(range(id, t[ii+1]))
            select = " ".join(annot['token'].iloc[index])
            store.append([select, ac])

    # To DataFrame.
    annot = pd.DataFrame(store, columns = ['at', 'ac'])
    return annot, annot['ac'].unique()

def f1_wrt_category(annot1: pd.DataFrame, 
                    annot2: pd.DataFrame, 
                    category: str, 
                    beta: float = 1.0) -> float:
    """
    Calculate pairwise agreement for a category using weighted F1 measure.

    Args:
        annot1 (pd.DataFrame): DataFrame Containing the tokens and their categories annotated by annotator 1.
        annot2 (pd.DataFrame): DataFrame Containing the tokens and their categories annotated by annotator 1.
        category (str): Aspect category w.r.t which the F1 measure is to be computed.
        beta (float, optional): Weights given to Precision. Defaults to 1.0.

    Returns:
        float: F1 Measure representing the pairwise agreement.
    """    
    assert category in annot1.ac.unique() and category in annot2.ac.unique()

    ## The tokens which are labelled as, say, 'profanity' by annotator A.
    a1 = annot1[annot1['ac'] == category]['token'].values

    ## The tokens which are labelled as, say, 'profanity' by annotator B.
    a2 = annot2[annot2['ac'] == category]['token'].values

    # Numerator.
    numerator = np.intersect1d(a1, a2).__len__()

    # Denomintor 1 -> Number of Tokens Annotator 1 tagged 'profanity'.
    denominator1 = len(a1)

    # Denomintor 2 -> Number of Tokens Annotator 2 tagged 'profanity'.
    denominator2 = len(a2)

    # Precision w.r.t Tag T.
    prec_t = numerator / denominator1

    # Recall w.r.t Tag T.
    rec_t = numerator / denominator2

    # F1 Measure -> Pairwise Inter-annotator agreement w.r.t Tag T.
    f1 = (2 * prec_t * rec_t) / (prec_t + rec_t)

    return f1

def calculate_agreement(df1: pd.DataFrame, df2: pd.DataFrame, type: str = 'token') -> dict:
    """
    Takes in the annotations from both annotators and gives the weighted F1 measure for all the aspect categories involved.
    Steps:
        - Preprocess the initial aspect category.
        - Obtain unique ac from both dfs.
        - Take union.
        - Obtain F1 measure for each category. 

    Args:
        df1 (pd.DataFrame): DataFrame containing all annotations from Annotator 1.
        df2 (pd.DataFrame): DataFrame containing all annotations from Annotator 1.
        type (str, optional): Can be one of {'instance', 'token'}. Defaults to 'token'.

    Returns:
        dict: Python Dictionary Containing Categories as keys and F1 measure as values.
    """    
    if type.lower().strip() == 'token':
        annot1, annot1_ac = preprocess_token_based(df1)
        annot2, annot2_ac = preprocess_token_based(df2)
    elif type.lower().strip() == 'instance':
        annot1, annot1_ac = preprocess_instance_based(df1)
        annot2, annot2_ac = preprocess_instance_based(df2)


    # Unique ac values.
    ac_unique = np.union1d(annot1_ac, annot2_ac)

    # Compute the F1 measure for each category and store them in a dictionary.    
    coll = {category : f1_wrt_category(annot1, annot2, category, beta=1.0) 
            for category in ac_unique}
    
    return coll