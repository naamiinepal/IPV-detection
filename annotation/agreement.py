# -*- coding: utf-8 -*-
"""
Created on Wednesday Apr 13 09:44:35 2022

@author: Sagun Shakya
"""
from warnings import filterwarnings

import numpy as np

#%%
import pandas as pd

filterwarnings("ignore")


class PairwiseAgreement:
    def __init__(
        self, df1: pd.DataFrame, df2: pd.DataFrame, type: str, beta: float = 1.0
    ) -> None:
        """
        Pairwise Inter-annotator agreement using Weighted F-measure.

        Args:
            df1 (pd.DataFrame): DataFrame containing all annotations from Annotator 1.
            df2 (pd.DataFrame): DataFrame containing all annotations from Annotator 1.
            type (str, optional): Can be one of {'instance', 'token'}. Defaults to 'token'.
            beta (float, optional): Weight given to Precision. Defaults to 1.0.
        """
        self.df1 = df1
        self.df2 = df2
        self.type = type
        self.beta = beta

    def calculate_agreement(self) -> dict:
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

        # Preprocessing step.
        annot1, annot1_ac = self._preprocess_tokens(self.df1, type=self.type)
        annot2, annot2_ac = self._preprocess_tokens(self.df2, type=self.type)

        # Unique ac values.
        ac_unique = np.intersect1d(annot1_ac, annot2_ac)

        # Compute the F1 measure for each category and store them in a dictionary.
        coll = {
            category: self.f1_wrt_category(annot1, annot2, category)
            for category in ac_unique
        }

        return coll

    def _preprocess_tokens(self, df: pd.DataFrame, type: str):
        """
        Takes in a dataframe containing all the annotations from an annotator and preprocess them.
        Steps:
            - Remove rows having "_" in their aspect category column.
            - If token based matching:
                - Remove B- and I- prefixes from the category.
            - Else if instance based matching:
                - Accumulate the tokens having B and I prefixes into a single string.
            - Return Tuple.

        Args:
            df (pd.DataFrame): Dataframe containing all the annotations from an annotator

        Returns:
            tuple: DataFrame containing the tokens with their ac values & the unique ac values.
        """
        assert type in [
            "token",
            "instance",
        ], "Type should be one of {'token', 'instance'}."

        annot = df[["token", "ac"]]

        # Remove rows having "_" in their aspect category column.
        annot["ac"] = annot["ac"].apply(lambda x: np.nan if x == "_" else x)
        annot.dropna(inplace=True)
        annot.reset_index(drop=True, inplace=True)

        if type == "token":
            # Removing B and I prefix.
            annot["ac"] = annot["ac"].apply(lambda x: x[2:])

        else:
            # Accumulate the B and I tokens into a single string.
            t = list(annot[annot["ac"].str.startswith("B-")].index)
            store = []
            for ii, id in enumerate(t):
                if ii <= len(t) - 2:
                    ac = annot["ac"].iloc[id][2:]
                    index = list(range(id, t[ii + 1]))
                    select = " ".join(annot["token"].iloc[index])
                    store.append([select, ac])

            # To DataFrame.
            annot = pd.DataFrame(store, columns=["token", "ac"])

        return annot, annot["ac"].unique()

    def f1_wrt_category(
        self, annot1: pd.DataFrame, annot2: pd.DataFrame, category: str
    ) -> float:
        """
        Calculate pairwise agreement for a category using weighted F1 measure.

        Args:
            annot1 (pd.DataFrame): DataFrame Containing the tokens and their categories annotated by annotator 1.
            annot2 (pd.DataFrame): DataFrame Containing the tokens and their categories annotated by annotator 2.
            category (str): Aspect category w.r.t which the F1 measure is to be computed.

        Returns:
            float: F1 Measure representing the pairwise agreement.
        """
        # assert category in annot1.ac.unique() and category in annot2.ac.unique(), f'The category {category} does not exist in either of the annotations.'

        ## The tokens which are labelled as, say, 'profanity' by annotator A.
        a1 = annot1[annot1["ac"] == category]["token"].values

        ## The tokens which are labelled as, say, 'profanity' by annotator B.
        a2 = annot2[annot2["ac"] == category]["token"].values

        # Numerator.
        numerator = np.intersect1d(a1, a2).__len__()

        # Denomintor 1 -> Number of Tokens Annotator 1 tagged 'profanity'.
        denominator1 = len(a1)

        # Denomintor 2 -> Number of Tokens Annotator 2 tagged 'profanity'.
        denominator2 = len(a2)

        # Precision w.r.t Tag T.
        prec_t = numerator / denominator1 if denominator1 > 0 else np.inf

        # Recall w.r.t Tag T.
        rec_t = numerator / denominator2 if denominator2 > 0 else np.inf

        # F1 Measure -> Pairwise Inter-annotator agreement w.r.t Tag T.
        f1 = (
            ((1 + self.beta**2) * prec_t * rec_t)
            / ((self.beta**2 * prec_t) + rec_t)
            if prec_t > 0 and rec_t > 0
            else 0.0
        )

        return f1
