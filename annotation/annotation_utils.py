# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:26:12 2021

@author: Sagun Shakya
"""
import os
import re

def get_file(filename):
    '''
    Gets the file in list format given the filename.

    Parameters:
        filename -- Path of the file.
        
    Returns:
        list. Contents of WebAnno TSV v3 exported file. Separated by \n.
    '''
    with open(filename, encoding = 'utf-8') as ff:
        text = ff.readlines()
        ff.close()
        return text

def replace_BIO(asp: str, start: bool = True) -> str:
    '''
    If you pass in an aspect category in the form <aspect_category>[anynumber],
    this function will strip the square brakets and 'anynumber'.
    Then, it will add the prefix 'B-' (if it is the beginning of the tag) or
    'I-' (if it lies inside the tag).  
    
    Parameters:
        asp -- str, aspect category in the form <aspect_category>[anynumber].
        start -- Bool, Represents whether the tag is a 'begin' tag.
        
    Returns:
        str. Aspect category in BIO format.
    '''
    
    pattern = r'[\[0-9\]]+'
    asp_bio = re.sub(pattern, '', asp)
    if start:
        asp_bio = 'B-' + asp_bio
    else:
        asp_bio = 'I-' + asp_bio
        
    return asp_bio


def convert_to_bio(highlight):
    '''
    Mapping of keywords and aspects of WebAnno TSVv3 to BIO format.
    
    Parameters:
        highlight -- List of all the highlights in the form <highlight>[anynumber].
        
    Returns:
        List. highlights in BIO format.
    '''
    
    result = ['_']*len(highlight)
    
    for ii, (prev, curr) in enumerate(zip([None]+highlight[:-1], highlight)):
        if curr == '_':
            continue
        else:
            if prev == '_':
                begin = replace_BIO(curr, start = True)
                result[ii] = begin
                
            elif prev != curr and prev != '_':
                begin = replace_BIO(curr, start = True)
                result[ii] = begin
            
            elif prev == curr and prev != '_':
                inside = replace_BIO(curr, start = False)
                result[ii] = inside
                
    return result
