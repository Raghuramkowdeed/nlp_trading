#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 04:28:29 2017

@author: raghuramkowdeed
"""

import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas as pd
import os as os
import codecs
import numpy as np


def get_words_from_string(text):
    lines = text.split("\n")
    lines = [ x.strip() for x in lines if len(x) > 0]
    lines = [ x.split(" ") for x in lines if len(x) > 0]
                
    words = []
    for line in lines:
        for tw in line:
            words.append(tw)
    words = [ i for i in words if len(i) > 3]            
    return words
        


def get_section_files(tickers, sec, begin_date, end_date):
    years_range = []
    curr_year = dt.date(begin_date.year, 1,1)
    delim = ' '
    suffix = ''
    if 'VW' in sec:
        delim = '.'
        suffix = '.npy'
    while curr_year <= end_date :
        years_range.append(curr_year)
        curr_year += relativedelta(years=1)
       
    files_df = pd.DataFrame()
    
    for tic in tickers:
        curr_dir = '../data/10-K/'+ tic+'/'+ sec +'/'
        try :
            file_names = os.listdir(curr_dir)
            file_names = [ i for i in file_names if not '.swp' in i]
            file_names = [ i.split(delim)[0] for i in file_names ]
        except :
            file_names = []
        
        avail_files = []    
        for curr_year in years_range :
            search_res = [ x for x in file_names if dt.datetime.strptime(x, '%Y-%m-%d').year == curr_year.year]
            if len(search_res) > 0 :
                avail_files.append(curr_dir +str(search_res[0])+suffix)
            else:
                avail_files.append(None)
         
        files_df[tic] = avail_files   

    return files_df

def get_section_scores(tickers, sec, begin_date, end_date, score_func,lag =6 ):
    d1 = begin_date - relativedelta(years=1)
    files_df = get_section_files(tickers, sec, d1, end_date)
    score_df = pd.DataFrame()
    
    for tic in tickers:
        score_vec = []
        for i in range( 1, files_df.shape[0] ):
            score = None
            if files_df[tic].iloc[i] != None and files_df[tic].iloc[i-1] != None:
                score = score_func(files_df[tic].iloc[i], files_df[tic].iloc[i-1] )
                if score == np.nan:
                    score = None                    
            score_vec.append(score)   
        score_df[tic] = score_vec
    
    years_range = []
    curr_year = begin_date
    
    while curr_year <= end_date :
        years_range.append(curr_year+relativedelta(months=lag))
        curr_year += relativedelta(years=1)
       
    score_df.index = years_range
        
    return score_df    