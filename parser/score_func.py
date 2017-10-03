#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 05:21:42 2017

@author: raghuramkowdeed
"""

import numpy as np
import codecs

def read_text(file_name):
    try :
        with codecs.open(file_name,'rb',encoding='utf-8') as fin:
             text = fin.read()
             fin.close()
             return text   
    except IOError:
        print file_name + ' not found'
        return 0


def similarity_score_word_count(curr_file, prev_file,  use_ret = True):
        curr_text = read_text(curr_file)
        prev_text = read_text(prev_file)
        score = None
        msg = True
        
        if ( curr_text == 0 ) or ( prev_text == 0) :
            msg = False
            return None
        else :
            curr_words = get_words_from_string(curr_text)
            prev_words = get_words_from_string(prev_text)

            if len(prev_words ) ==0  or len(curr_words) == 0:
                msg = False
                return None
            else:
                score =  - ( abs( (len(prev_words)*1.0) - (len( curr_words)*1.0) ) )
                if use_ret:
                    score = score/len(prev_words)
                msg = True
                return score


def similarity_score_word_vec(curr_file, prev_file, metric_type='mse' ):
    #metric = ['corr', 'mse', 'dot']
    def get_metric(v1, v2, metric_type):
        val = None
        if metric_type == 'corr':
            val = np.corrcoef(v1, v2)[0,1]
        if metric_type == 'mse':
            val = -np.mean((v1 - v2)*(v1 - v2))
        if metric_type == 'dot':    
            val = np.dot(v1, v2)
        return val
    
    curr_vec = np.load(curr_file)
    prev_vec = np.load(prev_file)

    if curr_vec.shape[0] == 0 or prev_vec.shape[0] == 0 :
        return None
    
    try :
        v1 = curr_vec.mean(axis =0)
        v2 = prev_vec.mean(axis=0)
        score =  get_metric(v1, v2, metric_type)
        msg = True
        return score
    except:
        return None
                        