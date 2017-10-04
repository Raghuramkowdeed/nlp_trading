#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 05:21:42 2017

@author: raghuramkowdeed
"""

import pandas as pd
import numpy as np
import codecs
#from matplotlib.mlab import PCA
from sklearn.decomposition import PCA

from nltk.corpus import stopwords


stop = set(stopwords.words('english'))

positive = pd.read_csv('../data/positive-words.txt', names=['a'])
positive =  set(positive['a'].tolist())

negative = pd.read_csv('../data/negative-words.txt', names=['a'], encoding='latin-1')
negative =  set(negative['a'].tolist())

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



def from_text_to_clean(tex):
    out_tex = [word.lower() for word in tex if word not in stop]
    out_tex = [word for word in out_tex if '$' not in word]
    out_tex = [word for word in out_tex if word.replace(',','',1).isdigit()==False]
    out_tex = [word for word in out_tex if word.replace('.','',1).isdigit()==False]
    out_tex = [word for word in out_tex if '-k' not in word]
    out_tex = [word for word in out_tex if '%' not in word]
    out_tex = [word for word in out_tex if 'www' not in word]
    return out_tex


def sentimental_score(curr_file, prev_file):
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
                
                commonp_prev = len(set(prev_words ).intersection(positive))
                commonn_prev = len(set(prev_words ).intersection(negative))
                
                commonp_curr = len(set(curr_words).intersection(positive))
                commonn_curr = len(set(curr_words).intersection(negative))
                
                score_previous= commonp_prev - commonn_prev
                score_curr= commonp_curr - commonn_curr
                score = (score_curr - score_previous)*1.0

                msg = True
                return score




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
                        
def similarity_score_pca_word_vec(curr_file, prev_file, metric_type='mse' , n_comp = 10):
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
        pca = PCA(n_components=n_comp)
        #all_data = np.concatenate((curr_vec, prev_vec), axis = 0)
        pca.fit(curr_vec)
        v1 = pca.transform(curr_vec).mean(axis=0)
        
        pca = PCA(n_components=n_comp)
        pca.fit(prev_vec)
        v2 = pca.transform(prev_vec).mean(axis=0)
        score =  get_metric(v1, v2, metric_type)

        return score
    except:
        return None    