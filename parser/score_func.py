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


np.random.seed(42)

stop = set(stopwords.words('english'))

positive = pd.read_csv('../data/positive-words.txt', names=['a'])
positive =  set(positive['a'].tolist())

negative = pd.read_csv('../data/negative-words.txt', names=['a'], encoding='latin-1')
negative =  set(negative['a'].tolist())


macdo=pd.read_excel('../data/LoughranMcDonald_MasterDictionary_2014.xlsx')
macdo['Negative-bis']=macdo['Negative'].apply(lambda x : 1 if x!=0 else 0)
macdo['Positive-bis']=macdo['Positive'].apply(lambda x : 1 if x!=0 else 0)
negative_macdo=set([str(worr).lower() for worr in macdo[macdo['Negative-bis']==1]['Word']])
positive_macdo=set([str(worr).lower() for worr in macdo[macdo['Positive-bis']==1]['Word']])


#positive = pd.read_csv('../data/positive-words.txt', names=['a'])
#positive =  set(positive['a'].tolist())
#positive = np.unique(positive)

#negative = pd.read_csv('../data/negative-words.txt', names=['a'], encoding='latin-1')
#negative =  set(negative['a'].tolist())
#negative = np.unique(negative)

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

def clean_text(text):
    import re
    from string import digits
    clean_text = re.sub(r'[?|$|!|,|(|)|[|]]',r'',text)
    r=re.compile(r'\d')
    clean_text = r.sub('', clean_text)
    clean_text = re.sub(r'[-|:|%|;|.]',r'',clean_text)
    clean_text = re.sub(r'[(|)]',r'',clean_text)
    clean_text = re.sub(r"http\S*", '', clean_text)
    clean_text = clean_text.lower()
    return clean_text

def from_text_to_clean_2(tex):
    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))
    out_tex = []
    for word in tex:
        clean_word = clean_text(word)
        if (clean_word != "") & (clean_word not in stop) & (clean_word != 'k'):
            out_tex.append(clean_word)
    out_tex = [word for word in out_tex if 'www' not in word]
    return out_tex

def from_unicode_to_string(text):
    import unicodedata
    curr_words_str = []
    for word in text:
        curr_words_str.append(word.encode('ascii','ignore'))
    return curr_words_str



def from_text_to_clean(tex):
    out_tex = [word.lower() for word in tex if word not in stop]
    out_tex = [word for word in out_tex if '$' not in word]
    out_tex = [word for word in out_tex if word.replace(',','',1).isdigit()==False]
    out_tex = [word for word in out_tex if word.replace('.','',1).isdigit()==False]
    out_tex = [word for word in out_tex if '-k' not in word]
    out_tex = [word for word in out_tex if '%' not in word]
    out_tex = [word for word in out_tex if 'www' not in word]
    return out_tex


def sentimental_score(curr_file, prev_file,method_sent='macdo'):
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
                
                curr_words = from_text_to_clean(curr_words)
                prev_words = from_text_to_clean(prev_words)
                
                if method_sent == 'classic':
                    
                    commonp_prev = len(set(prev_words ).intersection(positive))
                    commonn_prev = len(set(prev_words ).intersection(negative))
                
                    commonp_curr = len(set(curr_words).intersection(positive))
                    commonn_curr = len(set(curr_words).intersection(negative))
 
                if method_sent == 'macdo':
                    
                    commonp_prev = len(set(prev_words ).intersection(positive_macdo))
                    commonn_prev = len(set(prev_words ).intersection(negative_macdo))
                
                    commonp_curr = len(set(curr_words).intersection(positive_macdo))
                    commonn_curr = len(set(curr_words).intersection(negative_macdo))
                    
                score_previous= commonp_prev - commonn_prev
                score_curr= commonp_curr - commonn_curr
                score = 100.0*(score_curr - score_previous)*1.0/(0.5*len(prev_words )+0.5*len(curr_words))

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

def LDA_similar_topics_score(curr_file, prev_file,  use_ret = True):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    
    
    
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
            curr_text_token = get_words_from_string(curr_text)
            prev_text_token = get_words_from_string(prev_text)
            curr_text_token = from_text_to_clean_2(curr_text_token)
            prev_text_token = from_text_to_clean_2(prev_text_token)
            
            l1 = len(curr_text_token)
            l2 = len(prev_text_token)
            
            curr_text_token = from_unicode_to_string(curr_text_token)
            curr_text_assembled = [', '.join(curr_text_token)]
            prev_text_token = from_unicode_to_string(prev_text_token)
            prev_text_assembled = [', '.join(prev_text_token)]
            total_text_token = curr_text_token + prev_text_token
            total_text_assembled = [', '.join(total_text_token)]
            
            n_features = 1000
            n_components = 20
            n_top_words = 25

            vectorizer = TfidfVectorizer(max_features=n_features,  stop_words='english')
            #vectorizer = TfidfVectorizer( stop_words='english')
            tfidf = vectorizer.fit_transform(total_text_assembled)
            

            lda = LatentDirichletAllocation(learning_method='online')
            lda.fit(tfidf)
            
            tfidf_prev = vectorizer.fit_transform(prev_text_assembled)
            tfidf_curr = vectorizer.fit_transform(curr_text_assembled)
            
            topic_distrib_prev = np.array(lda.transform(tfidf_prev)[0])
            topic_distrib_curr = np.array(lda.transform(tfidf_curr)[0])
            
            score = 0
            for k in range(len(topic_distrib_prev)):
            
                try:
                    mot = topic_distrib_prev[k]*np.log(topic_distrib_prev[k]/topic_distrib_curr[k])
                 
                except:
                    print "bug"
            score = score + mot
            score = -1.0*score
            
            return score
    
def lda_score(curr_file, prev_file):
    s = 0.0
    try :
        s = LDA_similar_topics_score(curr_file, prev_file,  use_ret = True)
    except : 
        s = 0.0
    return s    
        
        
