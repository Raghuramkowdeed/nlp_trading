#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 05:48:48 2017

@author: raghuramkowdeed
"""

import urllib2, os
from bs4 import BeautifulSoup as BeautifulSoup
import pandas as pd
import numpy as np
#import datetime as dt
#import requests
import unicodedata
import re
import codecs

from multiprocessing import Pool
#from pathos.helpers import cpu_count
#from pathos.pools import ProcessPool

#for creating file with given path structure
def create_file(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    f = open(path, 'w')
    f.close()    
   
def html_to_text(html_file, text_file):
    create_file(text_file)
    with open(html_file) as f:
         data = f.read()
    f.close()     
    html = BeautifulSoup(data, 'html.parser')
    st = html.get_text()
    st_safe = st.encode('ascii','ignore')
    st_safe = st_safe.split('\n')
    
    st_clean = []
    
    for s in st_safe:
        this_s = s.strip() .lower() 
        if not  this_s.isdigit():
           if len(this_s) > 0 :
             if this_s[0] != '\n':
                   st_clean.append(  ( this_s ) + '\n' ) 
                   
    final_st = ''.join(st_clean)
    
    with open(text_file, 'w') as f:
         f.write(final_st) 
    f.close()

    
# Step 1: Define funtions to download filings

def get_10_k_links(cik_id, begin_year, end_year):

    cik_id = '0000'+cik_id
    base_url_part1 = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="+cik_id
    base_url_part2 = "&type=10-k&dateb="+str(end_year)
    base_url_part3 = "&owner=exclude&output=xml&count=100"
    base_url = base_url_part1 +  base_url_part2 +  base_url_part3
    href = []
    years = []
    href_dict = {}
       

    


    sec_page = urllib2.urlopen(base_url)
    sec_soup = BeautifulSoup(sec_page, "html.parser")
        
    filings = sec_soup.findAll('filing')
        
    for filing in filings:
        filing_date = filing.datefiled.get_text()
        report_year = int(filing_date[0:4])

        if (filing.type.get_text() == "10-K") :
            if (report_year >= begin_year) and  (report_year <= end_year):
                years.append(report_year)
                href.append(filing.filinghref.get_text())
                #href_dict[report_year] =  filing.filinghref.get_text()  
                href_dict[filing_date] =  filing.filinghref.get_text()  
    return href_dict


def process_text(text):
        """
            Preprocess Text
        """
        text = unicodedata.normalize("NFKD", text) # Normalize
        text = '\n'.join(text.splitlines()) # Let python take care of unicode break lines

        # Convert to upper
        text = text.upper() # Convert to upper

        # Take care of breaklines & whitespaces combinations due to beautifulsoup parsing
        text = re.sub(r'[ ]+\n', '\n', text)
        text = re.sub(r'\n[ ]+', '\n', text)
        text = re.sub(r'\n+', '\n', text)

        # To find MDA section, reformat item headers
        text = text.replace('\n.\n','.\n') # Move Period to beginning

        text = text.replace('\nI\nTEM','\nITEM')
        text = text.replace('\nITEM\n','\nITEM ')
        text = text.replace('\nITEM  ','\nITEM ')

        text = text.replace(':\n','.\n')

        # Math symbols for clearer looks
        text = text.replace('$\n','$')
        text = text.replace('\n%','%')

        # Reformat
        text = text.replace('\n','\n\n') # Reformat by additional breakline

        return text
    
def download_report(url_dict, dir_path):
    
    
    target_base_url = 'http://www.sec.gov'
    
    target_file_type = '10-K'
    
    for year, report_url in url_dict.iteritems():
#        html_file_path = dir_path + 'HTML/'+ str(year)
        text_file_path = dir_path + 'TEXT/'+ str(year)
        report_page = urllib2.urlopen(report_url)
        report_soup = BeautifulSoup(report_page, "html.parser")
        
        xbrl_file = report_soup.findAll('tr')
        
        for item in xbrl_file:
            try:
                if item.findAll('td')[3].get_text() == target_file_type:
                             
                    target_url = target_base_url + item.findAll('td')[2].find('a')['href']
                    
                 
                    r = urllib2.urlopen(target_url)
                    soup = BeautifulSoup( r, "html.parser" )
                    text = soup.get_text("\n") 
                    
                    text = process_text(text)
                    create_file(text_file_path)
                    
                    with codecs.open(text_file_path,'w',encoding='utf-8') as fout:
                        fout.write(text)
                    
            except:
                pass
def get_tickers_df():
    df = pd.read_csv("./cik_ticker.csv",  sep= "|" )
    return df


def download_10k_func(ticker, begin_year, end_year):
    df = get_tickers_df()
    x = df.loc[df['Ticker'] == ticker.upper(), ]
    cik_id = str(x.iloc[0,0])
    href_dict = get_10_k_links(cik_id, begin_year, end_year )
    dir_path = './data/10-K/'+ticker.upper()+"/"
    download_report(href_dict, dir_path)
            

def get_10k_reports(tickers, begin_year, end_year):
    df = get_tickers_df()
    
    for ticker in tickers:
        x = df.loc[df['Ticker'] == ticker.upper(), ]
        cik_id = str(x.iloc[0,0])
        href_dict = get_10_k_links(cik_id, begin_year, end_year )
        dir_path = './data/10-K/'+ticker.upper()+"/"
        download_report(href_dict, dir_path)
    

def get_10k_reports_parallel(tickers, begin_year, end_year):
#    begin_year = 2010
#    end_year = 2016
    
    def my_func(tic):
       return get_10k_reports([tic], begin_year=2010, end_year=2015)
 

    
    #ncpus = cpu_count() if cpu_count() <= 8 else 8;
    #pool = Pool( ncpus )
    p = Pool(processes = 4)
    result = p.map(my_func,tickers)
    return result

                
def extract_10k_item_1a(text_file):
  
    with open(text_file) as f:
        lines  = f.readlines()
        
    lines = np.array(lines)      
    lines_cl = [ x.strip() for x in lines ]
    lines_np = np.array(lines_cl)
    
    begin_words = [ 'item 1a',
                    'item 1a.', 
                    'item1a',
                    'item1a.',
                    'item.1a',
                    'item1.1a.',
                    'item1a.risk factors',
                    'item1a.risk factors.',
                    'item1a.risk',
                    'item1a.risk.',
                    'item1a. risk factors',
                    'item 1a. risk factors',
                  ]
    
    end_words   = [ 'item 1b',
                    'item 1b.',
                    'item1b',
                    'item1b.',
                    'item.1b',
                    'item.1b.',
                    'item 1b.unresolved',
                    'item 1b.unresolved.',
                    'item1b.unresolved',
                    'item1b.unresolved.',
                    'item1b.unresolved staff comments',
                    'item1b.unresolved staff comments.',
                    'item1b. unresolved staff comments.',
                    'item1b. unresolved',
                    'item1b. unresolved staff',
                   ]
    
    #denotes max num of characters item 1a heading can have
    max_char_1 = 22
    
    ind_h = [ i for i,s in enumerate(lines_cl) if len(s) < max_char_1 ]
    
    f_1 = lambda(t_s): reduce( lambda x,y: x |y, [ (i in t_s)  for i in begin_words ] )
    f_2 = lambda(t_s): reduce( lambda x,y: x |y, [ (i == t_s)  for i in end_words ] )
    f_3 = lambda(t_s): reduce( lambda x,y: x |y, [ (i in t_s)  for i in end_words ] )
    
    start_ind = [ i for i in ind_h if  f_1(lines_cl[i] ) ]
    print start_ind
    # if no line find with item 1a as heading, serach for it with in line content
    if len(start_ind) == 1 :
        new_start_ind = [ i for i in range(len(lines_cl)) 
                          if  f_1(lines_cl[i] ) & ( i > start_ind[0] ) ]
        start_ind.append(new_start_ind[0])
    start_line_num = start_ind[1]
    
    # check for line where item 1b occurs for the first time afer item 1a is found
    end_ind = [ i for i in range(len(lines_cl)) if  f_2(lines_cl[i] ) & (i > start_line_num) ]
    
    
    if(len(end_ind) == 0):
       end_ind = [ i for i in range(len(lines_cl)) if  f_3(lines_cl[i] ) & (i > start_line_num) ]
    
    end_line_num = [ i for i in end_ind if i > start_line_num][0]
     
    
    section_content = lines_np[start_line_num:(end_line_num+1)]
    return section_content
'''
this_dir =  "../data/10-K/AAPL/TEXT/"
file_names = os.listdir(this_dir)
file_names = [ i for i in file_names if not '.swp' in i]
content_dict = {}
for i in file_names:
    print i
    if dt.datetime.strptime(i, '%Y-%m-%d') > dt.datetime.strptime('2015-01-01', '%Y-%m-%d'):
       content = extract_10k_item_1a( this_dir+i)
       content_dict[i] = content


#content = extract_10k_item_1a("../data/10-K/GS/TEXT/2012-02-28")
'''