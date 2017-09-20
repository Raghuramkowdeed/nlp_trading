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
import datetime as dt

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
def get_10_k_links(ticker, begin_year):

    base_url_part1 = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
    base_url_part2 = "&type=&dateb=&owner=&start="
    base_url_part3 = "&count=100&output=xml"
    href = []
    years = []
    href_dict = {}
       
    for page_number in range(0,2000,100):
    
        base_url = base_url_part1 + ticker + base_url_part2 + str(page_number) + base_url_part3

        sec_page = urllib2.urlopen(base_url)
        sec_soup = BeautifulSoup(sec_page)
        
        filings = sec_soup.findAll('filing')
        
        for filing in filings:
            report_year = int(filing.datefiled.get_text()[0:4])
            if (filing.type.get_text() == "10-K") & (report_year < begin_year):

                years.append(report_year)
                href.append(filing.filinghref.get_text())
                href_dict[report_year] =  filing.filinghref.get_text()  
    return href_dict

def get_10_k_links_2(cik_id, begin_year, end_year):

    cik_id = '0000'+cik_id
    base_url_part1 = "http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="+cik_id
    base_url_part2 = "&type=10-k&dateb="+str(end_year)
    base_url_part3 = "&owner=exclude&output=xml&count=100"
    base_url = base_url_part1 +  base_url_part2 +  base_url_part3
    href = []
    years = []
    href_dict = {}
       

    


    sec_page = urllib2.urlopen(base_url)
    sec_soup = BeautifulSoup(sec_page)
        
    filings = sec_soup.findAll('filing')
        
    for filing in filings:
        filing_date = filing.datefiled.get_text()
        report_year = int(filing_date[0:4])

        if (filing.type.get_text() == "10-K") & (report_year >= begin_year):
                years.append(report_year)
                href.append(filing.filinghref.get_text())
                #href_dict[report_year] =  filing.filinghref.get_text()  
                href_dict[filing_date] =  filing.filinghref.get_text()  
    return href_dict


def download_report(url_dict, dir_path):
    
    
    target_base_url = 'http://www.sec.gov'
    
    target_file_type = '10-K'
    
    for year, report_url in url_dict.iteritems():
        html_file_path = dir_path + 'HTML/'+ str(year)
        text_file_path = dir_path + 'TEXT/'+ str(year)
        report_page = urllib2.urlopen(report_url)
        report_soup = BeautifulSoup(report_page)
        
        xbrl_file = report_soup.findAll('tr')
        
        for item in xbrl_file:
            try:
                if item.findAll('td')[3].get_text() == target_file_type:
                             
                    target_url = target_base_url + item.findAll('td')[2].find('a')['href']
                    

                    create_file(html_file_path)

                   
                    xbrl_report = urllib2.urlopen(target_url)
                    output = open(html_file_path,'wb')
                    output.write(xbrl_report.read())
                    output.close()
                    
                    html_to_text(  html_file_path, text_file_path )
                    
            except:
                pass
def get_tickers_df():
    df = pd.read_csv("./data/cik_ticker.csv",  sep= "|" )
    return df


            
def get_10k_reports(tickers, begin_year, end_year):
    df = get_tickers_df()
    for ticker in tickers :
        x = df.loc[df['Ticker'] == ticker.upper(), ]
        cik_id = str(x.iloc[0,0])
        href_dict = get_10_k_links_2(cik_id, begin_year, end_year )
        dir_path = './data/10-K/'+ticker.upper()+"/"
        download_report(href_dict, dir_path)

                

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