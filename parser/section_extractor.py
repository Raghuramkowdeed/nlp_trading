#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:49:21 2017

@author: raghuramkowdeed
"""

from bs4 import BeautifulSoup as soup
import numpy as np
import os
import codecs

#for creating file with given path structure
def create_file(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
       os.makedirs(dir)

    f = open(path, 'w')
    f.close()    



def parse_mda(text, start=0):
        debug = False
        """
            Return Values
        """

        mda = ""
        end = 0

        """
            Parsing Rules
        """

        # Define start & end signal for parsing
        item7_begins = [ '\nITEM 7.', '\nITEM 7 –','\nITEM 7:', '\nITEM 7 ', '\nITEM 7\n' ]
        item7_ends   = [ '\nITEM 7A' ]
        if start != 0:
            item7_ends.append('\nITEM 7') # Case: ITEM 7A does not exist
        item8_begins = [ '\nITEM 8'  ]

        """
            Parsing code section
        """
        text = text[start:]

        # Get begin
        for item7 in item7_begins:
            begin = text.find(item7)
            if debug:
                print(item7,begin)
            if begin != -1:
                break

        if begin != -1: # Begin found
            for item7A in item7_ends:
                end = text.find(item7A, begin+1)
                if debug:
                    print(item7A,end)
                if end != -1:
                    break

            if end == -1: # ITEM 7A does not exist
                for item8 in item8_begins:
                    end = text.find(item8, begin+1)
                    if debug:
                        print(item8,end)
                    if end != -1:
                        break

            # Get MDA
            if end > begin:
                mda = text[begin:end].strip()
            else:
                end = 0

        return mda, end
    
def parse_1a(text, start=0):
        debug = False
        """
            Return Values
        """

        mda = ""
        end = 0

        """
            Parsing Rules
        """

        # Define start & end signal for parsing
        item7_begins = [ '\nITEM 1A.', '\nITEM 1A –','\nITEM 1A:', '\nITEM 1A ', '\nITEM 1A\n' ]
        item7_ends   = [ '\nITEM 1B' ]
        if start != 0:
            item7_ends.append('\nITEM 1A') # Case: ITEM 7A does not exist
        item8_begins = [ '\nITEM 2'  ]

        """
            Parsing code section
        """
        text = text[start:]

        # Get begin
        for item7 in item7_begins:
            begin = text.find(item7)
            if debug:
                print(item7,begin)
            if begin != -1:
                break

        if begin != -1: # Begin found
            for item7A in item7_ends:
                end = text.find(item7A, begin+1)
                if debug:
                    print(item7A,end)
                if end != -1:
                    break

            if end == -1: # ITEM 7A does not exist
                for item8 in item8_begins:
                    end = text.find(item8, begin+1)
                    if debug:
                        print(item8,end)
                    if end != -1:
                        break

            # Get MDA
            if end > begin:
                mda = text[begin:end].strip()
            else:
                end = 0

        return mda, end


def parsing_job_mda(ticker,fname):
        print("Parsing: {}".format(fname))
        # Read text
        txt_dir = './data/10-K/' +ticker + '/TEXT/' 
        filepath = os.path.join(txt_dir,fname)
        with codecs.open(filepath,'rb',encoding='utf-8') as fin:
            text = fin.read()

        text = text.encode("utf-8")
        #text=text.encode('ascii', 'ignore').decode('ascii')
        name, ext = os.path.splitext(fname)
        # Parse MDA part

        msg = ""
        mda, end = parse_mda(text)
        # Parse second time if first parse results in index
        if mda and len(mda) < 1000:
        #if mda and len(mda.encode('utf-8')) < 1000:
            mda, _ = parse_mda(text, start=end)

        if mda: # Has value
            msg = "SUCCESS"
            #mda_path = os.path.join(self.mda_dir, name + '.mda')
            #with codecs.open(mda_path,'w', encoding='utf-8') as fout:
            #    fout.write(mda)
        else:
            msg = msg if mda else "MDA NOT FOUND"
        #print("{},{}".format(name,msg))
        #return name + '.txt', msg #
        return msg, mda

def parsing_job_1a(ticker,fname):
        print("Parsing: {}".format(fname))
        # Read text
        txt_dir = './data/10-K/' +ticker + '/TEXT/' 
        filepath = os.path.join(txt_dir,fname)
        with codecs.open(filepath,'rb',encoding='utf-8') as fin:
            text = fin.read()

        text = text.encode("utf-8")
        #text=text.encode('ascii', 'ignore').decode('ascii')
        name, ext = os.path.splitext(fname)
        # Parse MDA part

        msg = ""
        mda, end = parse_1a(text)
        # Parse second time if first parse results in index
        if mda and len(mda) < 1000:
        #if mda and len(mda.encode('utf-8')) < 1000:
            mda, _ = parse_1a(text, start=end)

        if mda: # Has value
            msg = "SUCCESS"
            #mda_path = os.path.join(self.mda_dir, name + '.mda')
            #with codecs.open(mda_path,'w', encoding='utf-8') as fout:
            #    fout.write(mda)
        else:
            msg = msg if mda else "MDA NOT FOUND"
        #print("{},{}".format(name,msg))
        #return name + '.txt', msg #
        return msg, mda

def check_ticker_1a(ticker):
    curr_dir = './data/10-K/'+ticker+'/TEXT/'
    file_names = os.listdir(curr_dir)
    file_names = [ i for i in file_names if not '.swp' in i]
    d = []
    for i,fn in enumerate(file_names):
        msg, mda  = parsing_job_1a(ticker,fn)
        print msg, len(mda)
        d.append (mda)
    return d    
    
   