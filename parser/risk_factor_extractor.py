#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 05:48:48 2017

@author: raghuramkowdeed
"""
from bs4 import BeautifulSoup as soup
import numpy as np
import os
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

