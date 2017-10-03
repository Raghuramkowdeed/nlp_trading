#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 02:00:28 2017

@author: raghuramkowdeed
"""
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import os
from dateutil.relativedelta import relativedelta


def create_file(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
       os.makedirs(dir)

    f = open(path, 'w')
    f.close()    

def store_prices( tickers, begin_date = dt.date(2001,1,1), end_date = dt.date(2017,1,1)):
    curr_dir = '../data/PRICES/'
    for tic in tickers:
        prices = web.DataReader(tic, 'yahoo', begin_date, end_date)[['Close']]
        file_name = curr_dir + tic +'.pkl'
        create_file(file_name)
        prices.to_pickle(file_name)
        
def store_factors(factor_name, begin_date = dt.date(2001,1,1), end_date = dt.date(2017,1,1)):
    #factor_names
    #F-F_Research_Data_Factors_daily
    #F-F_Research_Data_5_Factors_2x3_daily
    curr_dir = '../data/FACTORS/'
    file_name = curr_dir + factor_name +'.pkl'
    create_file(file_name)
    df = web.DataReader(factor_name, "famafrench", begin_date, end_date )
    df = df[0]
    df.to_pickle(file_name)

#def run_store_stocks_exposure(ticker, factor_name, halflife = 252):
#    for tic in tickers:

def get_factor_mean(factor_name, halflife=252):
    factor_file = '../data/FACTORS/'+ factor_name +'.pkl'
    factor_ret = pd.read_pickle(factor_file)
    factor_ret = factor_ret.iloc[:,0:(factor_ret.shape[1]-1)]
    
    factor_ret = factor_ret/252.0
    factor_ret.replace([np.inf, -np.inf], np.nan, inplace = True)
    factor_ret.dropna(axis=0, how='any', inplace = True)
    
    factor_mean = factor_ret.ewm(halflife=halflife).mean()
    factor_mean.replace([np.inf, -np.inf], np.nan, inplace = True)
    factor_mean.dropna(axis=0, how='any', inplace = True)
    
    return factor_mean
def get_factor_cov(factor_name, halflife = 252):
    
    factor_file = '../data/FACTORS/'+ factor_name +'.pkl'
    factor_ret = pd.read_pickle(factor_file)
    factor_ret = factor_ret.iloc[:,0:(factor_ret.shape[1]-1)]
    
    factor_ret = factor_ret/252.0
    factor_ret.replace([np.inf, -np.inf], np.nan, inplace = True)
    factor_ret.dropna(axis=0, how='any', inplace = True)
    
    factor_cov = factor_ret.ewm(halflife=halflife).cov()
    factor_cov.replace([np.inf, -np.inf], np.nan, inplace = True)
    factor_cov.dropna(axis=0, how='any', inplace = True)
    
    return factor_cov
        
def store_stocks_exposures(tickers, factor_name, halflife = 252):
    curr_dir = '../data/BETA/' + factor_name + '/'
    
    factor_file = '../data/FACTORS/'+ factor_name +'.pkl'
    factor_ret = pd.read_pickle(factor_file)
    
    #remove interest rate from data
    factor_ret = factor_ret.iloc[:,0:(factor_ret.shape[1]-1)]

    # converting anuual to daily ret    
    factor_ret = factor_ret/252.0
    factor_ret.replace([np.inf, -np.inf], np.nan, inplace = True)
    factor_ret.dropna(axis=0, how='any', inplace = True)
        

    factor_cov = get_factor_cov(factor_name, halflife )
    factor_mean = get_factor_mean(factor_name, halflife)

    for ticker in tickers:
        print ticker
        file_name = curr_dir + ticker
        tic_file = '../data/PRICES/' + ticker +'.pkl'
        try :
           tic_prices = pd.read_pickle(tic_file)
        except :
            continue
        
        if tic_prices.shape[0] < 1000 :
            continue
        
        tic_ret = tic_prices.pct_change()
        
        tic_ret.replace([np.inf, -np.inf], np.nan, inplace = True)
        tic_ret.dropna(axis=0, how = 'any', inplace = True)
        
        common_index = factor_ret.index.intersection(tic_ret.index)
        tic_ret = tic_ret.loc[common_index]
        factor_ret = factor_ret.loc[common_index]
        
        
        tic_mean = tic_ret.ewm(halflife=halflife).mean()
        tic_mean.dropna(axis=0, how='any', inplace = True)
        
        cross_cov_mat = factor_ret.ewm(halflife=halflife).cov(tic_ret, True)
        cross_cov_mat.dropna(axis=0, how= 'any', inplace = True)
        
        beta_df = []
        fac_names = factor_ret.columns.values
        fac_names = np.append( fac_names, 'vol' )
        
        fac_dates = np.array( [ x[0].date() for x in factor_cov.index ] )
        fac_dates = np.unique(fac_dates)
        cross_dates = np.array( [ x[0].date() for x in cross_cov_mat.index ] )
        cross_dates = np.unique(cross_dates)
        common_dates = np.intersect1d(fac_dates, cross_dates )
        
        for curr_date in  common_dates:
            
            curr_cov_mat = factor_cov.loc[curr_date]
            
            curr_cross_cov_mat = cross_cov_mat.loc[curr_date]
            
            pres_mat = np.linalg.inv(curr_cov_mat)
           

            beta_vec = np.dot(pres_mat, curr_cross_cov_mat.T)

            y_diff = factor_ret.loc[curr_date] - factor_mean.loc[curr_date]

            res = np.dot(beta_vec.T, y_diff)
            res = tic_mean.loc[curr_date] + res
            res = tic_ret.loc[curr_date] - res

            this_vec = beta_vec
            this_vec = np.append(this_vec, res)
            this_vec = pd.Series(this_vec, index = fac_names, name = curr_date)
            
            beta_df.append(this_vec)
        beta_df = np.array(beta_df)

        beta_df = pd.DataFrame(beta_df, index = common_dates, columns = fac_names)

        res_vec = beta_df[fac_names[-1]]
        omega_vec = res_vec.ewm(halflife).var()
        
        #annualizing volatility 
        omega_vec = np.sqrt( omega_vec * 252.0 )
        
        beta_df[fac_names[-1]] = omega_vec        
        create_file(file_name)
        beta_df.dropna(axis=0, how='any', inplace = True)
        
        #removing first few points
        beta_df = beta_df.iloc[int(halflife*2):, :]
        beta_df.to_pickle(file_name)
    return 0 
    
class PricesDateFrame:
    def __init__(self,tickers, price_type, begin_date, end_date):
        self.prices = pd.DataFrame()
        self.tickers = tickers[:]
        self.price_type = price_type
        self.begin_date = begin_date
        self.end_date = end_date
        self.loadPrices()
   
#    def get_prices_df(tickers, begin_year, end_year):
#       df = pd.DataFrame()
#    
#       begin_date = dt.datetime(begin_year,1,1)
#       end_date = dt.datetime(end_year,12,30)
#
#       for tic in tickers:
#          p = web.DataReader(tic, 'yahoo', begin_date, end_date).Close
#          df[tic] = p
#       df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
#      return df 
#     

    def loadPrices(self):
        self.prices = pd.DataFrame()
        for tic in self.tickers : 
            ticker_prices = web.DataReader(tic, 'yahoo', self.begin_date, self.end_date)[self.price_type]
            self.prices[tic] = ticker_prices
        
        self.prices.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    
    def getPrices(self, get_copy = True):
        if get_copy:
           return self.prices.copy()
        else:
            return self.prices
    
    def getNearestDates(self, given_dates, avail_dates ):
         running_index = 0
         nearest_dates = []

         running_index = 0        
         target_date = given_dates[running_index]

         for i, curr_date in enumerate(avail_dates):
             if curr_date >= target_date :
                nearest_dates.append(curr_date)
                running_index += 1
             
             if len(nearest_dates) < len(given_dates):
                target_date = given_dates[running_index]
             else:
                 break
       
         return nearest_dates        
    
    def  getReturnsforDatesSeries(self, begin_dates_ts, end_dates_ts):
                 
         avail_dates = [ x.date() for x in self.prices.index ]
         nearest_begin_dates = self.getNearestDates( begin_dates_ts, avail_dates)
         nearest_end_dates = self.getNearestDates( end_dates_ts, avail_dates)
         
         begin_prices = self.prices.loc[nearest_begin_dates].values
         end_prices = self.prices.loc[nearest_end_dates].values
         ret = (end_prices - begin_prices)/begin_prices
         ret = pd.DataFrame(ret,columns=self.prices.columns, index=begin_dates_ts)
         return ret
             
   
             
             
              