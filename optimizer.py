#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:28:38 2017

@author: raghuramkowdeed
"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

execfile('./price_reader.py')

def getZscore( signal_vec):
    z_score = np.copy(signal_vec)
    ind = [ i for i in range(len(z_score)) if z_score[i]!= None and ( not np.isnan(z_score[i]) ) ]

    mean = np.mean( z_score[ind] )
    std = np.std( z_score[ind] )
    z_score[ind] = ( z_score[ind] - mean )/std
    #print 'z_score'
    #print z_score
    return z_score

def getAlpha(signal_vec, omega, adj_omega):
    z_score = getZscore(signal_vec)
    alpha = z_score
    if adj_omega:
        alpha = alpha*omega
    return alpha

def getQuantileWeights(alpha, quantile = 0.25):
      ind_1 = np.where(alpha != np.nan )[0]
      alpha_valid = alpha[ind_1]
      ind_sort = np.argsort(alpha_valid)

      num_pos = int(ind_sort.shape[0]*quantile)
      
      weights = np.zeros(alpha.shape[0])*0.0
      weights[ind_1[ind_sort][:num_pos]] = -1.0/num_pos
      weights[ind_1[ind_sort][-num_pos:]] = 1.0/num_pos
      #print 'weights'
      #print weights
      return weights
    
def getZScoreWeights(alpha):
      ind_1 = [ i for i in range(len(alpha)) if alpha[i]!= None and ( not np.isnan(alpha[i]) ) ]
      alpha_valid = alpha[ind_1]

      weights = np.zeros(alpha.shape[0])*0.0
      weights[ind_1] = alpha_valid

      #print 'weights'
      #print weights
      #print 'alpha_valid'
      #print alpha_valid
      return weights

def solve_mean_var_equation( alpha_vec, total_cov, beta_mat):
    pres_mat = np.linalg.inv(total_cov)
    
    t1 = np.dot(beta_mat, pres_mat)
    t1 = np.dot(t1,beta_mat.transpose())
    t1 = np.linalg.inv(t1)
    t1 = np.dot(t1, beta_mat)
    t1 = np.dot(t1, pres_mat)
    t1 = np.dot(t1, alpha_vec)
    
    c= t1
    
    weights_vec = alpha_vec - np.dot(beta_mat.transpose(), c )
    weights_vec = np.dot(pres_mat, weights_vec)
    
    weights_vec = 2.0 * weights_vec/np.sum( np.abs(weights_vec) )
    
    s = np.dot(beta_mat, weights_vec)
    #print 'sum = ' + str( abs(s).sum() )
    return weights_vec

class Optimizer:
    def __init__(self, tickers, begin_date, end_date):
        self.tickers = tickers
        self.begin_date = begin_date
        self.end_date = end_date
    
    def getOmega(self, curr_date):
        return None
    
    def getWeights(self, signal_vec, opt_type = 'quantile', adj_omega = False, quantile = 0.25):

        if opt_type == 'quantile':
            alpha = getAlpha(signal_vec, None, adj_omega)
            weights = getQuantileWeights(alpha, quantile)
            return weights
        if opt_type == 'z_score':
            alpha = getAlpha(signal_vec, None, adj_omega)
            weights = getZScoreWeights(alpha)
            return weights
        if opt_type == '':
            return None
    #signal vec is np array

class OptimizationWizard:
    def __init__(self, tickers, factor_name, optimization_dates):
        self.tickers = tickers
        self.dates= optimization_dates
        self.factor_name = factor_name
        self.horizon = 12
        self.rf = 0
        
        self.beta_book = {} 
        self.risk_book = {}
        self.stock_pnl_book = {}
        self.factor_pnl_book = {}
        
        self.loadBetaBook()
        self.loadRiskBook()
        self.loadStockPnlBook()
        self.loadFactorPnlBook()
        
    def loadStockPnlBook(self):
        curr_dir = '../data/PRICES/' 
        
        #factor_data = pd.read_pickel(factor_file)
        
        ticker_info = {}
        
        for tic in self.tickers:

            this_file = curr_dir + tic + '.pkl'
            this_data = pd.read_pickle(this_file)
            ret_data = this_data.pct_change(periods=self.horizon*28)*12.0/self.horizon
            ret_data.dropna(inplace = True)

            avail_begin_dates = [ this_data.index[this_data.index.get_loc(date,method='nearest')] for date in self.dates ]
            avail_end_dates = [ this_data.index[this_data.index.get_loc(date + relativedelta(months = self.horizon),method='nearest')] for date in self.dates ]
            #print avail_dates
            tic_data = ret_data.loc[avail_end_dates] 
            
            tic_data.index = self.dates            
            ticker_info[tic] = tic_data
        
        self.stock_pnl_book = {}

        for date in self.dates:

            this_df = pd.DataFrame()

            for tic in self.tickers :
                tic_row = ticker_info[tic].loc[date]
                tic_row.name = tic
                this_df = this_df.append(tic_row)

            self.stock_pnl_book[date] = this_df

    
    def loadFactorPnlBook(self):
        factor_file = '../data/FACTORS/'+ self.factor_name +'.pkl'
        #convert to daily ret
        
        fac_ret = pd.read_pickle(factor_file)/252.0
        
        fac_ret.dropna(inplace = True)
        rf = fac_ret.iloc[:, -1] 
        fac_ret = fac_ret.iloc[:,0:(fac_ret.shape[1]-1)]
        
        fac_pnl = fac_ret + 1.0
        fac_pnl = fac_pnl.cumprod(axis=0)
        
        fac_ret_h = fac_pnl.pct_change(periods = self.horizon*28)*12.0/self.horizon
        fac_ret_h.dropna(inplace= True)
        
        avail_begin_dates = [ fac_ret_h.index[fac_ret_h.index.get_loc(date,method='nearest')] for date in self.dates ]
        avail_end_dates = [ fac_ret_h.index[fac_ret_h.index.get_loc(date + relativedelta(months = self.horizon),method='nearest')] for date in self.dates ]
        
        fac_ret_h = fac_ret_h.loc[avail_end_dates]
        fac_ret_h.index = self.dates

        self.factor_pnl_book = {}
        
        for date in self.dates:
           self.factor_pnl_book[date] = fac_ret_h.loc[date]
        
        self.rf = rf.loc[avail_end_dates]*252.0
        self.rf.index = self.dates

               
    def loadRiskBook(self):
        fac_cov = get_factor_cov(self.factor_name, halflife = 252)*252   
        
        temp_dict = {}
        
        fac_dates = np.array( [ x[0].date() for x in fac_cov.index ] )
        fac_dates = np.unique(fac_dates)
        
        search_ts = pd.Series(range(len(fac_dates)), index = fac_dates)
    
        for date in self.dates:
            near_ind = search_ts.index.get_loc(date,method='nearest')
            near_date = search_ts.index[near_ind]
            this_cov = fac_cov.loc[near_date]
            temp_dict[date] = this_cov
        
        self.risk_book = temp_dict
        
    def loadBetaBook(self):
        curr_dir = '../data/BETA/' + self.factor_name + '/'
        
        #factor_data = pd.read_pickel(factor_file)
        
        ticker_info = {}
        
        for tic in self.tickers:

            this_file = curr_dir + tic
            this_data = pd.read_pickle(this_file)

            avail_dates = [ this_data.index[this_data.index.get_loc(date,method='nearest')] for date in self.dates ]
            #print avail_dates
            tic_data = this_data.loc[avail_dates] 
            tic_data.index = self.dates            
            ticker_info[tic] = tic_data
        
        self.beta_book = {}
        
        for date in self.dates:

            this_df = pd.DataFrame()

            for tic in self.tickers :
                tic_row = ticker_info[tic].loc[date]
                tic_row.name = tic
                this_df = this_df.append(tic_row)

            self.beta_book[date] = this_df
    

    def getZscoreDF(self, signal_df):
        signal_df = signal_df.fillna(value=np.nan)
        z_df = pd.DataFrame()
        
        for i in range( signal_df.shape[0] ):
            this_z = getZscore( signal_df.iloc[i,:])
            this_z = pd.Series(this_z, name = signal_df.index[i], index = signal_df.columns)
            z_df = z_df.append(this_z)
        
        dic = { z_df.columns[i]:name for i, name in enumerate(signal_df.columns)}
        z_df.rename(inplace = True, columns=dic)
        return z_df
         
    
        
    
    def getAlpha(self, signal_df, adj_alpha, ic = 0.5, const_fac = 1.0 ):
        z_score = self.getZscoreDF(signal_df)
        alpha_df = pd.DataFrame()
 
        for i in range( z_score.shape[0] ):
            this_date = z_score.index[i]
            omega_vec = np.ones(z_score.shape[1])
            if adj_alpha :
               omega_vec = self.beta_book[this_date]['vol']
            alpha_vec = ic*const_fac*z_score.iloc[i,:] * omega_vec
            alpha_vec.name = this_date
            alpha_df = alpha_df.append(alpha_vec)
            alpha_df.fillna(value=np.nan, inplace=True)
            
        return alpha_df
       
    def getQuantileWeights(self, alpha_df, quantile):
        alpha_df = alpha_df.fillna(value=np.nan, inplace=False)
        weights_df = pd.DataFrame()

        for i in range(alpha_df.shape[0]):
            weights_vec = getQuantileWeights(alpha_df.iloc[i,:])
                        
            #making abs notional = 2      
            ind = np.where( np.isnan(weights_vec) == False)[0]
            ind_0 = np.where(np.isnan(weights_vec))[0]
            weights_vec[ind_0] = 0.0
            notional = ( np.sum(np.abs(weights_vec) ) )
            weights_vec = 2.0*weights_vec/notional
            
            weights_vec = pd.Series(weights_vec, name = alpha_df.index[i])
            weights_df = weights_df.append(weights_vec)

        dic = { weights_df.columns[i]:name for i, name in enumerate(alpha_df.columns)}
        weights_df.rename(inplace = True, columns=dic)
        weights_df.fillna(value=0.0, inplace=False)
        return weights_df  
     
    def getMeanVarWeights(self, alpha_df):
       alpha_df = alpha_df.fillna(value=np.nan, inplace=False)
       weights_df = pd.DataFrame()

       for i in range(alpha_df.shape[0]):
            curr_date = alpha_df.index[i]
            vol_vec = self.beta_book[curr_date]['vol'].values
            #print 'vol_vec'
            #print vol_vec
            #print 'alpha_vec'
            alpha_vec = (alpha_df.iloc[i,:]).values
            #print alpha_vec
            weights_vec = alpha_vec/(np.square( vol_vec) )
            #print 'weights vec'

            #making abs notional = 2      
            
            ind = np.where( np.isnan(weights_vec) == False)[0]
            ind_0 = np.where(np.isnan(weights_vec))[0]
            weights_vec[ind_0] = 0.0
            notional = ( np.sum(np.abs(weights_vec ) ) )
            #print weights_vec
            #print 'notional'
            #print notional
            weights_vec = 2.0*weights_vec/notional
                       
            weights_vec = pd.Series(weights_vec, name = alpha_df.index[i])
            #print weights_vec
            weights_df = weights_df.append(weights_vec)

       dic = { weights_df.columns[i]:name for i, name in enumerate(alpha_df.columns)}
       weights_df.rename(inplace = True, columns=dic)
       weights_df.fillna(value= 0.0, inplace = True) 
       
       return weights_df  
     
    def getZeroExposueWeights(self, alpha_df):
       alpha_df = alpha_df.fillna(value=np.nan, inplace=False)
       weights_df = pd.DataFrame()

       for i in range(alpha_df.shape[0]):
            curr_date = alpha_df.index[i]

            #print 'vol_vec'
            #print vol_vec
            #print 'alpha_vec'
            alpha_vec = alpha_df.iloc[i,:]
            ind = [ i for i,v in enumerate(alpha_vec) if not np.isnan(v) ]
            
            this_alpha_vec = alpha_vec[ind]
            beta_weights = self.beta_book[curr_date].iloc[ind, 0:-1]
            vol_vec = self.beta_book[curr_date]['vol'].iloc[ind]
            beta_var = self.risk_book[curr_date]
            
            var_1 = np.diag(vol_vec*vol_vec)
            var_2 = np.dot(beta_weights, beta_var)
            var_2 = np.dot(var_2, beta_weights.transpose())
            
            total_var = var_1 + var_2
            
            sol = solve_mean_var_equation(this_alpha_vec, total_var, beta_weights.transpose())
            
            weights_vec = pd.Series(np.zeros(alpha_vec.shape[0]), 
                                    index = alpha_vec.index, name = curr_date)
            
            weights_vec.iloc[ind] = sol
            
            #print alpha_vec

            #print 'weights vec'

            #making abs notional = 2      
            
            weights_df = weights_df.append(weights_vec)

       dic = { weights_df.columns[i]:name for i, name in enumerate(alpha_df.columns)}
       weights_df.rename(inplace = True, columns=dic)
       weights_df.fillna(value= 0.0, inplace = True) 
       
       return weights_df  
        
    def runStrategy(self, weights_df):
        
        port_pnl_vec = []
        fac_pnl_vec = []
        alpha_vec = []
        

        for date in self.dates :
            weights_vec = weights_df.loc[date]
            
            stocks_ret = self.stock_pnl_book[date]
            fac_ret = self.factor_pnl_book[date]
            rate = self.rf.loc[date]
       
            beta_vec = self.beta_book[date] 
            beta_vec = beta_vec.iloc[:, 0:(beta_vec.shape[1]-1) ]
            
            beta_vec = weights_vec.dot( beta_vec )
            
            port_pnl = weights_vec.dot(stocks_ret) - rate
            fac_pnl = beta_vec.dot(fac_ret)
            alpha = port_pnl - fac_pnl
            
            port_pnl_vec.append(port_pnl.iloc[0])
            fac_pnl_vec.append(fac_pnl)
            alpha_vec.append(alpha.iloc[0])
        
        port_pnl_vec = pd.Series(port_pnl_vec, index = self.dates)
        fac_pnl_vec = pd.Series(fac_pnl_vec, index = self.dates)
        alpha_vec = pd.Series(alpha_vec, index = self.dates)
        
        return port_pnl_vec, fac_pnl_vec, alpha_vec