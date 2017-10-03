#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:06:12 2017

@author: raghuramkowdeed
"""
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
execfile('./parser/score_utils.py')
execfile('./optimizer.py')
execfile('./price_reader.py')

class StrategyRunner:
      def __init__(self,tickers, begin_date, end_date):
         self.prices = pd.DataFrame()
         self.tickers = tickers[:]
         self.begin_date = begin_date
         self.end_date = end_date
         b1 = self.begin_date + relativedelta(years = -3)
         e1 = self.end_date + relativedelta(years = 3)
         self.price_obj = PricesDateFrame(tickers, 'Close', b1, e1)
         
      def run(self, sec, score_func, horizon = 12, lag = 6, opt_type = 'z_score' ):
          signal_df = get_section_scores(self.tickers, sec, self.begin_date, self.end_date, score_func )
          opt_obj = Optimizer(self.tickers, self.begin_date, self.end_date)
          run_dates = signal_df.index
          b1 = self.begin_date + relativedelta(years = 1, months = lag)
          e1 = self.end_date + relativedelta(months = lag)
          begin_dates_ts = []
          end_dates_ts = []

          curr_date = b1
          while curr_date <= e1:
              begin_dates_ts.append(curr_date)
              end_dates_ts.append(curr_date+relativedelta(months = horizon))
              t = curr_date+relativedelta(months = 12)
              curr_date = t

          #print begin_dates_ts
          #print end_dates_ts
          
          ret_df = self.price_obj.getReturnsforDatesSeries(begin_dates_ts, end_dates_ts)
          pnl_vec = []
          for i in range(signal_df.shape[0]) :
              signal = signal_df.iloc[i,:]
              #print 'signal'
              #print signal
              #ind = np.where(signal != None)
              weights = opt_obj.getWeights(signal, opt_type)
              #print 'weights'
              #print weights
              #print 'ret'
              #print ret_df.iloc[i,:]
              pnl = np.dot(weights, ret_df.iloc[i,:])
              pnl_vec.append(pnl)
          
          return pnl_vec  