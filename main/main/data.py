## This file will take care of dataset building

import pandas as pd
import numpy as np


def dataset_building(n_max = None):
    datafile = pd.ExcelFile('data_fx.xlsx')
    if n_max is None:
        dataset = datafile.parse('Sheet1')     
    else:
        dataset = datafile.parse('Sheet1').iloc[:n_max,:]
    return dataset


def add_lags(df,lags): # add lags in a dataframe
    # this function was imported from past code and needs to be checked
    # it would be better to code a version of this function using reference shifting instead of copying data    
    res=pd.DataFrame(index=df.index)
    for lag in lags: # add lags of y
        temp_df=df.shift(lag)
        temp_df.columns=temp_df.columns+" L"+str(lag)
        res=pd.concat([res,temp_df],axis=1)
    return res