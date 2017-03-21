## This file will take care of dataset building

import pandas as pd
import numpy as np


def dataset_building(n_max = None):
    datafile=pd.ExcelFile('data_fx.xlsx')
    price_data=datafile.parse('Sheet1')
    # Cut the dataset to a lower number of obs
    if n_max is not None:
        price_data = price_data.iloc[:n_max,:]     
    # Make sure the date index is ascending
    price_data=price_data.sort_index(axis=0,ascending=True)
    # Turn prices into returns
    return_data=price_data/price_data.shift(1)-1
    return return_data


def lagged(df,lags): # add lags in a dataframe
    # this function was imported from past code and needs to be checked
    # it would be better to code a version of this function using reference shifting instead of copying data    
    res=pd.DataFrame(index=df.index)
    for lag in lags: # add lags of y
        temp_df=df.shift(lag)
        temp_df.columns=temp_df.columns+" L"+str(lag)
        res=pd.concat([res,temp_df],axis=1)
    return res