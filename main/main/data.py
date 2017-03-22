## This file will take care of dataset building

import pandas as pd
import numpy as np


def dataset_building(n_max=None):
    price_data=pd.read_csv('data_fx.csv')
    # Cut the dataset to a lower number of obs
    price_data.set_index(price_data.columns[0],inplace=True)
    price_data.index=pd.to_datetime(price_data.index,infer_datetime_format=True) # Speed optimized after testing
    if n_max is not None:
        price_data = price_data.iloc[:n_max,:]     
    # Make sure the date index is ascending, we avoid to sort because of the complexity
    #price_data=price_data.sort_index(axis=0,ascending=True)
    # Turn prices into returns
    return_data=price_data/price_data.shift(1)-1
    return return_data


def lagged(df,lags): # add lags in a dataframe
    # this function was imported from past code and needs to be checked
    # it would be better to code a version of this function using reference shifting instead of copying data    
    dfs=[pd.DataFrame(index=df.index)]
    for lag in lags: # add lags of y
        temp_df=df.shift(lag)
        temp_df.columns=temp_df.columns+" L"+str(lag)
        dfs.append(temp_df)
    res=pd.concat(dfs,axis=1) # Speed optimized
    return res