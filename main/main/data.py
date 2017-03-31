## This file will take care of dataset building and all other work on data

import pandas as pd
import numpy as np
import os

def dataset_building(n_max=None):
    ''' Build the dataset
    For future datasets please improve this function with arguments to indicate how to build it
    '''
    cd=os.getcwd()
    price_data=pd.read_csv(cd+'\\data\\data_fx.csv') 
    price_data.set_index(price_data.columns[0],inplace=True)
    price_data.index=pd.to_datetime(price_data.index,infer_datetime_format=True) # Speed optimized after testing
    
    # Make sure the date index is ascending, we avoid to sort because of the complexity
    price_data=price_data.sort_index(axis=0,ascending=True)
    
    # Cut the dataset to a lower number of obs
    if n_max is not None:
        price_data = price_data.iloc[-n_max:,:]     
    
    # Turn prices into returns
    return_data=price_data/price_data.shift(1)-1
    
    return return_data


def lagged(df,lags): 
    ''' Add lags in a dataframe '''
    # It would be better to code a version of this function using reference shifting instead of copying data    
    dfs=[pd.DataFrame(index=df.index)]
    for lag in lags: # add lags of y
        temp_df=df.shift(lag)
        temp_df.columns=temp_df.columns+' L'+str(lag)
        dfs.append(temp_df)
    res=pd.concat(dfs,axis=1) # Speed optimized
    return res

def to_class(input, threshold=0): 
    ''' Converts a numeric dataframe into a class dataframe, works for a single value too '''
    output=(abs(input)>threshold)*np.sign(input) # Syntax using Bool*Float multiplication and DataFrame operations, be careful with the -0.0 though
    return output
        

def core_dataset(algos, algos_used):
    ''' Built a core dataset using predictions from the algos '''
    list_df=[]
    for key in algos_used:
        list_df.append(algos[key].get_output('predicted_values'))
    return pd.concat(list_df, axis=0)    