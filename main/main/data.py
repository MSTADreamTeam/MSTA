## This file will take care of dataset building and all other work on data

import pandas as pd
import numpy as np
import os
import quandl


def dataset_building(n_max=None, verbose=None):
    ''' Build the dataset '''
    if verbose: print('Dataset building...')
    #cd=os.getcwd()
    stock_data=pd.read_csv('C:/Users/Loic/Documents/work/interview/maven/reference.data/stock.data.csv')
    index_data=pd.read_csv('C:/Users/Loic/Documents/work/interview/maven/reference.data/reference.data.csv')
        
    for data in [stock_data, index_data]:
        data.set_index(data.columns[0],inplace=True)
        data.index=pd.to_datetime(data.index,infer_datetime_format=True) # Speed optimized after testing
        # Make sure the date index is ascending, we avoid to sort because of the complexity
        data=data.sort_index(axis=0,ascending=True)
       
        # Cut the dataset to a lower number of obs
        if n_max is not None:
            data = data.iloc[-(n_max + 1):]     
       
    if verbose: print('Dataset built')

    return stock_data, index_data


def add_returns(df, col_index=None, window=1):
    ''' Compute the returns for some preselected columns of the dataset
        And add them to the dataset
    '''
    if not col_index:
        col_index=df.columns
    for col in col_index:
        df[col+' Ret'] = ((1+df.loc[:,col])/(1+df.loc[:,col].shift(window))) - 1
    return df


def lagged(df,lags): 
    ''' Add lags in a dataframe '''
    # It would be better to code a version of this function using reference shifting instead of copying data    
    dfs=[pd.DataFrame(index=df.index)]
    for lag in lags: # add lags of y
        temp_df=df.shift(lag)
        if isinstance(temp_df, pd.DataFrame):
            temp_df.columns=temp_df.columns+' L'+str(lag)
        else: # Case where the input is a Serie
            temp_df.name=temp_df.name+' L'+str(lag) 
        dfs.append(temp_df)
    res=pd.concat(dfs,axis=1) # Speed optimized
    return res

def to_class(input, threshold=0): 
    ''' Converts a numeric dataframe/serie/value into a classified similar type output '''
    output=(abs(input)>threshold)*np.sign(input) # Syntax using Bool*Float multiplication and DataFrame operations, be careful with the -0.0 though
    return output
        

def core_dataset(algos, algos_used):
    ''' Built a core dataset using predictions from the algos '''
    list_df=[]
    for key in algos_used:
        list_df.append(algos[key].get_output('predicted_values'))
    return pd.concat(list_df, axis=0)

def lagdf_to_ts(df):
    ''' Transform a lagged dataset into a time series of all available prices
    It is used in TA algorithms, and only works with a dataframe with at least 2 rows as input
    '''
    res=df.iloc[0,::-1].append(df.iloc[1:,0])
    return res


#import matplotlib.pyplot as pyplt
#pyplt.plot(res.values)
#pyplt.show()