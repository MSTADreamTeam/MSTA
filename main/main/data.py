## This file will take care of dataset building and all other work on data

import pandas as pd
import numpy as np
import os
import quandl

def dataset_building(source='local',asset_ids=None,start_date=None, end_date=None, n_max=None):
    ''' Build the dataset
    It currently supports local and quandl source
    ''' 
    if source=='local':
        cd=os.getcwd()
        price_data=pd.read_csv(cd+'\\data\\data_fx.csv') 
        price_data.set_index(price_data.columns[0],inplace=True)
        price_data.index=pd.to_datetime(price_data.index,infer_datetime_format=True) # Speed optimized after testing
        # Make sure the date index is ascending, we avoid to sort because of the complexity
        price_data=price_data.sort_index(axis=0,ascending=True)
        
    elif source=='quandl':
        #quandl.ApiConfig.api_key = input('You chose Quandl as a data source, please provide your API key:')
        quandl.ApiConfig.api_key = '8WpG1NY8N1mMn3NT6y9u' # Please use the console input if your key is linked to a premium account
        price_data=quandl.get(asset_ids, start_date=start_date, end_date=end_date, order = 'asc') #Quandl natively gives the results in a desc order

    # Cut the dataset to a lower number of obs
    if n_max is not None:
        price_data = price_data.iloc[-(n_max + 1):]     
    return price_data


def add_returns(df, columns, window=1):
    ''' Compute the returns for some preselected columns of the dataset
        Please provide numarical columns indexes
        And add them to the dataset
    '''
    for i in columns:
        df[df.columns[i]+' Ret'] = (df.iloc[:,i]/df.iloc[:,i].shift(window)) - 1
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