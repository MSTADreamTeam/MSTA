## This file will be the main py file
## It will include all necessary dependencies and libraries, it will run all steps of the algo
## In order to compile this file into an executable file we might need an additional program, also check the pyc extension

import data
import numpy as np
from multithread import MultiThreadCP

from historical_mean import HM
from linear_regression import LR
from core_base_algos import CMW, BIS
from tree import DT
from random_forest import RF
from adaboost import ADAB
from multi_layer_perceptron import MLP
from golden_dead_cross import GDC


## Global Hyperparameters
# The window size of the rolling window used to define each training set size
# The models will never see more than this number of points at once
rolling_window_size=500
    
# Output type : C for Classification, R for Regression
output_type='C'
# Note that for a Classification, 1 means positive return, -1 means negative, and 0 means bellow threshold
# In case of 3 class Classification, please provide an absolute level for the zero return threshold
# Fix it to 0 for a binary classification
# The optimal value can also be globally optimized as a result of the PnL optimisation and will be function of the volatility of the asset
threshold=0.001

# This dictionary of global hyperparameters will be passed an an argument of all built algorithms
global_hyperparams={'rolling_window_size':rolling_window_size,
                    'output_type':output_type,
                    'threshold':threshold}    

## Building the dataset
# Define the main asset ID
main_id='CUR/EUR'
start_date='01/01/2014'
end_date=None # None to go until the last available data

# Define the additional data you want to recover
asset_ids=[main_id]+[]
dataset=data.dataset_building('quandl', asset_ids, start_date, end_date, n_max=1600)

# We select an asset returns time series to predict from the dataset
Y=dataset[dataset.columns[0]]
#Y.dropna(inplace=True)

# X: include all the lags of Y and additional data
lags=range(1,rolling_window_size+1)
X=data.lagged(Y,lags=lags) # In X please always include all the lags of Y that you want to use for the HM as first colunms
max_lags=max(lags)
# We could also turn X into classes data, is that meaningful?
# X=to_class(X,threshold)    

# In case of classification, we classify Y 
if output_type=='C':
    Y=data.to_class(Y, threshold)

## Creating & calibrating the different algorithms

# First define a dictionary of algorithm associated with their names
# As arguments please include the fixed hyperparams of the model as a named argument
# For the hyperparameters grid to use in cross validation please provide a dictionary using sklearn syntax 
algos={'HM AR Full window':HM(global_hyperparams),# hp_grid={'window_size':[10,100,500]}),
       'HM GEO Full window':HM(global_hyperparams,mean_type='geometric',hp_grid={'window_size':[1,10,50,100]}),
       'HM AR Short Term':HM(global_hyperparams,window_size=10),
       'LR':LR(global_hyperparams),
       'Lasso':LR(global_hyperparams, regularization='Lasso',hp_grid={'alpha':np.logspace(-4,1,10)}),
       'ElasticNet':LR(global_hyperparams, regularization='ElasticNet',hp_grid={'alpha':np.logspace(-3,1,20),'l1_ratio':np.linspace(0,1,20)}),
       'Tree':DT(global_hyperparams,hp_grid={'max_features':['sqrt',None],'criterion':['gini','entropy']}),
       'RF':RF(global_hyperparams, hp_grid={'max_features':['sqrt',None],'n_estimators':range(10,200,20)}),
       'ADAB':ADAB(global_hyperparams, hp_grid={'n_estimators':[1,5,10]}, base_algo=DT(global_hyperparams)),
       'MLP':MLP(global_hyperparams,hp_grid={'alpha':np.linspace(0,1,10),'hidden_layer_sizes':[(5,5),(10,10,10)]},activation='relu', solver='lbfgs')}
       #'GDC':GDC(global_hyperparams, hp_grid=None, stw=None, ltw=None, a=None, b=None, c=None)}

# Then we just allow ourselves to work only a subset of these algos
algos_used=algos.keys()
#algos_used=['Lasso']
#algos_used=['HM GEO Full window']
#algos_used=['HM AR Full window','LR','Lasso','ADAB']
#algos_used=['RF']
#algos_used=['ElasticNet']
#algos_used=['MLP']

# The default cross validation parameters dictionary
default_cv_params={'cross_val_type':'ts_cv',
                   'n_splits':5,
                   'calib_type':'GridSearch'}

# Fix the cross validation parameters of each algorithm you wish to use
algos_cv_params={key:dict(default_cv_params) for key in algos_used} # The dict constructor allows for a copy of the default dict
algos_cv_params['Lasso']['calib_type']='RandomSearch'
algos_cv_params['Lasso']['n_iter']=5
algos_cv_params['MLP'].update({'calib_type':'GeneticAlgorithm',
                   'scoring_type':None,
                   'n_iter':7,
                   'init_pop_size':4,
                   'select_rate':0.5, 
                   'mixing_ratio':0.5,  
                   'mutation_proba':0.1, 
                   'std_ratio':0.1})

# Define the multithreading call queue
# We define one thread by algorithm, it avoids problems with the GIL
# since we will avoid to have several thread working on the same object 
multithreading=True

if multithreading: mt=MultiThreadCP(thread_names=algos_used)

for key in algos_used:    
    # First we select the data we want to use with this algo
    algos[key].select_data(X)


for i in range(rolling_window_size+max_lags,len(Y.index)): # Note that i in the numeric index in Y of the predicted value
    for key in algos_used: # DO not invert the loops to make sure the multithreading is not lost
        train=range(i-rolling_window_size,i) # should be equal to i-rolling_window_size:i-1
        test=[i] # I am not sure of the index, we can check, it is inside [] to make sure the slicing produces a dataframe
        pred_index=Y.index[test] # This is the timestamp of i

        if multithreading: # We add the task to the MultiThreading calib & fit object
            mt.add_task(thread_name=key,
                        algo=algos[key], 
                        X_train=X.iloc[train], 
                        Y_train=Y.iloc[train], 
                        X_test=X.iloc[test], 
                        pred_index=pred_index,
                        algo_cv_params=algos_cv_params[key])        
        else: # We calibrate the hyperparameters and predict
            algos[key].calib_predict(X_train=X.iloc[train], 
                                     Y_train=Y.iloc[train], 
                                     X_test=X.iloc[test], 
                                     pred_index=pred_index,
                                     **algos_cv_params[key])

        
# Make sure that all the threads are done
if multithreading:
    print('*** Main thread waiting')
    mt.wait()
    print('*** Done')

# We compute the output
for key in algos_used:
    algos[key].compute_outputs(Y)
    
## Core algorithm
# Definition of the core algo
core=HM(global_hyperparams) # Average of the predictions 
core=BIS(global_hyperparams, 'accuracy')

# We first built a new dataset with all the predictions for the algos, it will be our new X
X_core=data.core_dataset(algos, algos_used)
Y_core=Y.loc[X_core.index]

# Again we do the rolling window estimation process
for i in range(rolling_window_size+1,len(Y_core.index)): # Note that i in the numeric index in Y of the predicted value
    train=range(i-rolling_window_size,i) # should be equal to i-rolling_window_size:i-1
    test=[i] # I am not sure of the index, we can check, it is inside [] to make sure the slicing produces a dataframe
    pred_index=Y_core.index[test] # This is the timestamp of i

    # We calibrate the hyperparameters and predict
    core.calib_predict(X_train=X_core.iloc[train], 
                             Y_train=Y_core.iloc[train], 
                             X_test=X_core.iloc[test], 
                             pred_index=pred_index,
                             **default_cv_params)

# We compute the outputs
core.compute_outputs(Y_core)
            
# We check that the core algo is better than all other algos:
# careful about computing the acuracy only the same window
scoring='accuracy'
for algo in algos_used:
    if getattr(algos[key],scoring)>getattr(core,scoring): print('Warning: {} got a better score than the core algo {}'.format(algos[key].name, core.name))

# for debug
print(core.best_hp)
pass

## Trading Strategy

## Backtest/Plots/Trading Execution

