## This file will be the main py file
## It will include all necessary dependencies and libraries, it will run all steps of the algo
## In order to compile this file into an executable file we might need an additional program, also check the pyc extension

import data
import numpy as np
import warnings
from multithread import MultiThreadCP

from historical_mean import HM
from linear_regression import LR
from core_base_algos import CMW, BIS
from tree import DT
from random_forest import RF
from adaboost import ADAB
from multi_layer_perceptron import MLP
from golden_dead_cross import GDC
from ma_enveloppe import MAE
from relative_strength_index import RSI
from momentum_roc import MOM, ROC
#from recurrent_neural_network import ESN
#from recurrent_neural_network import LST

from trading_strategy import TradingStrat


## Global Hyperparameters
# The window size of the rolling window used to define each training set size
# The models will never see more than this number of points at once
rolling_window_size=500

# The number of data poiints between each recalibration
recalib_window=50

# Output type : C for Classification, R for Regression
output_type='C'
# Note that for a Classification, 1 means positive return, -1 means negative, and 0 means bellow threshold
# In case of 3 class Classification, please provide an absolute level for the zero return threshold
# Fix it to 0 for a binary classification
# The optimal value can also be globally optimized as a result of the PnL optimisation and will be function of the volatility of the asset
threshold=0.000

# This dictionary of global hyperparameters will be passed an an argument of all built algorithms
global_hyperparams={'rolling_window_size':rolling_window_size,
                    'output_type':output_type,
                    'threshold':threshold}    

## Building the dataset
# Define the main asset ID
main_id='CUR/EUR'
start_date='01/01/2010'
end_date=None # None to go until the last available data

# Define the additional data you want to recover
asset_ids=[main_id]+[]
dataset=data.dataset_building('quandl', asset_ids, start_date, end_date, n_max=None) # please recode the dataset_building functio to make it support local and quandl data
dataset = data.add_returns(dataset, [0]) # creates some NANs as a result of the returns computation
dataset.dropna(inplace=True)

# We select an asset returns time series to predict from the dataset
Y_0=dataset[dataset.columns[1]] # need to find a reliable way to find the index of the column 

# X: include all the lags of Y and additional data
lags=range(1,rolling_window_size+1)
X=data.lagged(dataset,lags=lags) # In X please always include all the lags of Y that you want to use for the HM as first colunms
#max_lags=max(lags)
# We could also turn X into classes data, is that meaningful?
# X=to_class(X,threshold)    

# In case of classification, we classify Y 
if output_type=='C': Y=data.to_class(Y_0, threshold)
    

## Creating & calibrating the different algorithms

# First define a dictionary of algorithm associated with their names
# As arguments please include the fixed hyperparams of the model as a named argument
# For the hyperparameters grid to use in cross validation please provide a dictionary using sklearn syntax 
algos={'HM':HM(global_hyperparams, hp_grid={'window_size':[10,100,500]}),
       #'LR':LR(global_hyperparams),
       #'Lasso':LR(global_hyperparams, regularization='Lasso',hp_grid={'alpha':np.logspace(-4,1,10)}),
       #'ElasticNet':LR(global_hyperparams, regularization='ElasticNet',hp_grid={'alpha':np.logspace(-3,1,20),'l1_ratio':np.linspace(0,1,20)}),
       #'Tree':DT(global_hyperparams,hp_grid={'max_features':['sqrt',None],'criterion':['gini','entropy']}),
       #'RF':RF(global_hyperparams, hp_grid={'max_features':['sqrt',None],'n_estimators':range(10,200,20)}),
       #'ADAB':ADAB(global_hyperparams, hp_grid={'n_estimators':[1,5,10]}, base_algo=DT(global_hyperparams)),
       #'MLP':MLP(global_hyperparams,hp_grid={'alpha':np.linspace(0.1,1,9),'hidden_layer_sizes':[(10,),(100,),(200,)]},activation='relu', solver='lbfgs'),
       'GDC':GDC(global_hyperparams, hp_grid={'stw':[20,50,100],'ltw':[150,200,300],'a':np.linspace(0,1,10),'b':np.linspace(0,1,10)}, c=1),
       'MAE':MAE(global_hyperparams, hp_grid={'w':[10,20,100,200,500],'p1':np.linspace(0.001,0.01,10)}),
       'RSI':RSI(global_hyperparams, hp_grid={'ob':[60,70,80,90],'os':[10,20,30,40],'w':[10,20,100,200,500]}),
       'MOM':MOM(global_hyperparams, hp_grid={'n':range(1,500,10)}),
       'ROC':ROC(global_hyperparams, hp_grid={'n':range(1,500,10)})}

# Then we just allow ourselves to work only a subset of these algos
algos_used=algos.keys()

# The default cross validation parameters dictionary
default_cv_params={'cross_val_type':'ts_cv',
                   'n_splits':5,
                   'calib_type':'GeneticAlgorithm',
                   'scoring_type':None,
                   'n_iter':200,
                   'init_pop_size':100,
                   'select_rate':0.5, 
                   'mixing_ratio':0.5,  
                   'mutation_proba':0.1, 
                   'std_ratio':0.1}
# The params to avoid the cv
no_cv_params={'cross_val_type':None} 

# Fix the cross validation parameters of each algorithm you wish to use
algos_cv_params={key:dict(default_cv_params) for key in algos} # The dict constructor allows for a copy of the default dict

# Define the multithreading call queue
# We define one thread by algorithm, it avoids problems with the GIL
# since we will avoid to have several thread working on the same object 
multithreading=True

if multithreading: mt=MultiThreadCP(thread_names=algos_used)

for key in algos_used:    
    # First we select the data we want to use with this algo
    algos[key].select_data(X)


for i in range(rolling_window_size,len(Y.index)): # Note that i in the numeric index in Y of the predicted value
    recalib=(i-rolling_window_size)  % recalib_window == 0
    if recalib:
        train=range(i-rolling_window_size,i) # should be equal to i-rolling_window_size:i-1
        test=range(i,min(i+recalib_window,len(Y))) # the date we try to predict
        pred_index=Y.index[test] # This is the timestamp of the predicted indices
        X_train=X.iloc[train, algos[key].selected_data]
        Y_train=Y.iloc[train]
        X_test=X.iloc[test, algos[key].selected_data]
        for key in algos_used:
            if multithreading: # We add the task to the MultiThreading calib & predict object
                mt.add_task(thread_name=key, 
                            algo=algos[key], 
                            X_train=X_train,
                            Y_train=Y_train, 
                            X_test=X_test,
                            pred_index=pred_index,
                            algo_cv_params=algos_cv_params[key])        
            else: # We calibrate the hyperparameters and predict
                algos[key].calib_predict(X_train, Y_train, X_test, pred_index, **algos_cv_params[key])

        
# Make sure that all the threads are done
if multithreading:
    print('*** Main thread waiting ***')
    mt.wait()
    print('*** Done ***')

# We compute the output
for key in algos_used:
    algos[key].compute_outputs(Y)


## Core algorithm
# Definition of the core algo
core=DT(global_hyperparams)

# We first built a new dataset with all the predictions for the algos, it will be our new X
X_core=data.core_dataset(algos, algos_used)
Y_core=Y.loc[X_core.index]

# Again we do the rolling window estimation process
for i in range(rolling_window_size+1,len(Y_core.index)): # Note that i in the numeric index in Y of the predicted value
    recalib=(i-rolling_window_size-1)  % recalib_window == 0
    if recalib:
        train=range(i-rolling_window_size,i) # should be equal to i-rolling_window_size:i-1
        test=range(i,min(i+recalib_window,len(Y_core))) # the date we try to predict
        pred_index=Y_core.index[test] # This is the timestamp of i
        # We calibrate the hyperparameters and predict
        core.calib_predict(X_core.iloc[train], Y_core.iloc[train], X_core.iloc[test], pred_index, **default_cv_params)

# We compute the outputs
core.compute_outputs(Y_core)
            
# We check that the core algo is better than all other algos:
# careful about computing the acuracy only the same window
scoring='accuracy'
for algo in algos_used:
    if getattr(algos[key],scoring)>getattr(core,scoring): warnings.warn('Warning: {} got a better score than the core algo {}'.format(algos[key].name, core.name))

# for debug
print(core.best_hp)
pass

## Trading Strategy
TS=TradingStrat(global_hyperparams, core.get_output("predicted_values"), Y_0, threshold)

## Backtest/Plots/Trading Execution
TS.compute_output()
TS.plots()
