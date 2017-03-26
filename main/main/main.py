## This file will be the main py file
## It will include all necessary dependencies and libraries, it will run all steps of the algo
## In order to compile this file into an executable file we might need an additional program

## Important note: the imported modules do not run code but just import function and class

import data
import numpy as np
from historical_mean import HM
from linear_regression import LR


def __main__():
## Global Hyperparameters
    # The window size of the rolling window used to define each training set size
    # The models will never see more than this number of points at once
    rolling_window_size=100
    
    # Output type : C for Classification, R for Regression
    # Note that for a Classification, 1 means positive return, -1 means negative, and 0 means bellow threshold
    output_type="C"
    # In case of 3 class Classification, please provide an absolute level for the zero return threshold
    # Note that this global hyperparameter might be included as a locally tuned hyperparameters in future releases
    # Fix it to 0 for a binary classification
    # The optimal value can also be optimized as a result of the PnL optimisation and will be function of the volatility of the asset
    threshold=0.001

    # This dictionary of global hyperparameters will be passed an an argument of all built algorithms
    global_hyperparams={"rolling_window_size":rolling_window_size,
                        "output_type":output_type,
                        "threshold":threshold}    

## Building the dataset
    dataset=data.dataset_building(n_max=500)
    
    # We select an asset returns time series to predict from the dataset
    asset_label="EURUSD Curncy"
    Y=dataset[[asset_label]]
    Y.dropna(inplace=True)
    
    # With lags, used as X, maybe this implementation is not optimal, think about a slicing way to do that?
    lags=range(1,rolling_window_size+1)
    X=data.lagged(Y,lags=lags)
    max_lags=max(lags)
    # We could also turn X into classes data, is that meaningful?
    # X=to_class(X,threshold)    

    # In case of classification, we transform Y and put the classes labels into global_hyperparams 
    if output_type=="C":
        Y=data.to_class(Y, threshold) # Notice that now the values of Y is the index of the class in classes
        classes=np.unique(Y)
        global_hyperparams["classes"]=classes


## Creating & calibrating the different algorithms

    # First define a dictionary of algorithm associated with their names
    # As arguments please include the fixed hyperparams of the model as a named argument
    # For the hyperparameters grid to use in cross validation please provide a dictionary using sklearn syntax 
    algos={"HM AR Full window":HM(global_hyperparams,window_size=10),
           "HM GEO Full window":HM(global_hyperparams,mean_type="geometric"),
           "HM AR Short Term":HM(global_hyperparams,window_size=10),
           "LR":LR(global_hyperparams),
           "Lasso":LR(global_hyperparams, regularization="Lasso",hp_grid={"alpha":np.logspace(-5,0,10)})}
    
    # Then we just allow ourselves to work/calib/fit/train only a subsets of these algos
    #algos_used=algos.keys()
    algos_used=["Lasso"]

    for key in algos_used:
        # We let each algo select the relevent data to work on
        algos[key].select_data(X)

        for i in range(rolling_window_size+max_lags,len(Y.index)): # Note that i in the numeric index in Y of the predicted value
            train=range(i-rolling_window_size,i) # should be equal to i-rolling_window_size:i-1
            test=i # I am not sure of the index, we can check
            pred_index=Y.index[test] # This is the timestamp of i

            # We train all the algos on the testing set, it includes the calibration of hyperparameters
            algos[key].calib(X[train],Y[train],pred_index, cross_val_type="ts_cv",n_splits=5,calib_type="GridSearch")

            # We build the predictions
            algos[key].predict(X[test],pred_index)

            # for debug
            print(i)

        # We compute the outputs
        algos[key].compute_outputs(Y)
            
        # for debug
        print(algos[key].predicted_values)

## Core algorithm

## Trading Strategy

## Backtest/Plots/Trading Execution

    return 0


# For debugging purposes we run the main function
__main__()