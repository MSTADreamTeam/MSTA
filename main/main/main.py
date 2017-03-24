## This file will be the main py file
## It will include all necessary dependencies and libraries, it will run all steps of the algo
## In order to compile this file into an executable file we might need an additional program

## Important note: the imported modules do not run code but just import function and class

import data
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
    dataset=data.dataset_building(n_max=1000)
    
    # We select an asset returns time series to predict from the dataset
    asset_label="EURUSD Curncy"
    Y=dataset[[asset_label]]
    Y.dropna(inplace=True)

    # With lags, used as X, maybe this implementation is not optimal, think about a slicing way to do that?
    lags=range(1,rolling_window_size+1)
    X=data.lagged(Y,lags=lags)
    max_lags=max(lags)
    
## Creating & calibrating the different algorithms

    # First define a dictionary of algorithm associated with their names
    algos={"HM AR Full window":HM(global_hyperparams),
           "HM GEO Full window":HM(global_hyperparams,mean_type="geometric"),
           "HM AR Short Term":HM(global_hyperparams,window_size=10),
           "HM GEO Short Term":HM(global_hyperparams,mean_type="geometric",window_size=10),
           "LR":LR(global_hyperparams)}
    
    # Then we just allow ourselves to work/calib/fit/train only a subsets of these algos
    #algos_used=algos.keys()
    algos_used=["HM AR Full window","LR"]

    for key in algos_used:
        # We let each algo select the relevent data to work on
        algos[key].select_data(X)

        for i in range(rolling_window_size+max_lags,len(Y.index)): # Note that i in the numeric index in Y of the predicted value
            X_train=X.iloc[i-rolling_window_size:i-1] # Feel free to check the indices i am very bad at that shit it is killing me ffs
            Y_train=Y.iloc[i-rolling_window_size:i-1] # For example there are NaN in X_train at the beggining, needs further checking
            X_test=X.iloc[i]
            Y_test=Y.iloc[i]
            pred_index=Y.index[i] # This is the timestamp of i

            # We train all the algos on the testing set, it includes the calibration of hyperparameters
            algos[key].train(X_train,Y_train)

            # We build the predictions
            algos[key].predict(X_test,pred_index)

            # We compute the outputs
            # algos[key].compute_outputs(Y,pred_index)
            
            # for debug
            print(i)
       # print(algos[key].predicted_values)

## Core algorithm

## Trading Strategy

## Backtest/Plots/Trading Execution

    return 0


# For debugging purposes we run the main function
__main__()