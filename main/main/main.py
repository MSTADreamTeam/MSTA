## This file will be the main py file
## It will include all necessary dependencies and libraries, it will run all steps of the algo
## In order to compile this file into an executable file we might need an additional program

## Important note: the imported modules do not run code but just import functions

import data
from historical_mean import HM


def __main__():
## Building the dataset
    dataset0=data.dataset_building(n_max=1000)
    # We select an asset returns time series to predict from the dataset
    dataset=dataset0[["EURUSD Curncy"]]

    # With lags
    dataset_lag=data.lagged(dataset,lags=[1])
    
    # We define the testing and training set
    Y_train=dataset].iloc[:700]
    Y_test=dataset.iloc[701:]
    X_train=dataset_lag.iloc[:700]
    X_test=dataset_lag.iloc[701:]

## Creating & calibrating the different algorithms

    # First define a dictionary of algorithm associated with their names
    algos={"HM AR 10":HM(10),
           "HM GEO 10":HM(10,mean_type="geometric")}    
    
    # Then we just allow ourselves to work/calib/fit/train only a subsets of these algos
    algos_used=algos.keys()

    # We let each algo select the relevent data to work on
    for key in algos_used:
        algos[key].select_data(dataset_lags)

    # We train all the algos on the testing set, it includes the calibration of hyperparameters
    for key in algos_used:
        algos[key].train(Y_train,X_train)
        
    # We build the predictions
    for key in algos_used:
        algos[key].predict() # WORK ON THIS PART

## Core algorithm

## Trading Strategy

## Backtest/Plots/Trading Execution

    return 0


# For debugging purposes we run the main function
__main__()