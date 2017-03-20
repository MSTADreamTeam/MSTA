## This file will be the main py file
## It will include all necessary dependencies and libraries, it will run all steps of the algo
## In order to compile this file into an executable file we might need an additional program

## Important note: the imported modules do not run code but just import functions

import data
from historical_mean import HM


def __main__():
## Building the dataset
    dataset=data.dataset_building(n_max=1000)

    dataset_lag=data.add_lags(dataset,lags=[1,2])

    print("hello")

    
## Creating & calibrating the different algorithms
    # First define a dictionary of algorithm associated with their names
    algos={"HM":HM(1)}    
    # Then we just allow ourselves to work/calib/fit/train only a subsets of these algos
    algos_used=["HM"]
    # Then we


    ## Core algorithm

    ## Trading Strategy

    ## Backtest/Plots/Trading

    return 0


# For debugging purposes
__main__()