# This algorithm will just take the historical mean as a prediction
# It will be used as a benchmark for prediction as it represents the random walk hypothesis

from generic_algo import gen_algo
import pandas as pd
import numpy as np

class HM(gen_algo):
    def __init__(self, global_hyperparams, mean_type="arithmetic", window_size=None):
        gen_algo.__init__(self, global_hyperparams) # allow to run the init of the gen_algo class, and define all default arguments
        self.name="Historical Mean"
        self.algo_type="BA" # By convention
        self.mean_type=mean_type
        if window_size is not None:   # It is possible to define a window size different from the global rolling window size, but it has to be less or equal
            self.window_size=window_size
        else:
            self.window_size=self.global_hyperparams["rolling_window_size"]
            
    def predict(self, X_test, pred_index):
        w = self.window_size
        if self.mean_type=="arithmetic":
            predicted_value=X_test.iloc[-w:].mean(axis=0,skipna=None)
        elif self.mean_type=="geometric":
            predicted_value=1
            for idx in X_test.iloc[-w:].index:
                predicted_value=predicted_value*(1+X_test.loc[idx])
            predicted_value=np.power(predicted_value,1/w)-1
            
        # The output will be different in case of a regression or classification, no need to change the output for a regression
        if self.global_hyperparams["output_type"]=="C":
            threshold=self.global_hyperparams["threshold"]
            if threshold==0:
                predicted_value=np.sign(predicted_value)
            else:
                predicted_value=0 if abs(predicted_value)<threshold else predicted_value
                predicted_value=np.sign(predicted_value)
            
        self.predicted_values[pred_index]=predicted_value
        return predicted_value # here we have a redundency in the return and the side effect of the method, this is used to simplify coding

## testing code
#import matplotlib.pyplot as pyplt
#import data

#dataset=data.dataset_building(n_max=1000)
#data_test=dataset[["EURUSD Curncy"]]
#hm=HM(100)
#hm2=HM(100,mean_type="geometric")
#res=hm.predict(data_test)
#res2=hm2.predict(data_test)
#pyplt.plot(data_test)
#pyplt.plot(res)
#pyplt.plot(res2)
#pyplt.show()
## end of testing code
    