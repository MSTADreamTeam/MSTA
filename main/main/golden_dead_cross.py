# This file implements the Technical Analysis Signal of Golden/Dead Cross

import numpy as np
import pandas as pd
from data import lagdf_to_ts
from base_algo import BaseAlgo, AlgoError


class GDC(BaseAlgo):
    ''' Golden/Dead Cross Technical Analysis signal
    This basic Technical Analysis algo compares the long and short term moving average 
    The hyperparamas are a,b and c and the short and long term window size'''
    
    def __init__(self, global_hyperparams, hp_grid=None, stw=None, ltw=None, a=None, b=None, c=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.algo_type='TA'
        self.name='Golden Dead Cross'
        self.a=a # a>=0, it controls the min ratio of mz/z, a=0: always accept
        self.b=b # b>=0, it controls the impact of the criteria on mz, b=0: always accept
        self.c=c # c>=0, it controls the max values of z and the min value of mz
        self.stw=stw 
        self.ltw=ltw
        self.mz=None # absolute value of the last peak
        if global_hyperparams['output_type']=='R': raise AlgoError('You cannot call GDC as a regressor')
    
    def select_data(self, X):
        ''' Here the function select data will select the prices '''
        self.selected_data=[' Ret' not in col for col in X.columns]
        return self.selected_data
    
    def predict(self, X_test, pred_index = None):
        ''' In order to transform this strategy into a clasification algorithm,
        we had to translate the generated buy/sell signals as +/-1 and the rest of the dates as 0
        The result might be a relatively sparse predictor vector '''
        X_ts=lagdf_to_ts(X_test)
        short_term_ma=X_ts.rolling(self.stw).mean()
        long_term_ma=X_ts.rolling(self.ltw).mean()
        z=long_term_ma-short_term_ma
        z=z.values[self.ltw-1:] # Speed optimization
        predicted_values=[] # We initialize the predicted values at 0 and then look for signals
        for t in range(1,len(z)):
            if z[t]*z[t-1]<0: # We record the cross
                self.mz=np.abs(z[t]) # In case of a cross we re initialize the value of the peak
            if self.mz is not None: # If we have a recorded cross we can check for signals
                self.mz=max(np.abs(z[t]), self.mz) # We update the value of the peak
                if self.mz>self.b*self.c and np.abs(z[t])<min(self.mz/self.a,self.c): # If the strenght condition is satisfied
                   predicted_values.append(np.sign(z[t])) # We create a signal
                else:
                   predicted_values.append(0)
            else:
                predicted_values.append(0)
        predicted_values=predicted_values[-len(X_test.index):] # We select the needed prediction
        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values

    
    #def fit(self, X_train, Y_train):
    #    ''' We use the fit function to record past peaks and cross
    #    This process is similar to the predict function, apart from the fact that it does not predict 
    #    ISSUE: for performance questions it would be better not to fit the model, but this could lead to missed signals
    #    due to missing history of cross
    #    '''
    #    X_ts=lagdf_to_ts(X_train) # We first transform the input into a time series
    #    short_term_ma=X_ts.rolling(self.stw).mean()
    #    long_term_ma=X_ts.rolling(self.ltw).mean()
    #    z=long_term_ma-short_term_ma
    #    z=z.values[self.ltw-1:] # Speed optimization
    #    self.mz=None
    #    for t in range(1,len(z)):
    #        if z[t]*z[t-1]<0: # We record the cross
    #            self.mz=np.abs(z[t]) # In case of a cross we re initialize the value of the peak
    #        if self.mz is not None: # If we have a recorded cross we can check for signals
    #            self.mz=max(np.abs(z[t]), self.mz) # We update the value of the peak
    #    return self
    
    