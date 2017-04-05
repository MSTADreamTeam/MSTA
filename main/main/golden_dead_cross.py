# This file implements the Technical Analysis Signal of Golden/Dead Cross

import numpy as np
import pandas as pd
from base_algo import BaseAlgo, AlgoError


class GDC(BaseAlgo):
    ''' Golden/Dead Cross Technical Analysis signal
    This basic Technical Analysis algo compares the long and short term moving average 
    The hyperparamas are a,b and c and the short and long term window size'''
    
    def __init__(self, global_hyperparams, hp_grid=None, stw=None, ltw=None, a=None, b=None, c=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.algo_type='TA'
        self.name='Golden Dead Cross'
        self.a=a
        self.b=b
        self.c=c
        self.stw=stw 
        self.ltw=ltw
        self.mz=None 
        if global_hyperparams['output_type']=='R': raise AlgoError('You cannot call GDC as a regressor')
    
    def select_data(self, X):
        ''' Here the function select data will select the prices '''
        self.selected_data=[' Ret' not in col for col in X.columns]
        return self.selected_data
    
    def predict(self, X_test, pred_index = None):
        ''' In order to transform this strategy into a clasification algorithm,
        we had to translate the generated buy/sell signals as +/-1 and the rest of the dates as 0
        The result might be a relatively sparse predictor vector '''
        '''
        short_term_ma=pd.DataFrame(X_test.iloc[:,i:i+self.stw].mean(axis=1,skipna=None) for i in range(0,len(X_test.columns)-self.stw)).transpose()
        long_term_ma=pd.DataFrame(X_test.iloc[:,i:i+self.ltw].mean(axis=1,skipna=None) for i in range(0,len(X_test.columns)-self.ltw)).transpose()
        z=long_term_ma-short_term_ma
        z=z[z.columns[::-1]] # Reverse the columns of z so that they are time ascending
        predicted_values=pd.Series(data=0, index=X_test.index) # We initialize the predicted values at 0 and then look for signals
        for t in range(1,len(z.columns)):
            if z.iloc[:,t]*z.iloc[:,t-1]<0: # We record the cross
                self.mz=np.abs(z.iloc[:,t]) # In case of a cross we re initialize the value of the peak
            if self.mz is not None: # If we have a recorded cross we can check for signals
                self.mz=max(np.abs(z.iloc[:,t]), self.mz) # We update the value of the peak
                if self.mz>self.b*self.c and np.abs(z.iloc[:,t])<min(self.mz/self.a,self.c): # If the strenght condition is satisfied
                   predicted_values.iloc[:,t]=np.sign(z.iloc[:,t]) # We create a signal
        '''
        short_term_ma=pd.Series(X_test.iloc[:,i:i+self.stw].mean(axis=1,skipna=None) for i in range(0,len(X_test.columns)-self.stw)).transpose()
        long_term_ma=pd.Series(X_test.iloc[:,i:i+self.ltw].mean(axis=1,skipna=None) for i in range(0,len(X_test.columns)-self.ltw)).transpose()
        z=long_term_ma-short_term_ma
        z.dropna(inplace=True)
        z=z[z.index[::-1]]
        
       # NEED TO WORK HERE


        predicted_values=None
        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values
