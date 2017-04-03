# Basic core algorithms 

import numpy as np
from base_algo import BaseAlgo
from data import to_class

class CMW(BaseAlgo):
    ''' Core Manual Weighting: This algo used as a core algo just manually fixes the weights of the prediction '''
    def __init__(self, global_hyperparams, weights):
        BaseAlgo.__init__(self, global_hyperparams)
        self.algo_type='BA'
        self.weights=weights
        self.name='Core Manual Weighting'

    def predict(self, X_test, pred_index=None):
        predicted_values=X_test*self.weights
        if self.global_hyperparams['output_type']=='C':
            predicted_values=to_class(predicted_values, self.global_hyperparams['threshold'])    
        if pred_index is not None:
            self._store_predicted_values(pred_index, predicted_values)
        return predicted_values        

class BIS(BaseAlgo):
    ''' Best In Sample: This algo used as a core algo just pick the best selection IS to predict OOS '''
    def __init__(self, global_hyperparams, scoring):
        BaseAlgo.__init__(self, global_hyperparams)
        self.algo_type='BA'
        self.scoring=scoring
        self.index_best=None
        self.name='Best In Sample'

    def fit(self, X_train, Y_train):
        ''' The fit will select the best predictor IS given a soring metric '''
        best_score=-np.Inf
        for col in X_train.columns:
            self.compute_outputs(Y_train, X_train[col], self.scoring)
            score=getattr(self, self.scoring)
            if score>best_score:
                best_score=score
                self.index_best=col
        return self
    
    def predict(self, X_test, pred_index=None):
        predicted_values=X_test[self.index_best]        
        if pred_index is not None:
            self._store_predicted_values(pred_index, predicted_values)
        return predicted_values

