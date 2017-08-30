# This file will describe the Momentum and ROC signals

from base_algo import BaseAlgo, AlgoError

class MOM(BaseAlgo):
    ''' Momentum Technical Analysis signal '''
    def __init__(self, global_hyperparams, hp_grid=None, n=1):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.name='Momentum'
        self.algo_type='TA'
        self.n=n
        if global_hyperparams['output_type']=='R': raise AlgoError('You cannot call MOM as a regressor')

    def predict(self, X_test, pred_index = None):
        mom=X_test.iloc[:,0]-X_test.iloc[:,self.n]
        predicted_values=2*(mom>0)-1 
        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values 

class ROC(BaseAlgo):
    ''' Rate Of Change Technical Analysis signal '''
    def __init__(self, global_hyperparams, hp_grid=None, n=1):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.name='Rate Of Change'
        self.algo_type='TA'
        self.n=n
        if global_hyperparams['output_type']=='R': raise AlgoError('You cannot call ROC as a regressor')

    def predict(self, X_test, pred_index = None):
        roc=(X_test.iloc[:,0]/X_test.iloc[:,self.n])-1
        predicted_values=2*(roc>roc.shift(1))-1 
        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values 
