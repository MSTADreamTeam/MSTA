# This file will describe the MA enveloppe Technical Analysis signal

from base_algo import BaseAlgo, AlgoError

class MAE(BaseAlgo):
    ''' MA Enveloppe Technical Analysis signal '''
    def __init__(self, global_hyperparams, hp_grid=None, p1=0.01, p2=0.01, w=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.name='MA Enveloppe'
        self.algo_type='TA'
        self.p1=p1
        self.p2=p2
        self.w=w if w is not None else global_hyperparams['rolling_window_size']
        if global_hyperparams['output_type']=='R': raise AlgoError('You cannot call MAE as a regressor')

    def predict(self, X_test, pred_index = None):
        ''' The signal is defined when the price breaks the upper or lower MA enveloppe '''
        ma=X_test.iloc[:,:self.w].mean(axis=1,skipna=None)
        p2=self.p1 if self.p2 is None else self.p2
        predicted_values=-1*(X_test.iloc[:,0]>ma*(1+self.p1))+1*(X_test.iloc[:,0]<ma*(1-p2)) 
        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values 

        