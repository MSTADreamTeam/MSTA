# This file implemesnts the Technical Analysis Signal of Golden/Dead Cross

from generic_algo import gen_algo, AlgoError

class GDC(gen_algo):
    ''' Golden/Dead Cross Technical Analysis signal
    This basic Technical Analysis algo compares the long and short term moving average 
    The hyperparamas are a,b and c and the short and longg term window size'''
    
    def __init__(self, global_hyperparams, hp_grid = None, stw=None, ltw=None, a=None, b=None, c=None):
        gen_algo.__init__(self, global_hyperparams, hp_grid)
        self.algo_type='TA'
        self.a=a
        self.b=b
        self.c=c
        self.stw=stw 
        self.ltw=ltw
        if global_hyperparams['output_type']=='R': raise AlgoError('You cannot call GDC as a regressor')
    

    def predict(self, X_test, pred_index = None):
    
        predicted_values=None

        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values

    def fit(self, X_train, Y_train):
        return super().fit(X_train, Y_train)