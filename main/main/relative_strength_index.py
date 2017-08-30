# RSI Techinical Analysis signal

from base_algo import BaseAlgo, AlgoError

class RSI(BaseAlgo):
    ''' Relative Strength Index Technical Analysis signal '''
    def __init__(self, global_hyperparams, hp_grid=None, ob=80, os=20, w=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.name='Relative Strength Index'
        self.algo_type='TA'
        self.ob=ob # The overbought level for the RSI
        self.os=os # The oversold level for the RSI
        self.w=w if w is not None else global_hyperparams['rolling_window_size'] # The window size of the RS
        if global_hyperparams['output_type']=='R': raise AlgoError('You cannot call RSI as a regressor')

    def select_data(self, X):
        ''' This TA algo works with returns '''
        self.selected_data=[' Ret' in col for col in X.columns]
        return self.selected_data

    def predict(self, X_test, pred_index = None):
        ''' The signal is defined when the RSI breaks ob or os '''
        RS=(X_test*(X_test>0)).iloc[:,:self.w].mean(axis=1)/abs((X_test*(X_test<0)).iloc[:,:self.w].mean(axis=1))
        RSIndex=100*(1-1/(1+RS))
        predicted_values=1*(RSIndex<self.os)-1*(RSIndex>self.ob)

        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values 

