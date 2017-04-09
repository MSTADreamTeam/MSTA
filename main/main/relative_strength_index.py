# RSI Techinical Analysis signal

from base_algo import BaseAlgo

class RSI(BaseAlgo):
    ''' Relative Strength Index Technical Analysis signal '''
    def __init__(self, global_hyperparams, hp_grid=None, ob=80, os=20, s_ob=0, s_os=0, w=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.name='Relative Strength Index'
        self.algo_type='TA'
        self.ob=ob # The overbought level for the RSI
        self.os=os # The oversold level for the RSI
        self.s_ob=s_ob # The slope for the overbought signal
        self.s_os=s_os # The slope for the oversold signal
        self.w=w if w is not None else global_hyperparams['rolling_window_size'] # The window size of the RS

    def select_data(self, X):
        ''' This TA algo works with returns '''
        self.selected_data=[' Ret' in col for col in X.columns]
        return self.selected_data

    def predict(self, X_test, pred_index = None):
        ''' The signal is defined when the RSI breaks ob or os,
        then it needs to cross the defined slopes for the signal to be activated  '''
        RS=(X_test*(X_test>0)).iloc[:,:self.w].mean(axis=1)/(X_test*(X_test<0)).iloc[:,:self.w].mean(axis=1)
        RSIndex=100*(1-1/(1+RS))
        predicted_values=None

        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values 

