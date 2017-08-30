# This file will define the trading strategy
from data import to_class

import numpy as np
import datetime
from plotly.offline import plot, iplot
import plotly.graph_objs as go

class TradingStrat():
    ''' Tradig Strategy class
    Here is defined the trading tragegy 
    '''
    def __init__(self, global_hyperparams, pred_val, asset_data, threshold=None):
        self.pred_val=pred_val    
        self.output_type=global_hyperparams['output_type']
        self.threshold=threshold if threshold is not None else global_hyperparams['threshold']
        self.pred_val_class=pred_val if self.output_type=='C' else to_class(pred_val, threshold)         
        self.pred_val_class=self.pred_val_class.squeeze()
        self.asset_data=asset_data[self.pred_val_class.index]

    def plots(self):
        trace1=go.Scattergl(y=self.cumret,x=self.cumret.index, name='Cumulative returns')
        layout=go.Layout(showlegend=True)
        fig=go.Figure(data=[trace1],layout=layout)
        plot(fig)
        return self

    def compute_output(self):
        self.ret=self.pred_val_class.multiply(self.asset_data, axis=0)
        self.cumret=self.ret.cumsum()
        self.beta=self.ret.cov(self.asset_data)
        self.alpha=np.mean(self.ret)-self.beta*np.mean(self.asset_data) # anualized?
        nb_y=(self.ret.index[-1]-self.ret.index[0])/datetime.timedelta(days=365)
        self.vol=np.std(self.ret)*np.sqrt(252) # annualized vol
        self.ann_cumret=self.cumret[-1]/nb_y
        self.sharpe=self.ann_cumret/self.vol