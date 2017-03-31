# This file will define the trading strategy
from data import to_class


class TradingStrat():
    ''' Tradig Strategy class
    Here is defined the trading tragegy given a set of risk limits 
    '''

    def __init__(self, global_hyperparams, pred_val, threshold=None):
        self.pred_val=pred_val    
        self.output_type=global_hyperparams['output_type']
        self.threshold=threshold if threshold is not None else global_hyperparams['threshold']
        self.pred_val_class=pred_val if self.output_type=='C' else to_class(pred_val, threshold) 
            
        for idx in pred_val.index:
            pass

    def compute_results(self, Y):
        ''' Compute the results of the Trading Strategy '''
        # deprecated, made by zipline
        return self

    def plots(self, Y):
        # deprecated, made by zipline
        return self

    def generate_log(self):
        ''' This function generates a trading log that can be used for other applications ''' 
        return self