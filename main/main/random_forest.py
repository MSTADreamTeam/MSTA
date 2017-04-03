# This file inplements the Random Forest Estimator

from base_algo import BaseAlgo
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class RF(BaseAlgo):
    ''' This class implements the usual Random Forest Classifier and Regressor using sklearn implementation '''

    def __init__(self, global_hyperparams, hp_grid=None, **hyperparams):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        if global_hyperparams['output_type']=='C':
            self.model=RandomForestClassifier()
            self.name='Random Forest Classifier'
        else:
            self.model=RandomForestRegressor()
            self.name='Random Forest Regressor'
        self.algo_type='ML'
        self.set_hyperparams(**hyperparams)