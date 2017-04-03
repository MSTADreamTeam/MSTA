# This file implements the AdaBoost Regresor and Classifier using sklearn library

from base_algo import BaseAlgo
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.base import clone

class ADAB(BaseAlgo):
    ''' This class implements the Adaptative Boosting Classifier and Regressor using sklearn implementation '''

    def __init__(self, global_hyperparams, hp_grid=None, base_algo=None, **hyperparams):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        if global_hyperparams['output_type']=='C':
            self.model=AdaBoostClassifier()
            self.name='Adaptative Boosting Classifier'
        else:
            self.model=AdaBoostRegressor()
            self.name='Adaptative Boosting Regressor'
        self.algo_type='ML'
        self.set_hyperparams(**hyperparams)
        if base_algo is not None: self.model.base_estimator=clone(base_algo.model) # Allows you to copy the model from another algo
