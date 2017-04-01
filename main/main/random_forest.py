# This file inplements the Random Forest Estimator

from generic_algo import gen_algo
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class RF(gen_algo):
    ''' This class implements the usual Random Forest Classifier and Regressor using sklearn implementation '''

    def __init__(self, global_hyperparams, hp_grid=None, **hyperparams):
        gen_algo.__init__(self, global_hyperparams, hp_grid)
        if global_hyperparams['output_type']=='C':
            self.model=RandomForestClassifier()
            self.name='Random Forest Classifier'
        else:
            self.model=RandomForestRegressor()
            self.name='Random Forest Regressor'
        self.algo_type='ML'
        self.set_hyperparams(**hyperparams)