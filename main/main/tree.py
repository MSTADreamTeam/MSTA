from base_algo import BaseAlgo
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class DT(BaseAlgo):
    ''' This class implements the usual Decision Tree Classifier and Regressor using sklearn implementation '''

    def __init__(self, global_hyperparams, hp_grid=None, **hyperparams):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        if global_hyperparams['output_type']=='C':
            self.model=DecisionTreeClassifier()
            self.name='Decision Tree Classifier'
        else:
            self.model=DecisionTreeRegressor()
            self.name='Decision Tree Regressor'
        self.algo_type='ML'
        self.set_hyperparams(**hyperparams)