# This file implements the Multi Layer Perceptron using sklearn library


from generic_algo import gen_algo
from sklearn.neural_network import MLPClassifier, MLPRegressor

class MLP(gen_algo):
    ''' This class implements the Multi Layer Perception Classifier and Regressor using sklearn implementation '''

    def __init__(self, global_hyperparams, hp_grid=None, **hyperparams):
        gen_algo.__init__(self, global_hyperparams, hp_grid)
        if global_hyperparams['output_type']=='C':
            self.model=MLPClassifier()
            self.name='Multi Layer Perceptron Classifier'
        else:
            self.model=MLPRegressor()
            self.name='Multi Layer Perception Regressor'
        self.algo_type='ML'
        self.set_hyperparams(**hyperparams)