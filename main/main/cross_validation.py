# We had to recode these basic 

from generic_algo import gen_algo
from sklearn.model_selection import ParameterGrid, ParameterSampler
import numpy as np

class CrossVal():
    """ Cross Validation general class for GridSearch, RandomSearch, and GeneticAlgorithm """

    def __init__(self, algo: gen_algo, hp_grid, cv, scoring):
        self.cv=cv
        self.algo=algo
        self.scoring=scoring
        self.hp_grid=hp_grid
        self.best_hp={}
        self.hp_iterator=None
    
    def compute_cv(self, X, Y):
        """ Function that operate the cross validation
        Let us notive that this code is similar for the three different cross validation
        However the iterator that will yields the tested parameters will be different
        """
        best_score=-np.Inf
        for hp in self.hp_iterator: 
            algo.set_params(hp)
            score=[]
            for train, test in cv.split(X,Y):
                algo.fit(X[train], Y[train])
                # for the prediction check that predicting with an array as an input works
                algo.predict(X[test],X[test].index)
                algo.compute_outputs(Y[test], scoring)
                score.append(getattr(self.algo, scoring))    
            score_mean=best_hp.mean(score)
            if score_mean>best_score:
                best_score=score_mean
                best_hp=hp
        return self

    

class GridSearch(CrossVal):
    """ GridSearch 
    Performs an Exhoustive Search for optimal hyperparameters inside a cross validation process
    Defined as a new class, but could be defined just as an instance of CrossVal
    """
    def __init__(self, algo, hp_grid, cv, scoring):
        CrossVal.__init__(self, algo, hp_grid, cv, scoring)
        self.hp_iterator=ParameterGrid(self.hp_grid) # This is an exhaustive iterator defined by sklearn

class RandomSearch(CrossVal):
    """ RandomSearch 
    Performs a Random Search for optimal hyperparameters inside a cross validation process
    Defined as a new class, but could be defined just as an instance of CrossVal
    """
    def __init__(self, algo, hp_grid, cv, scoring, n_iter):
        CrossVal.__init__(self, algo, hp_grid, cv, scoring)
        self.hp_iterator=ParameterSampler(self.hp_grid, n_iter) # Random Sampler, let us notice that we could use different distributions and give a full continuous range as input

class GeneticAlgorithm(CrossVal):
    """ GeneticAlgorithm
        NOT CODED YET
    """
    def __init__(self, algo, hp_grid, cv, scoring):
        CrossVal.__init__(self, algo, hp_grid, cv, scoring)
        self.hp_iterator=None
    
    def compute_cv(self, X, Y):
        return super().compute_cv(X, Y)
