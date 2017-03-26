# We had to recode these basic 

from generic_algo import gen_algo
import numpy as np

class CrossVal():
    """ Cross Validation general class for GridSearch, RandomSearch, and GeneticAlgorithm """

    def __init__(self, algo: gen_algo, hp_grid, cv, scoring):
        self.cv=cv
        self.algo=algo
        self.scoring=scoring
        self.hp_grid=hp_grid
        self.best_hp={}
    
    def compute_cv(self, X, Y):
        return self
    

class GridSearch(CrossVal):
    """ GridSearch """
    def __init__(self, algo, hp_grid, cv, scoring):
        CrossVal.__init__(self, algo, hp_grid, cv, scoring)
        
    def compute_cv(self, X, Y):
        best_score=0
        for hp in hp_grid:
            algo.set_params(hp)
            score=[]
            for train, test in cv.split(X,Y):
                algo.fit(X[train], Y[train])
                # for the prediction check that predicting with an array as an input works
                algo.predict(X[test],X[test].index)
                algo.compute_outputs(Y[test])
                score.append(getattr(self.algo, scoring))    
            score_mean=best_hp.mean(score)
            if score_mean>best_score:
                best_score=score_mean
                best_hp=hp
        return self

class RandomSearch(CrossVal):
    """ RandomSearch 
        NOT CODED YET
    """
    def __init__(self, algo, hp_grid, cv, scoring, n_iter):
        CrossVal.__init__(self, algo, hp_grid, cv, scoring)
        self.n_iter=n_iter
    
    def compute_cv(self, X, Y):
        return super().compute_cv(X, Y)


class GeneticAlgorithm(CrossVal):
    """ GeneticAlgorithm
        NOT CODED YET
    """
    def __init__(self, algo, hp_grid, cv, scoring):
        CrossVal.__init__(self, algo, hp_grid, cv, scoring)
    
    def compute_cv(self, X, Y):
        return super().compute_cv(X, Y)
