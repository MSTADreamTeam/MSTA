# We had to recode these basic functions in order to make them compatible with our architecture

from sklearn.model_selection import ParameterGrid, ParameterSampler
import numpy as np

class CrossVal():
    """ Cross Validation general class for GridSearch, RandomSearch, and GeneticAlgorithm
        GridSearch 
    Performs an Exhaustive Search for optimal hyperparameters inside a cross validation process
        RandomSearch 
    Performs a Random Search for optimal hyperparameters inside a cross validation process
        GeneticAlgorithm
        NOT CODED YET
    """
    def __init__(self, algo, calib_type, hp_grid, cv, scoring, n_iter=None):
        self.cv=cv
        self.algo=algo
        self.scoring=scoring
        self.hp_grid=hp_grid
        self.best_hp={}
        if calib_type=='GridSearch':
            self.hp_iterator=ParameterGrid(self.hp_grid)
        elif calib_type=='RandomSearch':
            self.hp_iterator=ParameterSampler(self.hp_grid, n_iter)
        elif calib_type=='GeneticAlgorithm':
            self.hp_iterator=None # Not coded yet

    
    def compute_cv(self, X, Y):
        """ Function that operate the cross validation
        Let us notive that this code is similar for the three different cross validation
        However the iterator that will yield the tested parameters differs
        """
        best_score=-np.Inf
        for hp in self.hp_iterator: 
            self.algo.set_hyperparams(**hp)
            score=[]
            for train, test in self.cv.split(X,Y):
                self.algo.fit(X.iloc[train], Y.iloc[train])
                pred_values=self.algo.predict(X.iloc[test]) # Be careful not to stock predicted values in the algo, since it is only temporary internal results
                self.algo.compute_outputs(Y.iloc[test], pred_values, self.scoring)
                score.append(getattr(self.algo, self.scoring))    
                self.algo.reset_outputs() # For safety, it is currently needed, please do not change without rethinking the code
            score_mean=np.mean(score)
            if score_mean>best_score:
                best_score=score_mean
                self.best_hp=hp
        return self