# We had to recode these basic functions in order to make them compatible with our architecture

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from genetic_algorithm import GeneticAlgorithm

class CrossVal():
    ''' Cross Validation
    Search for optimal hyperparameters inside a cross validation process
    -GridSearch: 
        Performs an exhaustive search
    -RandomSearch: 
        Performs a random search
    -GeneticAlgorithm:
        Performs an optimized random search
    '''
    def __init__(self, algo, hp_grid, calib_type, cv, scoring, n_iter=None, **ga_args):
        self.cv=cv
        self.algo=algo
        self.scoring=scoring
        self.hp_grid=hp_grid
        self.best_hp={}
        if calib_type=='GridSearch':
            self.hp_iterable=ParameterGrid(self.hp_grid)
        elif calib_type=='RandomSearch':
            self.hp_iterable=ParameterSampler(self.hp_grid, n_iter)
        elif calib_type=='GeneticAlgorithm':
            self.hp_iterable=GeneticAlgorithm(self.hp_grid, n_iter, **ga_args)
    
    def compute_cv(self, X, Y):
        ''' Cross validation process '''
        best_score=-np.Inf
        # This syntax distinction is due to the face that the iterator is copied in a 'for _ in interator' statement, 
        # hence it makes it impossible to update it for the Generic Algorithm
        # We could work on rewritting this code, but it seems to be the easiest syntax as of now  
        iterator=self.hp_iterable.iter() if isinstance(self.hp_iterable, GeneticAlgorithm) else self.hp_iterable.__iter__()  
        while True:
            try:
                hp=iterator.next() if isinstance(self.hp_iterable, GeneticAlgorithm) else iterator.__next__()
            except StopIteration:
                break
            self.algo.set_hyperparams(**hp)
            score=[]
            for train, test in self.cv.split(X,Y):
                self.algo.fit(X.iloc[train], Y.iloc[train])
                pred_values=self.algo.predict(X.iloc[test]) # Be careful not to stock predicted values in the algo, since it is only temporary internal results
                self.algo.compute_outputs(Y.iloc[test], pred_values, self.scoring)
                score.append(getattr(self.algo, self.scoring))    
                self.algo.reset_outputs() # For safety, it is currently needed, please do not change without rethinking the code
            score_mean=np.mean(score)
            if isinstance(self.hp_iterable, GeneticAlgorithm): self.hp_iterable.update_score(score_mean)
            if score_mean>best_score:
                best_score=score_mean
                self.best_hp=hp
        return self

# Old syntax with copied iterator problem
    #def compute_cv(self, x, y):
    #    ''' cross validation process '''
    #    best_score=-np.inf
    #    for hp in self.hp_iterable: # here is the problem, this line copy the iterable object, making it immutable
    #        self.algo.set_hyperparams(**hp)
    #        score=[]
    #        for train, test in self.cv.split(x,y):
    #            self.algo.fit(x.iloc[train], y.iloc[train])
    #            pred_values=self.algo.predict(x.iloc[test]) # be careful not to stock predicted values in the algo, since it is only temporary internal results
    #            self.algo.compute_outputs(y.iloc[test], pred_values, self.scoring)
    #            score.append(getattr(self.algo, self.scoring))    
    #            self.algo.reset_outputs() # for safety, it is currently needed, please do not change without rethinking the code
    #        score_mean=np.mean(score)
    #        if isinstance(self.hp_iterable, geneticalgorithm): self.hp_iterable.update_score(score_mean)
    #        if score_mean>best_score:
    #            best_score=score_mean
    #            self.best_hp=hp
    #    return self

