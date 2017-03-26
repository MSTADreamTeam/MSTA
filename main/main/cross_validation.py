# Cross Validation general class and subclass GridSearch, RandomSearch, and GeneticAlgorithm
# We had to recode these basic functions because of our use of new estimators

from generic_algo import gen_algo

class CrossVal():
    def __init__(self, algo: gen_algo, cv, scoring):
        self.cv=cv
        self.algo=algo
        self.best_estimator=None        
        self.scoring=scoring

    def fit(self, X_train, Y_train):
        return self
    

class GridSearch(CrossVal):
    def __init__(self, cv, scoring):
        CrossVal.__init__(self, cv, scoring)
        
    def fit(self, X_train, Y_train):
        best_score=0
        for train, test in cv.split(X_train,Y_train):
            algo.fit(X_train[train], Y_train[train])
            # for the prediction check that predicting with an array as an input works
            algo.predict(X_train[test],X_train[test].index)
            algo.compute_outputs(Y_train[test])
            if self.algo.global_hyperparams["output_type"]==""
            best_score=best_score if  




        return self
