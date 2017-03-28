# This file defines the Generic Algorithm iterator used in cross validation
from sklearn.model_selection import ParameterSampler

class GeneticAlgorithm:
    """ Genetic Algorithm iterator """
    def __init__(self, hp_grid, n_max, init_pop_size=None, n_parents=2):
        """ In the __init__ are defined the fixed parameters """
        self.n_max=n_max
        self.hp_grid=hp_grid
        self.init_pop_size=init_pop_size
        self.n_parents=n_parents

    def __iter__(self):
        """ The __iter__ method is defined here as a generator function 
        Here are initialized the parameters that will change at each iteration
        """
        n_iter = 0
        self.pop_scores=[]
        self.population=list(ParameterSampler(hp_grid, self.init_pop_size)) # First population is random
        while True:
            for hp in self.population:
                if n<= self.n_max: 
                    yield hp 
                else:
                    raise StopIteration
                n += 1
            self.mutate()
 
    def mutate(self):
        """ This function executes the mutation phase: it defines the new population to try """
        # need to code here
        return self    

    def update_score(self, score):
        """ This method allows for an external frame to modify the iterator during the execution of the loop
        It is used to update the score of the tested valued to do the mutation process at the next step
        """
        self.pop_scores.append(score)
        return self