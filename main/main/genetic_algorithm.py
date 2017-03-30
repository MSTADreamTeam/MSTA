# This file defines the Generic Algorithm iterator used in cross validation
import numpy as np
from sklearn.model_selection import ParameterSampler
from random import sample

class GeneticAlgorithm:
    """ Genetic Algorithm iterator """
    def __init__(self, hp_grid, n_max, init_pop_size=None, select_rate=0.5, mixing_ratio=0.5, mutation_proba=0.1):
        """ In the __init__ are defined the fixed parameters """
        self.n_max=n_max
        self.hp_grid=hp_grid
        self.init_pop_size=init_pop_size
        self.mutation_proba=mutation_proba
        self.select_rate=select_rate
        

    def __iter__(self):
        """ The __iter__ method is defined here as a generator function 
        Here are initialized the parameters that will change at each iteration
        """
        n_iter = 0
        self.pop_scores=[]
        self.population=list(ParameterSampler(hp_grid, self.init_pop_size)) # First population is random, we turn it into list to fix it
        self.generation=0
        while True:
            for hp in self.population:
                if n<= self.n_max: 
                    yield hp 
                else:
                    break
                n += 1
            self.selection()
            self.crossover()
            self.mutate()
            self.generation += 1
        raise StopIteration
    
    def selection(self):
        """ This function executes the selection phase
        It will select a subset of the population as the survivors
        to be used in the mutation and crossover processes
        The seection is based on scoring
        Hyperparameter: The selection rate, which determines the exponential rate at which the population 
        will decrease generation after generation.
        """
        list_to_sort=[(hp,score) for hp in zip(self.pop_scores,self.population)]
        self.population=np.sort(list_to_sort, axis=1)[:floor(len(self.population)*self.select_rate)][0]
        self.pop_scores=[]
        return self 

    def crossover(self):
        """ This fuction executes the crossover phase
        It will create a new population by genetically mixing the past population
        Each new population member will inherit from a fixed number of randomly chosen parents
        This mixing allow the convergence to the optimum
        Hyperparameter: the mixing rate, which determines the proportion of
        changed features from one generation to another
        We fixed the number of parents to 2 in this implementation
        """
        new_pop=[]
        while len(new_pop)<=len(self.population):
            parents=sample(self.population, 2) # We select two random parents, note that a parent can be selected several times
            crossover_points=sample(hp.keys(), floor(self.mixing_ratio*len(hp))) # this defines the keys of the hyperparameters that will be mixed
            temp={key:parents[0][key] for key in crossover_points}
            new_pop.append(parrent[0].update({key:parents[1][key] for key in crossover_points}))
            new_pop.append(parrent[1].update(temp))
        self.population=new_pop
        return self

    def mutate(self):
        """ This function executes the mutation phase:
        It will randomly change a small subset of the population
        It allows the algorith to avoid local optimum
        Hyperparameters: the mutation probability of a population member, it should be very low, 10% max
         """
        # need to code here
        return self    

        

    def update_score(self, score):
        """ This method allows for an external frame to modify the iterator during the execution of the loop
        It is used to update the score of the tested valued to do the mutation process at the next step
        """
        self.pop_scores.append(score)
        return self