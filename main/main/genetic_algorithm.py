# This file defines the Generic Algorithm iterator used in cross validation
import numpy as np
from sklearn.model_selection import ParameterSampler
from random import sample

class GeneticAlgorithm:
    """ Genetic Algorithm iterator """
    def __init__(self, hp_grid, n_max, init_pop_size=None, select_rate=0.5, mixing_ratio=0.5, mutation_proba=0.1, variance_ratio=0.1):
        """ In the __init__ are defined the fixed parameters """
        self.n_max=n_max
        self.hp_grid=hp_grid
        self.init_pop_size=init_pop_size
        self.mutation_proba=mutation_proba
        self.select_rate=select_rate
        self.vars={key:self.variance_ratio*len(self.hp_grid[key]) for key in self.hp_grid}  # Used in the mutation phase      

    def __iter__(self):
        """ The __iter__ method is defined here as a generator function 
        Here are initialized the parameters that will change at each iteration
        """
        n_iter = 0
        self.pop_scores=[]
        self.population=list(ParameterSampler(self.hp_grid, self.init_pop_size)) # First population is random, we turn it into list to fix it
        self.generation=0
        while True:
            for hp in self.population:
                if n<= self.n_max: 
                    yield hp 
                else:
                    break
                n_iter += 1
            self.selection()
            self.crossover()
            self.mutate()
            self.generation += 1
        raise StopIteration
    
    def selection(self):
        """ This function executes the selection phase
        It will select a subset of the population as the survivors
        to be used in the mutation and crossover processes
        The selection follows the stochastic acceptance algorithm
        Hyperparameter: The selection rate, which determines the exponential rate at which the population 
        will decrease generation after generation.
        """
        new_pop=[]
        max_score=max(self.pop_scores)
        n=len(self.population)
        while len(new_pop)<=floor(n*self.select_rate):
            rnd_idx=sample(range(len(self.population)), 1) # We draw a random member of the population
            is_accepted=np.random.binomial(1,self.pop_scores[rnd_idx]/max_score) # Fix his acceptance probabiility as the ratio of his fitness and the max fitness
            if is_accepted: 
                new_pop.append(self.population[rnd_idx])  # We draw a bernouilli and see if we add him
                self.population.pop(rnd_idx) # We make sure not to draw several times the same member
        self.population=new_pop
        self.pop_scores=[]
        return self 

    def crossover(self):
        """ This fuction executes the crossover phase
        It will create a new population by genetically mixing the past population
        Each new population member will inherit from a fixed number (here 2) of randomly chosen parents
        This mixing allow the convergence to the optimum
        Here we use the uniform crossover methodology
        Hyperparameter: the mixing rate, which determines the proportion of
        changed features from one generation to another
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
        """ This function executes the mutation phase
        It will randomly change a small subset of the population
        The purpose of mutation is preserving and introducing diversity, allowing to avoid local minimum
        We use a gaussian mutation which covariance matrix is a diagonal matrix defined by the vars vector 
        Hyperparameter: the mutation probability of a population member, it should be very low, 10% max
        and the variance ratio that determines the amplitude of a each mutation
        """
        for i in range(len(self.population)):
            is_mutated=np.random.binomial(1, self.mutation_proba) # We see if we mutate the member
            if is_mutated:
                hp_indexes={key:self.hp_grid[key].index(self.population[i][key]) for key in self.hp_grid} # The indexes that describes the member in the grid
                mutated_idx={key:floor(np.random.normal(hp_indexes[key], self.vars[key])) for key in self.hp_grid} # We generate new mutated indices
                mutated_fc_idx={key:min(max(mutated_idx[key],0),len(self.hp_grid[key])) for key in self.hp_grid} # We floor/cap these indices to avoid out of bound problems
                self.population[i]={key:self.hp_grid[key][mutated_fc_idx[key]] for key in self.hp_grid} # We replace the old member by the mutated one
        return self    

    def update_score(self, score):
        """ This method allows for an external scope to modify the iterator during the execution of the loop
        It is used to update the score of the tested valued to do the mutation process at the next step
        """
        self.pop_scores.append(score)
        return self

    def __len__(self):
        return self.n_iter
        