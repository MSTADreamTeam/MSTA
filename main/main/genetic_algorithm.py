# This file defines the Generic Algorithm iterator used in cross validation
import numpy as np
from sklearn.model_selection import ParameterSampler
from random import sample, randint
from copy import deepcopy

class GeneticAlgorithm:
    ''' Genetic Algorithm iterable/iterator
    From a ramdomly generated initial population, the algorithm will improve it generation after generation
    The process of improvment is based on the selection, crossover and mutation phases applied at each new generation
    !!! please read for additional ideas: !!!
    James S. Bergstra, R´emi Bardenet, Yoshua Bengio, and Bal´azs K´egl. Algorithms for hyper-parameter
    optimization. In J. Shawe-Taylor, R.S. Zemel, P. Bartlett, F.C.N. Pereira, and K.Q. Weinberger, editors,
    Advances in Neural Information Processing Systems 23 (NIPS 2010), pages 2546–2554. 2011.
    '''
    def __init__(self, hp_grid, n_iter_max=10, init_pop_size=5, select_rate=0.5, mixing_ratio=0.5, mutation_proba=0.1, std_ratio=0.1):
        ''' In the __init__ are defined the fixed parameters '''
        self.n_iter_max=n_iter_max
        self.hp_grid=hp_grid
        self.init_pop_size=init_pop_size
        self.mutation_proba=mutation_proba
        self.select_rate=select_rate
        self.mixing_ratio=mixing_ratio
        self.stds={key:std_ratio*len(self.hp_grid[key]) for key in self.hp_grid}  # Used in the mutation phase      

    def iter(self):
        ''' The __iter__ method that returns an iterator 
        Since it is called at each new call of the iterable in a 'for' statement,
        it initializes all dynamic elements '''
        self.n_iter=0
        self.pop_scores=[]
        self.population=list(ParameterSampler(self.hp_grid, self.init_pop_size)) # First population is random, we turn it into list to copy it
        self.current_pop=self.population.copy()
        self.generation=0
        return self

    def __next__(self):
        ''' The __next__ method that returns a new element at each step of the iteration '''
        if self.n_iter>self.n_iter_max: 
            raise StopIteration
        if not self.current_pop:
            self.selection()
            self.crossover()
            self.mutation()
            self.generation += 1
            self.current_pop=self.population.copy() # The copy makes sure we are copying the list and not the ref
        self.n_iter += 1
        return self.current_pop.pop()
        
    def selection(self):
        ''' This function executes the selection phase
        It will select a subset of the population as the survivors
        to be used in the mutation and crossover phases
        The selection follows the stochastic acceptance algorithm
        Hyperparameter: The selection rate, which determines the exponential rate at which the population 
        will decrease generation after generation.
        '''
        n=len(self.population)
        new_pop=[]
        max_score=max(self.pop_scores)
        while len(new_pop)<=max(1,int(n*self.select_rate)): # Notice that the selection phase always select at least one element
            rnd_idx=randint(0,len(self.population)-1) # We draw a random member of the population
            is_accepted=np.random.binomial(1,self.pop_scores[rnd_idx]/max_score) # Fix his acceptance probability as the ratio of his fitness and the max fitness
            if is_accepted:   # We draw a bernouilli and see if we add him
                new_pop.append(self.population[rnd_idx])
                self.population.pop(rnd_idx) # We make sure not to draw several times the same member
        self.population=new_pop
        self.pop_scores=[]
        return self 

    def crossover(self):
        ''' This fuction executes the crossover phase
        It will create a new population by genetically mixing the past population
        Each new population member will inherit from a fixed number (here 2) of randomly chosen parents
        This mixing allow the convergence to the optimum
        Here we use the uniform crossover methodology
        Hyperparameter: the mixing rate, which determines the proportion of
        changed features from one generation to another
        '''
        new_pop=[]
        while len(new_pop)<=len(self.population):
            parents=sample(self.population, 2) # We select two random parents, note that a parent can be selected in several couples
            crossover_points=sample(self.hp_grid.keys(), int(self.mixing_ratio*len(self.hp_grid))) # this defines the keys of the hyperparameters that will be mixed
            temp={key:parents[0][key] for key in crossover_points}
            parents[0].update({key:parents[1][key] for key in crossover_points})
            new_pop.append(parents[0])
            parents[1].update(temp)
            new_pop.append(parents[1])
        self.population=new_pop
        return self

    def mutation(self):
        ''' This function executes the mutation phase
        It will randomly change a small subset of the population
        The purpose of mutation is preserving and introducing diversity, allowing to avoid local minimum
        We use a gaussian mutation which covariance matrix is a diagonal matrix defined by the stds vector 
        Hyperparameter: the mutation probability of a population member, it should be very low, 10% max
        and the variance ratio that determines the amplitude of each mutation
        '''
        for i in range(len(self.population)):
            is_mutated=np.random.binomial(1, self.mutation_proba) # We see if we mutate the member
            if is_mutated:
                hp_indexes={key:list(self.hp_grid[key]).index(self.population[i][key]) for key in self.hp_grid} # The indexes that describes the member in the hp grid
                mutated_idx={key:int(np.random.normal(hp_indexes[key], self.stds[key])) for key in self.hp_grid} # We generate new mutated indices
                mutated_fc_idx={key:min(max(mutated_idx[key],0),len(self.hp_grid[key])-1) for key in self.hp_grid} # We floor/cap these indices to avoid out of bounds problems
                self.population[i]={key:self.hp_grid[key][mutated_fc_idx[key]] for key in self.hp_grid} # We replace the old member by the mutated one
        return self    

    def update_score(self, score):
        ''' This method allows for an external scope to modify the iterator during the execution of the loop
        It is used to update the score of the tested valued to do the selection phase for the next generation
        '''
        self.pop_scores.append(score)
        return self

    def __len__(self):
        return self.n_max
        