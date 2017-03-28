# This file defines the Generic Algorithm iterator used in cross validation

class GeneticAlgorithm:
    """ Genetic Algorithm iterator """
    def __init__(self, hp_grid, n_iter):
        self.current_val = None
        self.n_iter = 0
        self.n_max = n_iter
        self.hp_grid=hp_grid

    def __iter__(self):
        return self

    def __next__(self):
        if self.n_iter > self.n_max:
            raise StopIteration
        else:
            self.n_iter += 1
            # Need to code here
            return None # Has to return the value of the iterator