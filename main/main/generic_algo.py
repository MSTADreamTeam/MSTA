# Trading Strategy mother class

class gen_algo:
    def __init__(self):
        self.name="Mother algorithm class instance"
        self.model=None
        
    def select_data(self, X, *Y):
        return X

    def predict(self, X):
        return self

    def train(self, X, Y):
        return self
    
    def predict(self, X):
        return self