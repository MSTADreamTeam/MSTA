# Trading Strategy mother class

class algorithm_mother:
    def __init__(self, **kwargs):
        self.name="Mother algorithm class instance"
        
    def select_data(self, X, *Y, **kwargs):
        return X

    def predict(self, X, Y, **kwargs):
        return 0

    def fit(self, X, Y, **kwargs):
        return 0
    
    def predict(self, X, Y, **kwargs):
        return 0