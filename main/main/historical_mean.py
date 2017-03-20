# This algorithm will just take the historical mean as a prediction
# It will be used as a benchmark for prediction as it represents the random walk hypothesis

from generic_algo import gen_algo


class HM(gen_algo):
    def __init__(self, window_size):
        self.name="Historical Mean"
        self.window_size=window_size
            
    def predict(self, X):
        return super().predict(X)


    