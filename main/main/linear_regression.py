# This algorithm is the standard OLS Linear Regression
# We use here the skilearn implementation of OLS

from generic_algo import gen_algo
from sklearn.linear_model import LinearRegression

class LR(gen_algo):
    def __init__(self, global_hyperparams):
        gen_algo.__init__(self, global_hyperparams) # allow to run the init of the gen_algo class, and define all default arguments
        self.name="Linear Regression"
        self.model=LinearRegression()
        self.algo_type="BA" # By convention

    def train(self, X_train, Y_train): # Here no calib of hyperparams, just a usual OLS fitting
        self.model.fit(X_train,Y_train)
        return self

    def predict(self, X_test, pred_index):
        predicted_value=self.model.predict(X_test.reshape(1, -1))
        self.predicted_values[pred_index]=predicted_value[0][0] # Syntax check
        return predicted_value
        
        