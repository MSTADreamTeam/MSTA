
from generic_algo import gen_algo

class CMW(gen_algo):
    """ Core Manual Weighting: This algo used as a core algo just manually fixes the weights of the prediction """

    def __init__(self, global_hyperparameters, weights):
        gen_algo.__init__(self, global_hyperparams)
        self.algo_type="BA"
        self.weights=weights

    def predict(self, X_test, pred_index=None):
        predicted_values=X_test*self.weights        
        if pred_index is not None:
            self._store_predicted_values(pred_index, predicted_values)
        return predicted_values        

class BIS(gen_algo):
    """ Best In Sample: This algo used as a core algo just pick the best selection IS to predict OOS """
    def __init__(self, global_hyperparameters, scoring):
        gen_algo.__init__(self, global_hyperparams)
        self.algo_type="BA"
        self.scoring=scoring

    

