# Trading Strategy mother class

# QUESTION: Should we define a subclass for ML algos, avoiding the repetitive fitting/predicting code?
# We may need the equivalent for TA algos

import pandas as pd

class gen_algo:
    def __init__(self, global_hyperparams):
        ## General defining arguments
        self.name="Generic Algorithm"
        self.model=None # Will stay None for manually coded algo, otherwise will contain the model object for ML algos for example
        self.algo_type=None # Has to be either ML for machine learning or TA for Technical Analysis, and BA for basic algorithms
        self.output_type=None # Has to be either C for Classification or R for Regression 
        self.predicted_values={} # This dict of Date/Value will be converted in a dataframe later for optimization purposes
        self.real_values={}
        self.selected_data=[] # List of column names from the main dataset 
        self.global_hyperparams=global_hyperparams # Dictionary of global hyperparameters

        ## Outputs
        # For Regression
        self.mse=None 
        self.cum_rmse=None
        self.r2oos=None 

        # For Classification
        self.err_class_rate=None
        
    def select_data(self, X):
        return X

    def predict(self, X_test, pred_index):
        return self

    def train(self, X_train, Y_train):
    # The train function defined here includes the calibration of hyperparameters
        return self
    
    def compute_outputs(self, Y):
    # This function will compute all the desired outputs from the predicted data and the real data
        return self

    def get_predicted_values(self):
        return pd.DataFrame.from_dict(self.predicted_values, orient='index')

    def get_real_values(self):
        return pd.DataFrame.from_dict(self.real_values, orient='index')

    def __str__(self):
        return self.name