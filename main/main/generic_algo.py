# Trading Strategy mother class

import pandas as pd

class gen_algo:
    def __init__(self):
        ## General defining arguments
        self.name="Generic Algorithm"
        self.model=None # Will stay None for manually coded algo, otherwise will contain the model object for ML algos for example
        self.algo_type=None # Has to be either ML for machine learning or TA for Technical Analysis
        self.output_type=None # Has to be either C for Classification or R for Regression 
        self.predicted_values=pd.DataFrame(columns=["Predicted Value"])
        self.real_values=pd.DataFrame(columns=["Real Value"])
        self.selected_data=[] # List of column names from the main dataset 
        self.global_hyperparams={} # Dictionary of global hyperparameters

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
    
    def compute_outputs(self, pred_index):
    # This function will compute all the desired outputs from the predicted data and the real data
        return self

    def __str__(self):
        return self.name