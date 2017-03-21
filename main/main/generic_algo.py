# Trading Strategy mother class

class gen_algo:
    def __init__(self):
        # General defining arguments
        self.name="Generic Algorithm"
        self.model=None # Will stay None for manually coded algo, otherwise will contain the model object for ML algos for example
        self.algo_type=None # Has to be either ML for machine learning or TA for Technical Analysis
        self.output_type=None # Has to be either C for Classification or R for Regression 
        self.predicted_values=None
        self.real_values=None
        self.selected_data=None # List of column names from the main dataset 

        # For Regression
        self.mse=None 
        self.cum_rmse=None
        self.r2oos=None 

        # For Classification
        self.err_class_rate=None
        
    def select_data(self, X, Y):
        return X

    def predict(self, X_test):
        return self

    def train(self, X_train, Y_train):
    # The train function defined here includes the calibration of hyperparameters
        return self
    
    def compute_errors(self):
    # This function will compute MSE, CUM RMSE and R2 OOS or the Classification Rate
        return self

    def __str__(self):
        return self.name