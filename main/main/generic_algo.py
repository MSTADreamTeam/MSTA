# Trading Strategy mother class

# QUESTION: Should we define a subclass for ML algos, avoiding the repetitive fitting/predicting code?
# We may need the equivalent for TA algos

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV



class gen_algo:
    def __init__(self, global_hyperparams, hp_grid=None):
        ## General defining arguments
        self.name="Generic Algorithm"
        self.model=None # Will stay None for manually coded algo, otherwise will contain the model object for ML algos for example
        self.algo_type=None # Has to be either ML for machine learning or TA for Technical Analysis, and BA for basic algorithms
        self.predicted_values={} # This dict of Date/Value will be converted in a dataframe later for optimization purposes
        self.real_values={}
        self.selected_data=[] # List of column names from the main dataset 
        self.global_hyperparams=global_hyperparams # Dictionary of global hyperparameters
        if hp_grid is not None:
            self.hp_grid=hp_grid # The hyperparameters grid used in GridSearch or RandomSearch
        else:
            self.hp_grid={}
        self.best_hp={} # The best hyperparameters outputed by the cross validation

        ## Outputs
        # For Regression
        self.mse=None 
        self.cum_rmse=None
        self.r2is=None
        self.r2oos=None 

        # For Classification
        self.accuracy=None
        
    def select_data(self, X):
        return X

    def predict(self, X_test, pred_index):
        if self.algo_type=="ML":
            predicted_value=self.model.predict(X_test.reshape(1, -1))
        else:
            predicted_value=np.nan # Not integrated yet
        if self.global_hyperparams["output_type"]=="C":
            predicted_value=to_class(predicted_value, self.global_hyperparams["threshold"])
        self.predicted_values[pred_index]=predicted_value[0][0] # Syntax check
        return predicted_value
 

    def train(self, X_train, Y_train, pred_index=None, cross_val_type=None, hyperparams_grid=None, n_split=10, calib_type=None,scoring_type=None):
    # The train function defined here includes the calibration of hyperparameters
        if cross_val_type is not None: # We do the calibration by cross val
            hp_grid=self.hp_grid if hyperparams_grid is None else hyperparams_grid
            if cross_val_type=="k_folds":
                cv=KFold(n_splits,shuffle=False)        
            elif cross_val_type=="ts_cv":
                cv=TimeSeriesSplit(n_splits)
            if self.algo_type=="ML": 
                if scoring_type is None:
                    scoring='neg_mean_squared_error' if self.output_type=='R' else 'accuracy' # Let us note here that a lot of different scoring methodologies are available
                else:
                    scoring=scoring_type
                if calib_type=="GridSearch":
                    optimiser=GridSearchCV(self.model,param_grid=hp_grid,cv=cv,scoring=scoring).fit(X_train,Y_train)
                elif calib_type=="RandomSearch":
                    optimiser=RandomizedSearchCV(self.model,param_distributions=hp_grid,cv=cv,scoring=scoring).fit(X_train,Y_train)
                elif calib_type=="GeneticAlgorithm":
                    # This option is not coded yet
                    optimizer=None
                self.model=optimiser.best_estimator_
                self.best_hp[pred_index]=optimiser.best_params_
            else: # Case where we have a TA or BA algo
                None # Not integrated yet
        else: # Here we do not calibrate the hyperparameters
            if self.algo_type=="ML":
                self.model.fit(X_train,Y_train)
            else:
                None # Not integrated yet
        return self

    def compute_outputs(self, Y):
    # This function will compute all the desired outputs from the predicted data and the real data
        return self

    def get_predicted_values(self):
        return pd.DataFrame.from_dict(self.predicted_values, orient='index')

    def get_real_values(self):
        return pd.DataFrame.from_dict(self.real_values, orient='index')

    def get_best_hp(self):
        return pd.DataFrame.from_dict(self.best_hp, orient='index')

    def __str__(self):
        return self.name