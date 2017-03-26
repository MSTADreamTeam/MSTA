import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator # Allows to define our algo as an sklearn estimator, allowing the GridSeatch function to recognize the predict, fit, and others methods
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from data import to_class
from cross_validation import GridSearch, RandomSearch, GeneticAlgorithm

class gen_algo():
    """ Predictive Algorithm mother class
    Let us notice that this class inherits from the BaseEstimator class of sklearn, 
    this architecture allows us to easily use sklearn cross validation functions directly
   
    QUESTION: Should we define a subclass for ML algos, avoiding the repetitive fitting/predicting code? We may need the equivalent for TA algos
    """
        
    def __init__(self, global_hyperparams, hp_grid=None, **hyperparams):
        ## General defining arguments
        self.name="Generic Algorithm"
        self.model=None # This attribute will recieve the sklearn estimator object for ML algos
        self.algo_type=None # Has to be either ML for machine learning or TA for Technical Analysis, and BA for basic algorithms
        self.predicted_values={} # This dict of Date/Value will be converted in a dataframe later for optimization purposes
        self.real_values={}
        self.selected_data=[] # List of column names from the main dataset 
        self.global_hyperparams=global_hyperparams # Dictionary of global hyperparameters
        if hp_grid is not None:
            self.hp_grid=hp_grid # The hyperparameters grid used in GridSearch or RandomSearch
           # self.set_params(**{param_key:param_values[0] for param_key, param_values in hp_grid.items()}) # Initialize the value of the hyperparams attributes with the first value in the hp_grid
        else:
            self.hp_grid={}
        self.best_hp={} # The best hyperparameters outputed by the cross validation
        self.set_hyperparams(**hyperparams) # Set the fixed hyperparams of the model
        #self.hyperparams_names=np.unique(list(hyperparams.keys())+list(hp_grid.keys())) # List all the names of hyperparams of the model
        #self.sklearn_estimator=sklearn_estimator(self, **hyperparams) # This attribute will be a BaseEstimator instance used in cross validation, please do not confuse with model: model is used in predict and fit only for ML algos, and sklearn_estimator is used in calib, and is always defined
        
    
        ## Outputs
        # For Regression
        self.se=None
        self.mse=None 
        self.cum_rmse=None
        self.r2is=None
        self.r2oos=None 

        # For Classification
        self.accuracy=None
        self.wrong_way_metric=None


    def set_hyperparams(self, **parameters):
        """ Hyperparameters setter used in cross validation
        Please DO NOT modify any hyperparameters of the model directly, always use this function
        Directly derived from sklearn BaseEstimator, this implementation is not necessary since it is already inherited,
        for debugging purposes we will let it here for now though
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        if self.algo.model is not None: # Important not to forget to inherit the values of hyperparams to the model object
            self.algo.model.set_params(**parameters)
        return self



    def select_data(self, X):
        """ Selecting data function
        This function will output a list of column labels from X to be used in fit, calib and predict
        It allows us to make all the algo work on the same complete dataset and just take slices of it for each algo
        """
        return X

    def predict(self, X_test, pred_index=None):
        """ Predict function used in main and in the cross validation process
        We need a syntax compatible with sklean, as a result please be careful when modifying this code
        It can accept as an X_test input either an array or a dataframe and gives a corresponding output
        This version of the function only works for ML algorithm and it has to be recoded for TA algorithms
        If a pred_index is provided, the prediction will be stored in predicted_values with this index
        """
        if self.algo_type=="ML":
            predicted_value=self.model.predict(X_test.values.reshape(1, -1))
            if self.global_hyperparams["output_type"]=="C" and self.model._estimator_type!='classifier':
                predicted_value=to_class(predicted_value, self.global_hyperparams["threshold"])    
        else:
            predicted_value=np.nan # Not integrated yet
        if pred_index is not None:
            self.predicted_values[pred_index]=predicted_value[0] # Syntax check, need to work for array and single value prediction
        return predicted_value
 
    
    def fit(self, X_train, Y_train):
        """ This method is used in the calib, it does a basic fitting, only works for ML algos """
        if self.algo_type=="ML":
            self.model.fit(X_train,Y_train)
        else: # Case where we have a TA or BA algo
            None # Not integrated yet
        return self

    def calib(self, X_train, Y_train, pred_index=None, cross_val_type=None, hyperparams_grid=None, n_splits=10, calib_type=None, scoring_type=None, n_iter=10):
        """ The calib function defined here includes the calibration of hyperparameters and the fitting """
        if cross_val_type is not None: # We do the calibration by cross val
            hp_grid=self.hp_grid if hyperparams_grid is None else hyperparams_grid
            if cross_val_type=="k_folds":
                cv=KFold(n_splits,shuffle=False)        
            elif cross_val_type=="ts_cv":
                cv=TimeSeriesSplit(n_splits)
            if self.algo_type=="ML": 
                if scoring_type is None:
                    scoring='neg_mean_squared_error' if self.global_hyperparams["output_type"]=='R' else 'accuracy' # Let us note here that a lot of different scoring methodologies are available
                else:
                    scoring=scoring_type
                if calib_type=="GridSearch":
                    optimiser=GridSearch(self,hp_grid=hp_grid,cv=cv,scoring=scoring).compute_cv(X_train,Y_train)
                elif calib_type=="RandomSearch":
                    optimiser=RandomSearch(self,hp_grid=hp_grid,cv=cv,scoring=scoring,n_iter=n_iter).compute_cv(X_train,Y_train).compute_cv(X_train,Y_train)
                elif calib_type=="GeneticAlgorithm":
                    optimiser=GeneticAlgorithm(self,hp_grid=hp_grid,cv=cv,scoring=scoring).compute_cv(X_train,Y_train).compute_cv(X_train,Y_train)
                self.set_params(**optimiser.best_hp)
                self.best_hp[pred_index]=optimiser.best_hp
            else: # Case where we have a TA or BA algo
                None # Not integrated yet
        else: # Here we do not calibrate the hyperparameters
            self.fit(X_train,Y_train)
        return self

    def compute_outputs(self, Y, output_to_compute=[]):
        """ This function will compute all the desired outputs from the predicted data and the real data
        Check optimization here, the use of dataframe might not be the best 
        Since the class is inherited from BaseEstimator, it might not be necessary to recode the output manually,
        we might want to check if sklearn has functions doing this job manually
        The not yet used argument output_to_compute will allow us to select to compute only a subset of all outputs to optimize speed
        """    
        if self.global_hyperparams["output_type"]=='R':
            self.se=((self.get_output('predicted_values').items-Y)**2)
            self.mse=self.se.mean(axis=0)
            # The other outputs are not coded yet
        else:
            self.good_pred=self.get_output('predicted_values')==Y
            self.accuracy=(good_pred).sum(axis=0)
        return self

    def reset_outputs(self):
        """ Used to reset the values of the output during the cross val, please keep updated with new outputs """        
        # For Regression
        self.se=None
        self.mse=None 
        self.cum_rmse=None
        self.r2is=None
        self.r2oos=None 

        # For Classification
        self.accuracy=None
        self.wrong_way_metric=None

    def get_output(self, key):
        return pd.DataFrame.from_dict(getattr(self,key), orient='index')

    def __str__(self):
        return self.name