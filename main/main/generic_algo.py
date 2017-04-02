import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
from data import to_class
from cross_validation import CrossVal

class gen_algo:
    ''' Predictive Algorithm mother class 
    Here is implemented most of the common code performed by algos, including the fit, predict and calib functions
    For TA algos the predict function is often overloaded
    '''
        
    def __init__(self, global_hyperparams, hp_grid=None, **hyperparams):
        ## General defining arguments
        self.name='Generic Algorithm'
        self.model=None # This attribute will recieve the sklearn estimator object for ML algos
        self.algo_type=None # Has to be either ML for machine learning or TA for Technical Analysis, and BA for basic algorithms
        self.predicted_values=[] # Will store predicted values as a list
        self.pred_index=[] # Used to build the outputs dictionaries
        self.real_values=[] # Not used yet, need to delete?
        self.selected_data=[] # List of column names from the main dataset 
        self.global_hyperparams=global_hyperparams # Dictionary of global hyperparameters
        self.hp_grid=hp_grid # The hyperparameters grid used in the CV
        # self.set_hyperparams(**{param_key:param_values[0] for param_key, param_values in hp_grid.items()}) # Initialize the value of the hyperparams attributes with the first value in the hp_grid
        self.best_hp=[] # The best hyperparameters outputed by the cross validation
        self.set_hyperparams(**hyperparams) # Set the fixed hyperparams of the model
    
        ## Outputs
        # For Regression
        self.se=None
        self.mse=None 
        self.nmse=None
        self.cum_rmse=None
        self.r2is=None
        self.r2oos=None 

        # For Classification
        self.good_pred=None
        self.accuracy=None
        self.wrong_way_metric=None


    def set_hyperparams(self, **parameters):
        ''' Hyperparameters setter used in cross validation
        Please DO NOT modify any hyperparameters of the model directly, 
        always use this function to make sure it also impacts the model attribute when needed.
        This function should not create new hyperparameters
        '''
        for parameter, value in parameters.items(): 
            if hasattr(self, parameter): setattr(self, parameter, value) 
        if self.model is not None: # Important not to forget to inherit the values of hyperparams to the model object
            self.model.set_params(**parameters)
        return self



    def select_data(self, X):
        ''' Selecting data function
        This function will output a list of column labels from X to be used in fit, calib and predict
        It allows us to make all the algo work on the same complete dataset and just take slices of it for each algo
        '''
        return X

    def predict(self, X_test, pred_index=None):
        ''' Predict function used in main and in the cross validation process
        It can accept as an X_test input either an array or a dataframe and gives a corresponding output
        This version of the function only works for ML algorithm and it has to be recoded for TA algorithms
        If a pred_index is provided, the prediction will be stored in predicted_values with this index
        '''
        if self.model is not None:
            predicted_values=self.model.predict(X_test)
            if self.global_hyperparams['output_type']=='C' and self.model._estimator_type!='classifier': # If we use a regression model and we still need to output a class
                predicted_values=to_class(predicted_values, self.global_hyperparams['threshold'])    
        else:
            predicted_values=np.nan # Not integrated yet
        if pred_index is not None:
            self._store_predicted_values(pred_index, predicted_values)
        return predicted_values
 
    def _store_predicted_values(self, pred_index, pred_values):
        ''' Used to store the predicted values properly '''
        pred_values=np.atleast_1d(pred_values)
        self.pred_index+=list(pred_index)
        self.predicted_values+=list(pred_values)
        return self
    
    def fit(self, X_train, Y_train):
        ''' This method is used in the calib, it does a basic fitting,
        only works for ML algos, it does not do anything for TA or BA algos since usually they do not need any fit '''
        if self.model is not None:
            self.model.fit(X_train,Y_train)
        return self

    def calib(self, X_train, Y_train, pred_index=None, cross_val_type=None, hyperparams_grid=None, n_splits=10, calib_type=None, scoring_type=None, n_iter=10, **ga_args):
        ''' The calib function defined here includes the calibration of hyperparameters and the fitting '''
        hp_grid=self.hp_grid if hyperparams_grid is None else hyperparams_grid
        if cross_val_type is not None and hp_grid is not None: # We do the calibration by cross val
            if cross_val_type=='k_folds':
                cv=KFold(n_splits,shuffle=False)        
            elif cross_val_type=='ts_cv':
                cv=TimeSeriesSplit(n_splits)
            if scoring_type is None:
                scoring='nmse' if self.global_hyperparams['output_type']=='R' else 'accuracy' # Let us note here that a lot of different scoring methodologies are available
            else:
                scoring=scoring_type
            optimiser=CrossVal(self, hp_grid, calib_type, cv, scoring, n_iter, **ga_args)
            optimiser.compute_cv(X_train,Y_train)
            self.set_hyperparams(**optimiser.best_hp) # This could be done inside the CV, as of now self will have as active hp the last tested
            self.best_hp.append(optimiser.best_hp)

        self.fit(X_train,Y_train) # We always fit the model after the calib
        return self

    def calib_predict(self, X_train, Y_train, X_test, pred_index, **algos_cv_params):
        ''' Used as the multithread targer function '''
        self.calib(X_train, Y_train, pred_index, **algos_cv_params)
        self.predict(X_test, pred_index)
        # For debug            
        print('{} prediction: {}'.format(self.name, pred_index[0]))
        return self
        
    def compute_outputs(self, Y, pred_val=None,*output_to_compute):
        ''' This function will compute all the desired outputs from the predicted data and the real data
        It relies on the internal methods _compute, please keep the methods and the dictionaries updated
        '''
        # Let us notice that defining a dictionary of functions is very non python way of coding, we might want to think of a best way
   
        pred_val=pred_val if pred_val is not None else np.array(self.predicted_values)            
        Y= Y.values if len(Y)==len(pred_val) else Y.loc[self.pred_index].values # Turn Y in an np.array
        output_r={'se':self._compute_se,
                  'mse':self._compute_mse,
                  'nmse':self._compute_nmse}
        output_c={'good_pred':self._compute_good_pred,
                  'accuracy':self._compute_accuracy}

        output_dict=output_r if self.global_hyperparams['output_type']=='R' else output_c

        if output_to_compute:
            output_keys=output_to_compute
        else:
            output_keys=output_dict.keys()
        return {key:output_dict[key](Y, pred_val) for key in output_keys} # The return line will compute the outputs and give as an output the dictionary of values
        
    def _compute_se(self, Y, pred_val):
        self.se=(pred_val-Y.values)**2
        return self.se

    def _compute_mse(self, Y, pred_val):
        if self.se is None:
            self._compute_se(Y, pred_val)
        self.mse=np.mean(self.se)
        return self.mse

    def _compute_nmse(self,Y, pred_val):
        if self.mse is None:
            self._compute_mse(Y, pred_val)
        self.nmse=-self.mse
        return self.nmse

    def _compute_good_pred(self, Y, pred_val):
        self.good_pred=pred_val==Y
        return self.good_pred
    
    def _compute_accuracy(self, Y, pred_val):
        if self.good_pred is None:
            self._compute_good_pred(Y, pred_val)
        self.accuracy=np.mean(self.good_pred)
        return self.accuracy
       

    def reset_outputs(self):
        ''' Used to reset the values of the output during the cross val, please keep updated with new outputs '''        
        # For Regression
        self.se=None
        self.mse=None 
        self.cum_rmse=None
        self.r2is=None
        self.r2oos=None 

        # For Classification
        self.good_pred=None
        self.accuracy=None
        self.wrong_way_metric=None

    def get_output(self, key):
        ''' Turn an np.array output into a proper DataFrame
        External use only
        This can be applied on all np.array stocking values at each prediction such as: best_hp, mse, good_pred, ...
        '''
        return pd.DataFrame(getattr(self,key),index=self.pred_index, columns=[key])

    def __str__(self):
        return self.name


class AlgoError(Exception):
    ''' Error raised in case of an error in an algo
    Especifically when a classifier is called to regress '''
    def __init__(self, msg):
        self.msg=msg

    def __str__(self):
        return self.msg