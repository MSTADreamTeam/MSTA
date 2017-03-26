# DEPRECATED, NOT USED ANYMORE


from sklearn.base import BaseEstimator

class sklearn_estimator(BaseEstimator):
    """ sklearn BaseEstimator object
    This class allow us to copy an instance of gen_algo into a sklean BaseEstimator object
    This will be usefull to use cross val on user defined algorithms
    """

    def __init__(self, algo, **hyperparams):
        BaseEstimator.__init__(self)
        self.algo=algo
        self.set_params(**hyperparams)
        self.hyperparams_names=hyperparams.keys()
        self._estimator_type="classifier" if self.algo.global_hyperparams["output_type"]=='C' else 'regressor' # This attribute is checked by sklearn cross val
        self.classes_=self.algo.global_hyperparams["classes"] if self.algo.global_hyperparams["output_type"]=='C' else None # List of classes label, used by sklearn cross val        
        

    def get_params(self, deep=True):
        """ Init parameters getter
        This has to include any parameters used to copy the algo
        Hence, it will copy the input of the __init__ 
        The deep argument is here for the function to be syntaxproof with sklearn
        This function could be recoded so that it automatically creates the missing parameters, but we will first avoid it
        """
        output={param_key:getattr(self, param_key) for param_key in self.hyperparams_names}
        output['algo']=self.algo
        return output  

    def set_params(self, **parameters):
        """ Hyperparameters setter
        Please DO NOT modify any hyperparameters of the model directly, always use this function
        Directly derived from sklearn BaseEstimator, this implementation is not necessary since it is already inherited,
        for debugging purposes we will let it here for now though
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        if self.algo.model is not None: # Important not to forget to inherit the values of hyperparams to the model object
            self.algo.model.set_params(**parameters)
        return self


    def predict(self, X_test, pred_index=None):
        """ We just call the algo predict function """
        predicted_value=self.algo.predict(X_test, pred_index)
        return predicted_value
 
    def fit(self, X_train, Y_train):
        """ We just call the algo fit function """
        self.algo.fit(X_train, Y_train)
        return self


        