from generic_algo import gen_algo
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

class LR(gen_algo):
    """  This algorithm is the standard OLS Linear Regression, including the regularized versions Lasso, Ridge and Elastic
    We use here the skilearn implementation of OLS
        
    Let us notice that the LR class can output regression and classification values as of now. However, the cross val only supports a regression scoring (MSE)
    since the transformation from regression to classification is only done when outputting the results
    For this reason it is advised not to use LR as a classification algorithm

    For these algos, the CV could be done more efficiently directly using built in tools, because of the specific impact of the regularisation parameter
    Hence, we will overload the train method for this algo later, for speed optimization purposes
    """

    def __init__(self, global_hyperparams, hp_grid={}, regularization=None, **hyperparams):
        gen_algo.__init__(self, global_hyperparams, hp_grid, **hyperparams) # allow to run the init of the gen_algo class, and define all default arguments
        if regularization is None:
            self.model=LinearRegression()
            self.name="Linear Regression"
        elif regularization=="Lasso":
            self.model=Lasso(normalize=True)
            self.name="Lasso"
        elif regularization=="Ridge":
            self.model=Ridge(normalize=True)
            self.name="Ridge"
        elif regularization=="ElasticNet":
            self.model=ElasticNet(normalize=True)
            self.name="Elastic Net"
        self.algo_type="ML"

        
        