# This algorithm is the standard OLS Linear Regression, including the regularized versions
# We use here the skilearn implementation of OLS

from generic_algo import gen_algo
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

class LR(gen_algo):
    def __init__(self, global_hyperparams,regularization=None, hp_grid=None):
        gen_algo.__init__(self, global_hyperparams, hp_grid) # allow to run the init of the gen_algo class, and define all default arguments
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


        
        