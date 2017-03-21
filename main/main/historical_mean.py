# This algorithm will just take the historical mean as a prediction
# It will be used as a benchmark for prediction as it represents the random walk hypothesis

from generic_algo import gen_algo
import pandas as pd
import numpy as np

class HM(gen_algo):
    def __init__(self, window_size=None,mean_type="arithmetic"):
        self.name="Historical Mean"
        self.window_size=window_size
        self.mean_type=mean_type
            
    def predict(self, Y, store_value=False): # Y has to be a non lagged dataframe with only one column
        w=self.window_size
        label=Y.columns[0]+" HM W"+str(w)+" "+self.mean_type
        df=pd.DataFrame(index=Y.index)
        if self.mean_type=="arithmetic":
            df[label]=[Y.iloc[i-w-1:i-1].mean(axis=0,skipna=None) for i in range(0,len(Y.index))]
        elif self.mean_type=="geometric":
            res=[np.nan for k in range(w)]
            for i in range(w,len(Y.index)):
                geo_mean=1
                for j in range(i-w-1,i):
                    geo_mean=geo_mean*(1+Y.iloc[i])
                geo_mean=np.power(geo_mean,1/w)-1
                res.append(geo_mean)
            df[label]=res
        if store_values:
            self.predicted_values=df
        return df # here we have a redundency in the return and the side effect of the method, this is used to simplify coding

# testing code
#import matplotlib.pyplot as pyplt
#import data

#dataset=data.dataset_building(n_max=1000)
#data_test=dataset[["EURUSD Curncy"]]
#hm=HM(100)
#hm2=HM(100,mean_type="geometric")
#res=hm.predict(data_test)
#res2=hm2.predict(data_test)
#pyplt.plot(data_test)
#pyplt.plot(res)
#pyplt.plot(res2)
#pyplt.show()
# end of testing code
    