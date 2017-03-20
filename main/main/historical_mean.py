# This algorithm will just take the historical mean as a prediction
# It will be used as a benchmark for prediction as it represents the random walk hypothesis

from generic_algo import gen_algo
import pandas as pd

class HM(gen_algo):
    def __init__(self, window_size=None):
        self.name="Historical Mean"
        self.window_size=window_size
            
    def predict(self, Y): # Y has to be a non lagged dataframe
        w=self.window_size
        df=pd.DataFrame(index=Y.index)
        df["HM W"+str(w)]=[Y.iloc[i-w-1:i-1].mean(axis=0,skipna=None) for i in range(0,len(Y.index))]
        return df

# testing code
import matplotlib.pyplot as pyplt

data_test=dataset["EURUSD Curncy"]
hm=HM(100)
res=hm.predict(data_test)

pyplt.plot(data_test)
pyplt.plot(res)
pyplt.show()
# end of testing code
    