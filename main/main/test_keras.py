# tests

import keras as kr
from keras.models import Sequential
from keras.layers.core import Dense
from keras.initializers import RandomNormal
import data

dataset=data.dataset_building('local', asset_ids, start_date, end_date, n_max=2000) # please recode the dataset_building functio to make it support local and quandl data
dataset = data.add_returns(dataset, [0]) # creates some NANs as a result of the returns computation
dataset.dropna(inplace=True)
# We select an asset returns time series to predict from the dataset
Y=dataset[dataset.columns[1]] # need to find a reliable way to find the index of the column 

# X: include all the lags of Y and additional data
lags=range(1,rolling_window_size+1)
X=data.lagged(dataset,lags=lags) # In X please always include all the lags of Y that you want to use for the HM as first colunms


model=Sequential()
model.add(Dense(1, input_shape=(100,)))
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x,y)


