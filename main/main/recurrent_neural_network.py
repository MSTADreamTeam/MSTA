# This file defines the ESN using Keras library with Tensorflow backend

from base_algo import BaseAlgo

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import Recurrent, SimpleRNN, LSTM
from keras.initializers import RandomUniform, RandomNormal
from keras.regularizers import l2
from keras import backend as K
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.ops import random_ops, math_ops, array_ops

class ESN(BaseAlgo):
    ''' This class defines the Echo State Network
    This is a specific case of Recurrent Neural Network using a reservoir of sparcely connected neurons
    and recurrent cells to feed an output layer which is the only one fitted
    IMPORTANT NOTE: When you feed the ESN with training data, always make sure to provide data from a single timestamp, not multiple lags
    This is crucial to make sure the algorithm behaves as needed
    '''
    def __init__(self, global_hyperparams, hp_grid = None, n_res=None, sparcity_ratio=None, 
                 distrib=None, spectral_radius=None, activation='tanh',
                 leaking_rate=None, alpha=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.algo_type='ML'
        self.n_res=n_res # Size of the reservoir
        self.sparcity_ratio=sparcity_ratio # Ratio of sparcity of the reservoir
        self.spectral_radius=spectral_radius # The spectral radius of the weight matrix of the reservoir,
        # spectral_radius<1 to have the echo state property, but sometimes the optimal value can be greater than one,
        # hence do not hesitate to try it
        self.activation=activation # The function used in the activation of the network
        self.leaking_rate=leaking_rate # The speed at which the reservoir will update through time
        self.alpha=alpha # The alpha parameter of the Ridge regularization
        
    def init_net(self, input_size, batch_size):
        ''' Initialize the network 
        Reinitializing it erases the memory of the network '''
        self.model=Sequential()
        #self.model.add(Dense(input_size, activation=self.activation, input_shape=()))
        self.model.add(ESNCell(input_size, batch_size, self.n_res, self.spectral_radius, self.sparcity_ratio, 
                               self.leaking_rate, self.activation))
        self.model.add(Dense(1, kernel_regularizer=l2(self.alpha))) # Ridge regularization
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # Change the learning rate given the fact that we have a lin reg?
        return self

    def select_data(self, X):
        # Make sure we don't take any lags, by default we take all the input
        self.selected_data=[True for col in X.columns]
        return self.selected_data

    def predict(self, X_test, pred_index = None):
         # check the shape of X with the batch this tensor
        return super().predict(X_test, pred_index, batch_size=len(X_test))

    def fit(self, X_train, Y_train, verbose=0):
        self.init_net(X_train.shape[1],X_train.shape[0])
        self.model.fit(X_train, Y_train, verbose=verbose, validation_split=0.0, shuffle=False) # Avoid shuffling because of the statefulness of the network
        return self

    def plot(self):
        ''' Used to plot the reservoir of the ESN '''
        print(self.model.layers[0].recurrent_kernel)

class ESNCell(SimpleRNN):
    ''' Subclass of Keras Recurrent Neural Net cell, used to model the reservoir of the ESN
    Code inspired from https://github.com/m-colombo/Tensorflow-EchoStateNetwork/blob/master/esn_cell.py 
    and adapted to Keras syntax '''
    def __init__(self, input_size, batch_size, n_res, norm_scale, sparcity_ratio, leaking_rate, activation,  
                 w_input_init=RandomNormal(),
                 w_res_init=RandomNormal(), # Try to use different distributions, such as uniform 
                 b_init=RandomNormal()): 
        def w_res_initializer(shape):   
            ''' Initialize the random weights and scale them using norm 2, 
            an upper bound for spectral radius,
            to make sure the echo state property is verified
            __future__: find a way to compute the spectral radius to release the constraint
            '''
            w=w_res_init(shape)
            connectivity_mask=K.cast(K.less_equal(random_ops.random_uniform(shape),sparcity_ratio), np.dtype(np.float32))
            w=math_ops.multiply(w,connectivity_mask)
            w_norm2=K.sqrt(K.sum(K.square(w)))
            is_norm_0=K.cast(K.equal(w_norm2, 0), np.dtype(np.float32))
            w=w*norm_scale/(w_norm2+1*is_norm_0)
            return w
        SimpleRNN.__init__(self, n_res, activation=activation, use_bias=True, kernel_initializer=w_res_initializer, 
                           recurrent_initializer=w_res_init, bias_initializer=b_init,
                           dropout=0.0, recurrent_dropout=0.0, stateful=True, 
                           batch_input_shape=(input_size, 1, batch_size), trainable=False)
        self.leaking_rate=leaking_rate
        
    def build(self, input_shape):
        ''' Initialize the weights and connection of the ESN Cell '''
        SimpleRNN.build(self, input_shape)

    def call(self, inputs, state=None):
        ''' Run one step of the ESN Cell '''
        inputs=convert_to_tensor(inputs)
        if not state: state=self.get_initial_states(inputs)[0] 
        input_size = inputs.get_shape().as_list()[1]
        in_mat = array_ops.concat(1.0, [inputs, state]) 
        weights_mat = array_ops.concat(0.0, [self.w_input, self.w_res])
        output = (1-self.leaking_rate)*state+self.leaking_rate*self.activation(math_ops.matmul(in_mat, weights_mat)+self.b)
        return output, output

class LST(BaseAlgo):
    ''' This class defines the Long Short Tern Memory network '''
    def __init__(self, global_hyperparams, hp_grid = None, n_units=None,
                 dropout=0, recurrent_dropout=0):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.algo_type='ML'
        self.n_units=n_units # Size of the LSTM layer
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        
    def init_net(self, input_dim):
        ''' Initialize the neural network '''
        self.model=Sequential()
        self.model.add(LSTM(input_dim=input_dim, output_dim=self.n_units, return_sequences=True,  
                    dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))
        self.model.add(Dense(1)) # output layer 
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy']) # Change the learning rate given the fact that we have a lin reg?
        return self

    def select_data(self, X):
        # Make sure we don't take any lags, by default we take all the input
        self.selected_data=[True for col in X.columns]
        return self.selected_data

    def predict(self, X_test, pred_index = None):
         # check the shape of X with the batch this tensor
        return super().predict(X_test, pred_index)

    def fit(self, X_train, Y_train, verbose=0):
        self.init_net(X_train.shape[1])
        X_train=X_train.values
        Y_train=Y_train.values
        X_train=X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        Y_train=Y_train.reshape(Y_train.shape[0], 1, 1)
        self.model.fit(X_train, Y_train, verbose=verbose)
        return self
