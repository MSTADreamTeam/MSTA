# This file defines the ESN

from base_algo import BaseAlgo

import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import init_ops, math_ops, random_ops, array_ops
from tensorflow.python.ops import variable_scope as vs


class ESN(BaseAlgo):
    ''' This class defines the Echo State Network
    This is a specific case of Recurrent Neural Network using a reservoir of sparcely connected neurons
    and recurrent cells to feed an output layer which is the only one fitted '''
    def __init__(self, global_hyperparams, hp_grid = None, n_res=None, sparcity_ratio=None, 
                 distrib=None, spectral_radius=None, input_scaling=None, activation='tanh',
                 leaking_rate=None, regularization=None, alpha=None, l1_ratio=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.n_res=n_res # Size of the reservoir
        self.sparcity_ratio=sparcity_ratio # Ratio of sparcity of the reservoir
        self.distrib=distrib # Distribution of the non zero elements of the weights of the reservoir
        self.spectral_radius=spectral_radius # The spectral radius of the weight matrix of the reservoir, spectral_radius<1 to have the echo state property
        self.input_scaling=input_scaling # The method/parans of the input layer weights scaling
        self.activation=getattr(tf.nn,activation) # The function used in the activation of the network
        self.leaking_rate=leaking_rate # The speed at which the reservoir will update through time
        self.regularization=regularization # The regularization used in the loss function
        self.alpha=alpha # The alpha parameter of the regularization
        self.l1_ratio=l1_ratio # the l1 ratio parameter of an ElasticNet regularization
        self.rnn_cell=ESNCell(n_res, spectral_radius, sparcity_ratio, leaking_rate, self.activation)

    



class ESNCell(rnn_cell.RNNCell):
    ''' Subclass of tensorflow Recurrent Neural Net cell, used to model the reservoir of the ESN
    Code inspired from https://github.com/m-colombo/Tensorflow-EchoStateNetwork/blob/master/esn_cell.py '''
    def __init__(self, n_res, n2_scale, sparcity_ratio, leaking_rate, activation, 
                 w_input_init=init_ops.random_normal_initializer(),
                 w_res_init=init_ops.random_normal_initializer(), 
                 b_init=init_ops.random_normal_initializer()):
        self.n_res=n_res
        self.leaking_rate=leaking_rate
        self.activation=activation
        self.w_res_init=w_res_init
        self.w_input_init=w_input_init
        def w_res_initializer():
            pass
        self.w_res_init=w_res_initializer        

    def __call__(self):
        pass



