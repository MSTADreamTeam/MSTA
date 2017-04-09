# This file defines the ESN

from base_algo import BaseAlgo

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell
from tensorflow.python.ops import init_ops, math_ops, random_ops, array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework.ops import convert_to_tensor

class ESN(BaseAlgo):
    ''' This class defines the Echo State Network
    This is a specific case of Recurrent Neural Network using a reservoir of sparcely connected neurons
    and recurrent cells to feed an output layer which is the only one fitted '''
    def __init__(self, global_hyperparams, hp_grid = None, n_res=None, sparcity_ratio=None, 
                 distrib=None, spectral_radius=None, input_scaling=None, activation='tanh',
                 leaking_rate=None, regularization=None, alpha=None, l1_ratio=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.algo_type='ML'
        self.n_res=n_res # Size of the reservoir
        self.sparcity_ratio=sparcity_ratio # Ratio of sparcity of the reservoir
        self.distrib=distrib # Distribution of the non zero elements of the weights of the reservoir
        self.spectral_radius=spectral_radius # The spectral radius of the weight matrix of the reservoir, spectral_radius<1 to have the echo state property
        self.input_scaling=input_scaling # The method/params of the input layer weights scaling
        self.activation=getattr(tf.nn,activation) # The function used in the activation of the network
        self.leaking_rate=leaking_rate # The speed at which the reservoir will update through time
        self.regularization=regularization # The regularization used in the loss function
        self.alpha=alpha # The alpha parameter of the regularization
        self.l1_ratio=l1_ratio # the l1 ratio parameter of an ElasticNet regularization

    def init_rnn_cell(self, input_size):
        ''' Initialize the RNN cell, the reservoir '''
        self.rnn_cell=ESNCell(input_size, self.n_res, self.spectral_radius, self.sparcity_ratio, self.leaking_rate, self.activation)
        return self

    def select_data(self, X):
        # should we rewrite the select data here?
        return super().select_data(X)

    def predict(self, X_test, pred_index=None):
        state=self.rnn_cell.zero_state() # Are we sure?
        predicted_values=self.rnn_cell(X_test, state)
        if pred_index is not None:
            self._store_predicted_values(pred_index,predicted_values)
        return predicted_values

    def fit(self, X_train, Y_train):
        self.init_rnn_cell(len(X_train)) # We initialize the resevoir
        state=self.rnn_cell.zero_state() # We fix the first state as the zero state
        for i in X_train.index:
            state = self.rnn_cell(X_train.iloc[i], state) # We update the state foro each data in our time serie
        return self




class ESNCell(_RNNCell):
    ''' Subclass of tensorflow Recurrent Neural Net cell, used to model the reservoir of the ESN
    Code inspired from https://github.com/m-colombo/Tensorflow-EchoStateNetwork/blob/master/esn_cell.py '''
    def __init__(self, input_size, n_res, n2_scale, sparcity_ratio, leaking_rate, activation, 
                 w_input_init=init_ops.random_normal_initializer(),
                 w_res_init=init_ops.random_normal_initializer(), 
                 b_init=init_ops.random_normal_initializer()):
        self.n_res=n_res
        self.leaking_rate=leaking_rate
        self.activation=activation
        self.w_input=tf.Variable(initial_value=w_input_init, trainable=False, shape=[input_size, n_res])
        self.b=tf.Variable(initial_value=b_init, trainable=False, shape=[n_res])
        def w_res_initializer(shape):   
            ''' Initialize the random weights and scale them using norm 2, an upper bound for spectral radius,
            to make sure the echo state property is verified
            '''
            w=w_res_init(shape)
            connectivity_mask=math_ops.cast(math_ops.less_equal(random_ops.random_uniform(shape),sparcity_ratio))
            w=math_ops.mul(w, connectivity_mask)
            w_norm2=math_ops.sqrt(math_ops.reduce_sum(math_ops.square(w)))
            is_norm_0=math_ops.cast(math_ops.equal(wr_norm2, 0))
            w=w*n2_scale/(w_norm2+1*is_norm_0)
            return w
        self.w_res=tf.Variable(initial_value=self.w_res_initializer, trainable=False, shape=[n_res, n_res])
        
    @property
    def output_size(self):
        return self.n_res

    @property
    def state_size(self):
        return self.n_res
    
    def __call__(self, inputs, state, scope=None):
        ''' Run one step of the ESN Cell '''
        inputs=convert_to_tensor(inputs)
        input_size = inputs.get_shape().as_list()[1]
        in_mat = array_ops.concat(1, [inputs, state])
        weights_mat = array_ops.concat(0, [self.w_input, self.w_res])
        output = (1-self.leaking_rate)*state+self.leaking_rate*self.activation(math_ops.matmul(in_mat, weights_mat)+self.b)
        return output

