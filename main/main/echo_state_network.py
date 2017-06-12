# This file defines the ESN using Keras library with Tensorflow backend
from base_algo import BaseAlgo
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import Recurrent, SimpleRNN
from keras.initializers import RandomUniform, RandomNormal
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras import backend as K
from tensorflow.python.framework.ops import convert_to_tensor

class ESN(BaseAlgo):
    ''' This class defines the Echo State Network
    This is a specific case of Recurrent Neural Network using a reservoir of sparcely connected neurons
    servign as a recurrent cell to feed an output layer which is the only one trained
    IMPORTANT NOTE: When you feed the ESN with training data, always make sure to feed data with a single timestamp, not multiple lags
    This is crucial to make sure the algorithm behaves as needed
    '''
    def __init__(self, global_hyperparams, hp_grid = None, n_res=None, sparcity_ratio=None, 
                 spectral_radius=None, input_scaling=None, activation='tanh', leaking_rate=None,
                 regularization=None, alpha=None, l1_ratio=None):
        BaseAlgo.__init__(self, global_hyperparams, hp_grid)
        self.algo_type='ML'
        self.n_res=n_res # Numner of neurons in the reservoir
        self.sparcity_ratio=sparcity_ratio # Ratio of sparcity of the reservoir
        self.spectral_radius=spectral_radius # The spectral radius of the weight matrix of the reservoir,
        # spectral_radius<1 to have the echo state property, but sometimes the optimal value can be greater
        #  than one, hence do not hesitate to try it
        self.input_scaling=input_scaling # The method/params of the input layer weights scaling NOT USED
        self.activation=activation # The function used in the activation of the network beween all layers
        self.leaking_rate=leaking_rate # The speed at which the reservoir will update through time
        if regularization=='Lasso':  # Regularization of the output layer
            self.reg=l1(alpha)
        elif regularization=='Ridge':
            self.reg=l2(alpha)
        elif regularization=='ElasticNet':
            self.reg=l1_l2(alpha*l1_ratio,alpha*(1-l1_ratio))
        else:
            self.reg=None

    def init_net(self, input_size, batch_size):
        ''' Initialize the network 
        Reinitializing it erase the memory of the network '''
        self.model=Sequential()
        #self.model.add(Dense(input_size, activation=self.activation, input_shape=()))
        self.model.add(ESNCell(input_size, batch_size, self.n_res, self.spectral_radius, self.sparcity_ratio, 
                               self.leaking_rate, self.activation))
        n_outputs=1 if self.global_hyperparams['output_type']=='R' else 3 
        self.model.add(Dense(n_outputs, kernel_regularizer=self.reg)) # Activate the output layer non linearly?
        if self.global_hyperparams['output_type']=='C':
            loss='categorial_crossentropy'
            metric='accuracy'
        else:
            loss='mean_squared_error'
            metric=None
        self.model.compile(optimizer=SGD(lr=1), loss=loss, metrics=[metric])
        return self

    #def select_data(self, X):
        # Make sure we don't take any lags, by default we take all the input
     #   self.selected_data=[True for col in X.columns]
      #  return self.selected_data

    def predict(self, X_test, pred_index = None):
        # check the shape of X with the batch
        # check the output too, in terms of categorical shape
        return super().predict(X_test, pred_index, batch_size=1)

    def fit(self, X_train, Y_train, verbose=0):
        self.init_net(len(X_train),1)
        if self.global_hyperparams['output_type']=='C': Y_train=to_categorical(Y_train)
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
            an upper bound for spectral radius, to make sure the echo state property is verified.
            It also takes care of the sparcity of the connections
            NOTE: find a way to compute the spectral radius to release the constraint
            '''
            w=w_res_init(shape)
            connectivity_mask=K.cast(K.less_equal(RandomUniform(shape),sparcity_ratio))
            w=K.dot(w,connectivity_mask)
            w_norm2=K.sqrt(K.sum(K.square(w)))
            is_norm_0=K.cast(K.equal(w_norm2, 0))
            w=w*norm_scale/(w_norm2+1*is_norm_0)
            return w
        SimpleRNN.__init__(self, n_res, activation=activation, use_bias=True, kernel_initializer=w_res_initializer, 
                           recurrent_initializer=w_res_init, bias_initializer=b_init,
                           dropout=0.0, recurrent_dropout=0.0, stateful=True, 
                           batch_input_shape=(batch_size, input_size,), trainable=False)
        # RQ on the recurrent cell init:   
        # Should we activate the cell or only activate inside it?
        # Check that the trainable=False works well
        self.leaking_rate=leaking_rate
        
    
    def build(self, input_shape):
        ''' Initialize the weights of the ESN Cell '''    
        SimpleRNN.build(self, input_shape)
        #self.recurrent_kernel.trainable=False
        #self.kernel.trainable=False
        #self.bias.trainable=False
        
    def call(self, inputs, state):
        ''' Run one step of the ESN Cell '''
        #inputs=convert_to_tensor(inputs)
        in_mat = K.concatenate(1, [inputs, state])
        weights_mat = K.concatenate(0, [self.kernel, self.recurrent_kernel])
        output = (1-self.leaking_rate)*state+self.leaking_rate*self.activation(K.dot(in_mat, weights_mat)+self.bias)
        return output

    def compute_output_shape(self, input_shape):
        # should we overload it?
        return super().compute_output_shape(input_shape)

