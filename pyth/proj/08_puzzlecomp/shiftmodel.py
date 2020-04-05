'''
DNN Model to drive UCT search for a shifting puzzle
Python 3
AHN, Mar 2020
'''

from pdb import set_trace as BP
import os,shutil
import numpy as np
from numpy.random import random

import keras.layers as kl
import keras.models as km
import keras.optimizers as kopt
import keras.preprocessing.image as kp
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

#num_cores = 4
num_cores = 1
GPU=0

if GPU:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session( config=config)
    K.set_session( session)
else:
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto( intra_op_parallelism_threads=num_cores,\
                             inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
                             device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session( config=config)
    K.set_session( session)

#------------------
class ShiftModel:

    N_VALUE_DENSE_UNITS=32

    def __init__( self, size, n_filters=32, n_blocks=6):
        self.size = size
        self.model = None
        self.n_filters = n_filters
        nxn = self.size * self.size
        shape = ( self.size, self.size, nxn)
        self.model = ShiftModel.dual_conv_network( shape, n_blocks, self.n_filters)
        opt = kopt.Adam()
        self.model.compile(
            loss=['categorical_crossentropy', 'mse'],
            optimizer=opt)
        # self.model.fit(
        #     model_input, [action_target, value_target],
        #     batch_size=batch_size)
        self.model.summary()


    @classmethod
    def dual_conv_network( cls, input_shape, n_blocks, n_filters):
        'Dual convolutional architecture from Deep Learning and the Game of Go.'

        inputs = kl.Input( shape=input_shape)
        first_conv = ShiftModel.conv_bn_relu_block( name="init", n_filters=n_filters)(inputs)
        conv_tower = ShiftModel.convolutional_tower( n_blocks, n_filters)(first_conv)
        policy = ShiftModel.policy_head()( conv_tower)
        value = ShiftModel.value_head()( conv_tower)
        return km.Model( inputs=inputs, outputs=[policy, value])

    @classmethod
    # Try bn_conv_relu_block
    def conv_bn_relu_block( cls, name, n_filters, kernel_size=(3,3),
                            strides=(1,1), padding="same", init="he_normal"): # try glorot_normal or glorot_uniform
        def f(inputs):
            conv = kl.Conv2D( filters=n_filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              kernel_initializer=init,
                              #data_format='channels_first',
                              name="%s_conv_block" % name)(inputs)
            batch_norm = kl.BatchNormalization( axis=-1, name="%s_batch_norm" % name)(conv)
            return kl.Activation( "relu", name="%s_relu" % name)(batch_norm)
        return f

    @classmethod
    def convolutional_tower( cls, n_blocks, n_filters):
        def f(inputs):
            x = inputs
            for i in range( n_blocks):
                x = ShiftModel.conv_bn_relu_block( name=i, n_filters=n_filters)(x)
            return x
        return f

    @classmethod
    def policy_head( cls):
        def f(inputs):
            conv = kl.Conv2D( filters=2,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              name="policy_head_conv_block")(inputs)
            batch_norm = kl.BatchNormalization( axis=1, name="policy_head_batch_norm")(conv)
            activation = kl.Activation( "relu", name="policy_head_relu")(batch_norm)
            # Try to use 4 channels and average pooling instead
            policy_flat = kl.Flatten()( activation)
            # Four policy outputs for LEFT, RIGHT, UP, DOWN
            return kl.Dense( units=4, name="policy_head_output", activation='softmax')(policy_flat)
        return f

    @classmethod
    def value_head( cls):
        def f(inputs):
            conv = kl.Conv2D( filters=1,
                              kernel_size=(1, 1),
                              strides=(1, 1),
                              padding="same",
                              name="value_head_conv_block")(inputs)
            batch_norm = kl.BatchNormalization( axis=1, name="value_head_batch_norm")(conv)
            activation = kl.Activation( "relu", name="value_head_relu")(batch_norm)
            # Try to use 1 conv channel and average pooling instead
            value_flat = kl.Flatten()( activation)
            dense =  kl.Dense( units=ShiftModel.N_VALUE_DENSE_UNITS, name="value_head_dense", activation="relu")(value_flat)
            return kl.Dense( units=1, name="value_head_output", activation="tanh")(dense)
        return f

    def predict( self, input):
        res = self.model.predict( input[np.newaxis])
        return res

    def train_on_batch( self, inputs, targets):
        res = self.model.train_on_batch( inputs, targets)
        return res

    def save_weights( self, fname):
        weightsfname = fname + '.weights'
        if os.path.exists( weightsfname):
            shutil.move( weightsfname, weightsfname + '.bak')
        self.model.save_weights( weightsfname)

    def load_weights( self, fname):
        if not fname.endswith( '.weights'):
            fname += '.weights'
        self.model.load_weights( fname)
        return True

    def get_v_p( self, state):
        'Run the net, which has a policy head and a value head.'
        res = self.predict( state.encode())
        p = res[0][0]
        v = res[1][0]
        # v is a tanh, so in (-1,1). We need (0,1).
        v = (v+1) / 2
        # Eliminate illegal moves
        flags = state.action_flags()
        p = [ x if flags[i] else 0.0 for i,x in enumerate(p) ]
        ssum = np.sum(p)
        if ssum: p /= ssum
        return v,p
