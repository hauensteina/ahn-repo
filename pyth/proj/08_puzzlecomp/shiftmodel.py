'''
DNN Model to drive UCT search for a shifting puzzle
Python 3
AHN, Mar 2020
'''

from pdb import set_trace as BP
#import os,sys,re,json,shutil
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

#==================
class ShiftModel:
    def __init__( self, size):
        self.size = size
        self.model = None
        self.__build_model()

    def __add_layers( self, inputs):
        NFILTERS = 4 * self.size * self.size
        NLAYERS = 6

        x = inputs
        for i in range( NLAYERS):
            x = kl.BatchNormalization(axis=-1)(x)
            x = kl.Conv2D( NFILTERS, (3,3), activation='relu', padding='same', name='x_%03d' % i)(x)

        # Convolutional single channel, dense layer, tanh for value estimate
        lastconv = kl.Conv2D( 1, (3,3), padding='same', name='lastconv')(x)
        value_flat = kl.Flatten()(lastconv)
        # Sigmoid ranges from 0 (far away from solution) to 1 (found solution)
        value_out = kl.Dense( 1, activation='sigmoid', name='value_out')(value_flat)
        return value_out

    def __build_model( self):
        ' Input has nxn channels of width=height=n, one bit set in each channel. '
        nxn = self.size * self.size
        inputs = kl.Input( shape = ( self.size, self.size, nxn), name = 'puzzle')
        value_out = self.__add_layers( inputs)

        # Just use highest convolutional layer as output
        self.model = km.Model( inputs=inputs, outputs=value_out)

        opt = kopt.Adam()
        self.model.compile( loss='mse', optimizer=opt, metrics=['accuracy'])

        self.model.summary()

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
        weightsfname = fname + '.weights'
        self.model.load_weights( weightsfname)
        return True

    def get_v_p( self, state):
        '''
        Run the net, which only gives us v. Derive p from the v values
        of the children.
        '''
        # Current quality
        v = self.predict( state.encode())[0][0]

        # quality with lookahead 1 gives p
        p = np.zeros( 4, float)
        actions = state.action_list()
        for action in actions:
            next_state = state.act( action)
            p[action] = self.predict( next_state.encode())

        ssum = np.sum(p)
        if ssum:
            p /= ssum

        return v,p
