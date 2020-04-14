'''
DNN Model to drive UCT search for a shifting puzzle
Python 3
AHN, Apr 2020
'''

from pdb import set_trace as BP
import os,shutil
import numpy as np
from numpy.random import random

import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as kopt
import tensorflow.keras.preprocessing.image as kp
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint

'''
num_cores = 3
#num_cores = 1
GPU=1

if GPU:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session( config=config)
    tf.compat.v1.keras.backend.set_session( session)
    #K.set_session( session)
else:
    tf.config.experimental.set_visible_devices([], 'GPU')
    # num_CPU = 1
    # num_GPU = 0
    # config = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=num_cores,\
    #                          inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
    #                          device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    # session = tf.compat.v1.Session( config=config)
    # tf.compat.v1.keras.backend.set_session( session)
    #K.set_session( session)
'''

#------------------
class ShiftModel:

    def __init__( self, size, mode, n_units=32, n_layers=8):
        self.size = size
        self.model = None
        self.n_units = n_units
        self.n_layers = n_layers
        self.mode = mode # 'v' or 'd'
        nxn = self.size * self.size
        shape = ( self.size, self.size, nxn)
        self.model = self.value_dense_network( shape, self.n_units, self.n_layers)
        opt = kopt.Adam()
        if self.mode == 'v':
            self.model.compile(
                loss=['mse'], metrics=['mse'] # TODO: Use custom metric here
            )
        else:
            self.model.compile(
                loss=['mse'], metrics=['mse'] # TODO: Use custom metric here
            )

        self.model.summary()

    def value_dense_network( self, input_shape, n_units, n_layers):
        inputs = kl.Input( shape=input_shape)
        layer = kl.Flatten()( inputs)
        for idx in range( n_layers):
            lname = 'dense_%d' % idx
            layer = kl.Dense( units=n_units, name=lname, kernel_initializer='glorot_uniform', activation='relu')(layer)
        if self.mode == 'v':
            output_layer = kl.Dense( units=1, name='value_head_output', activation='tanh')(layer)
        else:
            output_layer = kl.Dense( units=1, name='value_head_output', activation='linear')(layer)
        return km.Model( inputs=inputs, outputs=[output_layer])

    def predict( self, input):
        res = self.model.predict( input[np.newaxis])[0][0]
        if mode == 'v': # res is a tanh, so in (-1,1). We need (0,1).
            res = (res+1) / 2.0
        return res

    def train_on_batch( self, inputs, targets):
        res = self.model.train_on_batch( inputs, targets)
        return res

    def save_weights( self, fname):
        #if not fname.endswith( '.h5'):
        #    fname += '.h5'
        if os.path.exists( fname):
            shutil.move( fname, fname + '.bak')
        self.model.save_weights( fname)

    def load( self, fname):
        #if not fname.endswith( '.h5'):
        #    fname += '.h5'
        BP()
        try:
            self.model = km.load_model( fname)
        except: # Try again. Collision between train.py and generate.py
            self.model = km.load_model( fname)
        return True

    # def get_v( self, state):
    #     'Run the net, return value estimate as a scalar.'
    #     res = self.predict( state.encode())
    #     v = res[0][0]
    #     # v is a tanh, so in (-1,1). We need (0,1).
    #     v = (v+1) / 2.0
    #     return v
