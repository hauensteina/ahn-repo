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

def mean_pred(y_true, y_pred):
    ' First attempt at a custom metric '
    return K.mean(y_pred)

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
                #loss=['mse'], metrics=[mean_pred] # TODO: Use custom metric here
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

    def predict( self, inputs):
        res = self.model.predict( inputs)
        if self.mode == 'v': # res is a tanh, so in (-1,1). We need (0,1).
            res = (res+1) / 2.0
        return res

    def train_on_batch( self, inputs, targets):
        res = self.model.train_on_batch( inputs, targets)
        return res

    def save( self, fname):
        self.model.save( fname)

    def load( self, fname):
        try:
            self.model = km.load_model( fname)
        except: # Try again. Collision between train.py and generate.py
            self.model = km.load_model( fname)
        return True
