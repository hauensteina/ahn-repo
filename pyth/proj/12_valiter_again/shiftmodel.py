'''
DNN Model to drive valiter search for a shifting puzzle
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

#------------------
class ShiftModel:

    #def __init__( self, size, n_units=32, n_layers=16):
    def __init__( self, size, n_units=32, n_layers=8):
        self.size = size
        self.model = None
        self.n_units = n_units
        self.n_layers = n_layers
        nxn = self.size * self.size
        shape = ( self.size, self.size, nxn)
        self.model = self.value_dense_network( shape, self.n_units, self.n_layers)
        opt = kopt.Adam()
        self.model.compile( loss=['mse'], metrics=['mse'])
        self.model.summary()

    def value_dense_network( self, input_shape, n_units, n_layers):
        inputs = kl.Input( shape=input_shape)
        layer = kl.Flatten()( inputs)
        for idx in range( n_layers):
            lname = 'dense_%d' % idx
            layer = kl.Dense( units=n_units, name=lname, kernel_initializer='glorot_uniform', activation='relu')(layer)
        output_layer = kl.Dense( units=1, name='value_head_output', activation='tanh')(layer)
        return km.Model( inputs=inputs, outputs=[output_layer])

    def predict( self, inputs):
        res = self.model.predict( inputs)
        return res

    def predict_one( self, inpt):
        res = self.model.predict( inpt[np.newaxis])
        return res[0][0]

    def save( self, fname):
        self.model.save( fname)

    def load( self, fname):
        try:
            self.model = km.load_model( fname)
        except: # Try again. Collision between train.py and generate.py
            self.model = km.load_model( fname)
        return True

    @classmethod
    def from_file( cls, fname):
        ' Reconstruct a ShiftModel model from an hd5 file '
        model = km.load_model( fname)
        size = model.layers[0].output_shape[0][1]
        names = [ layer.name for layer in model.layers ]
        n_layers = len( [ n for n in names if 'dense' in n ] )
        n_units = model.layers[2].output_shape[1]
        res = cls( size, n_units, n_layers)
        res.model = model
        return res
