#!/usr/bin/env python

# /********************************************************************
# Filename: vimodel.py
# Author: AHN
# Creation Date: Mar 2020
# **********************************************************************/
#
# A model for value iteration to solve an nxn shifting puzzle.
# Input representation:
# n*n 2D layers of size nxn with one bit set to indicate tile position.
# Output:
# A float estimate of remaining number of moves to solution (cost to go).

from pdb import set_trace as BP

import os,sys,re,json, shutil
import numpy as np

from numpy.random import random
import keras.layers as kl
import keras.models as km
import keras.optimizers as kopt
import keras.preprocessing.image as kp
import tensorflow as tf
from keras import backend as K

#==================
class VIModel:
    #-------------------------
    def __init__( self, size):
        self.size = size
        self.build_model()

    #---------------------------------
    def add_layers( self, inputs):
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

    # Input has nxn channels of width=height=n, one bit set in each channel.
    #------------------------------------------------------------------------
    def build_model( self):
        nxn = self.size * self.size
        inputs = kl.Input( shape = ( self.size, self.size, nxn), name = 'puzzle')
        value_out = self.add_layers( inputs)

        # Just use highest convolutional layer as output
        self.model = km.Model( inputs=inputs, outputs=value_out)

        opt = kopt.Adam()
        self.model.compile( loss='mse', optimizer=opt, metrics=['accuracy'])

        self.model.summary()

    #----------------------------
    def predict( self, inputs):
        res = self.model.predict( inputs[np.newaxis])
        return res

    #--------------------------------------------
    def train_on_batch( self, inputs, targets):
        res = self.model.train_on_batch( inputs, targets)
        return res

    #---------------------------------
    def save_weights( self, fname):
        weightsfname = fname + '.weights'
        if os.path.exists( weightsfname):
            shutil.move( weightsfname, weightsfname + '.bak')
        self.model.save_weights( weightsfname)

    #---------------------------------
    def load_weights( self, fname):
        weightsfname = fname + '.weights'
        if not os.path.exists( weightsfname):
            return False
        self.model.load_weights( weightsfname)
        return True
