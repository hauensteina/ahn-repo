#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Sep 21, 2017
# **********************************************************************/
#
# Build and train gcount model
#

from __future__ import division, print_function
from pdb import set_trace as BP
import inspect
import os,sys,re,json,shutil
import numpy as np
import scipy
from numpy.random import random
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt
import keras.layers as kl
import keras.layers.merge as klm
import keras.models as km
import keras.optimizers as kopt
import keras.activations as ka
import keras.backend as K
import theano as th

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

BATCH_SIZE=1
GRIDSIZE=0
RESOLUTION=0
MODELFILE='model.h5'
WEIGHTSFILE='weights.h5'
EMPTY=0
WHITE=1
BLACK=2
NCOLORS=3


inputs = kl.Input(shape=(2,3,3))
chan0  = kl.Lambda(lambda x: x[:,0,:,:], output_shape=(3,3)) (inputs)
chan1  = kl.Lambda(lambda x: x[:,1,:,:], output_shape=(3,3)) (inputs)
flat0  = kl.Flatten(name='flat0')(chan0)
flat1  = kl.Flatten(name='flat1')(chan1)
#sum0   = kl.dot((flat0,K.ones(9)),1)
#sum0   = kl.dot([flat0,K.variable(np.ones(1,9))],axes=1,name='sum0')
sum0   = kl.Lambda(lambda x: K.dot(x,K.ones(9)).reshape((1,1)), output_shape=((1,)), name='sum0') (flat0)

# flat0  = kl.Flatten(name='flat0')(chan0)
# flat1  = kl.Flatten(name='flat1')(chan1)
# count0 = kl.Lambda(lambda x: K.sum(K.equal( K.ones(3*3) * 3, x)).astype('float32').reshape((1,1)),
#                      output_shape=(1,),name='count0') (flat0)
# count1 = kl.Lambda(lambda x: K.sum(K.equal( K.ones(3*3) * 3, x)).astype('float32').reshape((1,1)),
#                      output_shape=(1,),name='count1') (flat1)
#out    = kl.concatenate([count0,count1],name='out')

#out    = kl.concatenate([flat0,flat1],name='out')

model = km.Model(inputs=inputs, outputs=sum0)
#self.model.compile(loss='mse', optimizer=kopt.Adam(), metrics=[ut.element_match])
model.compile(loss='mse', optimizer=kopt.Adam(), metrics=['accuracy'])
model.summary()

#data = np.array([[1,2,3,4,5,6,7,8,9], [10,11,12,3,3,15,16,17,18]]).reshape((1,2,3,3))
data = np.array([[1,2,3,4,5,6,7,8,9], [1,1,1,1,1,1,1,1,1]]).reshape((1,2,3,3))

flat0 = ut.get_output_of_layer(model,'flat0',data)
print('flat0'); print(flat0)

#flat1 = ut.get_output_of_layer(model,'flat1',data)
#print('flat1'); print(flat1)

sum0 = ut.get_output_of_layer(model,'sum0',data)
print('sum0'); print(sum0)

# count0 = ut.get_output_of_layer(model,'count0',data)
# print('count0'); print(count0)

# count1 = ut.get_output_of_layer(model,'count1',data)
# print('count1'); print(count1)

# out = ut.get_output_of_layer(model,'out',data)
# print('out'); print(out)
