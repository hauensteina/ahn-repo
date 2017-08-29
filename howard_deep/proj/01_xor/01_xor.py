#!/usr/bin/env python

# /********************************************************************
# Filename: xor.py
# Author: AHN
# Creation Date: Aug 28, 2017
# **********************************************************************/
#
# Train network on xor problem.
# Adapted from github stewartpark/xor.py
# Two dummy dimensions always zero have been added.
#

from pdb import set_trace as BP
import sys
import numpy as np

import keras.layers as kl
import keras.models as km
import keras.optimizers as kopt

X = np.array([[0,0,0,0],[0,255,0,0],[255,0,0,0],[255,255,0,0]])
X2D = np.array([
    [[[0,0],[0,0]]],
    [[[0,255],[0,0]]],
    [[[255,0],[0,0]]],
    [[[255,255],[0,0]]]
])
# XOR on the first two elements
y = np.array([[0],[1],[1],[0]])
# Training will fail unless you normalize your input data
X = X - np.mean(X)
X = X / np.std(X)
# Make it look like four images with one channel two rows two cols
X2D = np.reshape(X,(4,1,2,2))

nb_epoch=100
try:
    mode=sys.argv[1]
except:
    mode='sequential'

# Functional interface
if mode == 'func':
    inputs = kl.Input(shape=(4,))
    x = kl.Dense(8,activation='relu')(inputs)
    output = kl.Dense(1,activation='sigmoid')(x)
    model = km.Model(input=inputs, output=output)
    sgd = kopt.SGD(lr=0.2)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.fit(X, y, batch_size=1, nb_epoch=nb_epoch)
    print(model.predict(X, batch_size=1))
# How to handle grayscale images (2D, 1 channel)
elif mode == 'func2d':
    inputs = kl.Input(shape=(1,2,2))
    x = kl.Flatten()(inputs)
    x = kl.Dense(8,activation='relu')(x)
    output = kl.Dense(1,activation='sigmoid')(x)
    model = km.Model(input=inputs, output=output)
    sgd = kopt.SGD(lr=0.2)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.fit(X2D, y, batch_size=1, nb_epoch=nb_epoch)
    print(model.predict(X2D, batch_size=1))
# Sequential interface
elif mode == 'sequential':
    model = km.Sequential()
    model.add(kl.Dense(8,input_dim=4))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(1))
    model.add(kl.Activation('sigmoid'))
    sgd = kopt.SGD(lr=0.2)
    model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.fit(X, y, batch_size=1, nb_epoch=nb_epoch)
    print(model.predict_proba(X))
else:
    print 'unknown mode'

"""
after 100 epochs:
[[ 0.00906752]
 [ 0.9968642 ]
 [ 0.9969992 ]
 [ 0.0043143 ]]
"""
