#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Aug 26, 2017
# **********************************************************************/
#
# Build and train blobcount model
#

from __future__ import division, print_function
from pdb import set_trace as BP
import inspect
import os,sys,re,json
import numpy as np
from numpy.random import random
import argparse
#import matplotlib as mpl
#mpl.use('Agg') # This makes matplotlib work without a display
#from matplotlib import pyplot as plt
import keras.layers as kl
import keras.models as km
import keras.optimizers as kopt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut


BATCH_SIZE=1

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Build and train blobcount model
    Synopsis:
      %s --resolution <n> --epochs <n>
    Description:
      Build a NN model with Keras, train on the data in the train subfolder.
    Example:
      %s --resolution 128 --epochs 20
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# Models
# Try to find out if there is a circle or not
#================================================

# My simplest little model
#--------------------------
class SimpleModel:
    #------------------------
    def __init__(self,resolution):
        self.resolution = resolution
        self.build_model()

    #-----------------------
    def build_model(self):
        nb_colors=1
        inputs = kl.Input(shape=(nb_colors,self.resolution,self.resolution))
        x = kl.Flatten()(inputs)
        x = kl.Dense(8, activation='relu')(x)
        #x = kl.Dense(64, activation='relu', name='dense_lower')(x)
        #x = kl.Dropout(0.5)(x)
        #x = kl.Dense(64, activation='relu', name='dense_upper')(x)
        #x = kl.Dropout(0.5)(x)
        #nb_classes=2
        #outputs = kl.Dense(nb_classes, activation='softmax', name='output')(x)
        output = kl.Dense(1, activation='sigmoid')(x)
        #print(inspect.getargspec(km.Model.__init__))
        self.model = km.Model(input=inputs, output=output)
        self.model.summary()
        #self.model.compile(loss='binary_crossentropy', optimizer=kopt.Adam(), metrics=['accuracy'])
        #self.model.compile(loss='binary_crossentropy', optimizer=kopt.SGD(lr=1.0), metrics=['accuracy'])
        opt = kopt.SGD(lr=0.1)
        #self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model.compile(loss='binary_crossentropy', optimizer=opt)
        # self.model.compile(optimizer=kopt.SGD(lr=0.001),
        #           loss='categorical_crossentropy',
        #           metrics=['accuracy'])

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--resolution", required=True, type=int)
    parser.add_argument( "--epochs", required=True, type=int)
    args = parser.parse_args()
    model = SimpleModel(args.resolution)
    #BP()
    images = ut.get_data(SCRIPTPATH, (args.resolution,args.resolution))
    meta   = ut.get_meta(SCRIPTPATH)
    #BP()
    for i in range(1):
        # print('>>>>>iter %d' % i)
        # for idx,layer in enumerate(model.model.layers):
        #     weights = layer.get_weights() # list of numpy arrays
        #     print('Weights for layer %d:',idx)
        #     print(weights)
        #BP()
        model.model.fit(images['train_data'], meta['train_classes'],
                        batch_size=BATCH_SIZE, nb_epoch=args.epochs)
        # model.model.fit(images['train_data'], meta['train_classes'],
        #                 batch_size=BATCH_SIZE, nb_epoch=args.epochs,
        #                 validation_data=(images['train_data'], meta['train_classes']))
                    #validation_data=(images['valid_data'], meta['valid_classes_hot']))
    #model.model.save('dump1.hd5')
    preds = model.model.predict(images['train_data'], batch_size=BATCH_SIZE)
    # ...and the probabilities of being a cat
    #probs = model.model.predict_proba(images['train_data'], batch_size=BATCH_SIZE)[:,0]
    print(preds)

if __name__ == '__main__':
    main()
