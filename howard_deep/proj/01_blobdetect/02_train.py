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

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

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
# Try to count up to ten blobs per image
#==========================================

# One dense layer, output one hot
#----------------------------------
class Dense1:
    #------------------------
    def __init__(self,resolution):
        self.resolution = resolution
        self.build_model()

    #-----------------------
    def build_model(self):
        inputs = kl.Input(shape=(3,self.resolution,self.resolution))
        x = kl.Flatten()(inputs)
        x = kl.Dense(64, activation='relu')(x)
        x = kl.Dense(64, activation='relu')(x)
        predictions = kl.Dense(4, activation='softmax', name='class')(x)
        #print(inspect.getargspec(km.Model.__init__))
        self.model = km.Model(input=inputs, output=predictions)
        self.model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--resolution", required=True, type=int)
    parser.add_argument( "--epochs", required=True, type=int)
    args = parser.parse_args()
    model = Dense1(args.resolution)
    batch_size=4
    images = ut.get_data(SCRIPTPATH, (args.resolution,args.resolution))
    meta   = ut.get_meta(SCRIPTPATH)
    #BP()
    model.model.fit(images['train_data'], meta['train_classes_hot'],
                    batch_size=batch_size, nb_epoch=args.epochs,
                    validation_data=(images['train_data'], meta['train_classes_hot']))
                    #validation_data=(images['valid_data'], meta['valid_classes_hot']))
    BP()
    tt=42

if __name__ == '__main__':
    main()
