#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Nov 29, 2018
# **********************************************************************/
#
# Train a two class 1D time series model
#

from __future__ import division, print_function
from pdb import set_trace as BP
import inspect
import os,sys,re,json
import numpy as np
from numpy.random import random
import argparse
import keras.layers as kl
import keras.models as km
import keras.optimizers as kopt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

BATCH_SIZE=64

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s -- Build and train a two class time series model
    Synopsis:
      %s --epochs <n> --rate <rate>
    Description:
      Build a NN model with Keras, train on the data in the train subfolder.
    Example:
      %s --epochs 100 --rate 0.001
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# My simplest little model
#--------------------------
class SimpleModel:
    #---------------------------------
    def __init__(self,length, input_shape, rate):
        self.input_shape = input_shape
        self.length = length
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        inputs = kl.Input( shape=self.input_shape)
        x = kl.Flatten()(inputs)
        x = kl.Dense( 4, activation='relu')(x)
        output = kl.Dense( 2, activation='softmax')(x)
        self.model = km.Model( inputs=inputs, outputs=output)
        self.model.summary()
        opt = kopt.Adam( self.rate)
        #opt = kopt.Adam(0.001)
        #opt = kopt.SGD(lr=0.01)
        self.model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    LENGTH = 100
    BATCH_SIZE = 1

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--epochs", required=True, type=int)
    parser.add_argument( "--rate", required=True, type=float)
    args = parser.parse_args()

    train_data, train_classes = ut.read_series( SCRIPTPATH + '/train/all_files', ['y'])
    valid_data, valid_classes = ut.read_series( SCRIPTPATH + '/valid/all_files', ['y'])
    # train_data = np.array( [[1,2,3,4,5],
    #                         [10,20,30,40,50]], float)
    # train_classes = ut.onehot( [0,1])

    # valid_data = np.array( [[1,2,3,4,5],
    #                         [10,20,30,40,50]], float)
    # valid_classes = ut.onehot( [0,1])

    model = SimpleModel( LENGTH, (train_data.shape[1],train_data.shape[2]), args.rate)
    model.model.fit( train_data, train_classes,
                     batch_size=BATCH_SIZE,
                     epochs=args.epochs,
                     validation_data=(valid_data, valid_classes))
    # preds = model.model.predict(images['valid_data'], batch_size=BATCH_SIZE)
    # print(preds)

if __name__ == '__main__':
    main()
