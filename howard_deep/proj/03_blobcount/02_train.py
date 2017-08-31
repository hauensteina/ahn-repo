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
      %s --  Build and train blobcount model
    Synopsis:
      %s --resolution <n> --epochs <n> --rate <learning_rate>
    Description:
      Build a NN model with Keras, train on the data in the train subfolder.
    Example:
      %s --resolution 120 --epochs 10 --rate 0.01
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# Models
# Try to find out if there is something in the image
#=====================================================

# My simplest little model
#--------------------------
class CountModel:
    #------------------------
    def __init__(self,resolution,rate=0):
        self.resolution = resolution
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        nb_colors=1
        inputs = kl.Input(shape=(nb_colors,self.resolution,self.resolution))
        x = kl.Flatten()(inputs)
        x = kl.Dense(50, activation='relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.5)(x)
        x = kl.Dense(50, activation='relu')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Dropout(0.5)(x)
        output = kl.Dense(26, activation='sigmoid')(x)
        self.model = km.Model(input=inputs, output=output)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam(self.rate)
        else:
            opt = kopt.Adam()
        #opt = kopt.Adam(0.001)
        #opt = kopt.SGD(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--resolution", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--rate", required=False, default=0, type=float)
    args = parser.parse_args()
    model = CountModel(args.resolution, args.rate)
    images = ut.get_data(SCRIPTPATH, (args.resolution,args.resolution))
    meta   = ut.get_meta(SCRIPTPATH)
    # Normalize training and validation data by train data mean and std
    means,stds = ut.get_means_and_stds(images['train_data'])
    ut.normalize(images['train_data'],means,stds)
    ut.normalize(images['valid_data'],means,stds)

    # Load the model and train some more
    if os.path.exists('model.h5'): model.model.load_weights('model.h5')
    model.model.fit(images['train_data'], meta['train_classes_hot'],
                    batch_size=BATCH_SIZE, nb_epoch=args.epochs,
                    validation_data=(images['valid_data'], meta['valid_classes_hot']))
    model.model.save_weights('model.h5')
    # print('>>>>>iter %d' % i)
    # for idx,layer in enumerate(model.model.layers):
    #     weights = layer.get_weights() # list of numpy arrays
    #     print('Weights for layer %d:',idx)
    #     print(weights)
    #model.model.fit(images['train_data'], meta['train_classes'],
    #                batch_size=BATCH_SIZE, nb_epoch=args.epochs)
    #model.model.save('dump1.hd5')
    preds = model.model.predict(images['valid_data'], batch_size=BATCH_SIZE)
    #print(preds)

if __name__ == '__main__':
    main()
