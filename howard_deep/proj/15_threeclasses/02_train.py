#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Feb 16, 2018
# **********************************************************************/
#
# Build and train threeclasses model for Go board intersections (BEW)
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


BATCH_SIZE=8

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Build and train three classes model for Go board intersections (BEW)
    Synopsis:
      %s --resolution <n> --epochs <n> --rate <learning_rate>
    Description:
      Build a NN model with Keras, train on the data in the train subfolder.
    Example:
      %s --resolution 23 --epochs 10 --rate 0.001
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# The model
#===================================================================================================
class BEWModel:
    #------------------------------
    def __init__(self, resolution, rate=0):
        self.resolution = resolution
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        nb_colors=1
        inputs = kl.Input( shape = ( nb_colors, self.resolution, self.resolution))
        x = kl.Flatten()(inputs)
        x = kl.Dense( 3, activation='relu')(x)
        #x = kl.Dense( 16, activation='relu')(x)
        #x = kl.Dense(4, activation='relu')(x)
        output = kl.Dense( 3,activation='softmax', name='class')(x)
        self.model = km.Model(inputs=inputs, outputs=output)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam(self.rate)
        else:
            opt = kopt.Adam()
        #opt = kopt.Adam(0.001)
        #opt = kopt.SGD(lr=0.01)
        self.model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#===================================================================================================

# Get metadata from the image filenames
#-----------------------------------------
def get_meta_from_fnames( path):
    batches = ut.get_batches(path, shuffle=False, batch_size=1)
    train_batches = batches['train_batches']
    valid_batches = batches['valid_batches']

    train_classes=[]
    for idx,fname in enumerate(train_batches.filenames):
        if '/B_' in fname:
            train_classes.append(0)
        elif '/E_' in fname:
            train_classes.append(1)
        elif '/W_' in fname:
            train_classes.append(2)
        else:
            print( 'ERROR: Bad filename %s' % fname)
            exit(1)
    train_classes_hot = ut.onehot(train_classes)

    valid_classes=[]
    for idx,fname in enumerate(valid_batches.filenames):
        if '/B_' in fname:
            valid_classes.append(0)
        elif '/E_' in fname:
            valid_classes.append(1)
        elif '/W_' in fname:
            valid_classes.append(2)
        else:
            print( 'ERROR: Bad filename %s' % fname)
            exit(1)
    valid_classes_hot = ut.onehot(valid_classes)

    res = {
        'train_classes':train_classes,
        'train_classes_hot':train_classes_hot,
        'train_filenames':train_batches.filenames,
        'valid_classes':valid_classes,
        'valid_classes_hot':valid_classes_hot,
        'valid_filenames':valid_batches.filenames
    }
    return res

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--resolution", required=True, type=int)
    parser.add_argument( "--epochs", required=True, type=int)
    parser.add_argument( "--rate", required=True, type=float)
    args = parser.parse_args()
    model = BEWModel( args.resolution, args.rate)
    images = ut.get_data( SCRIPTPATH, (args.resolution,args.resolution))
    meta   = get_meta_from_fnames( SCRIPTPATH)
    # Normalize training and validation data by train data mean and std
    means,stds = ut.get_means_and_stds(images['train_data'])
    ut.normalize(images['train_data'],means,stds)
    ut.normalize(images['valid_data'],means,stds)
    model.model.fit(images['train_data'], meta['train_classes_hot'],
                    batch_size=BATCH_SIZE, epochs=args.epochs,
                    validation_data=(images['valid_data'], meta['valid_classes_hot']))
    # print('>>>>>iter %d' % i)
    # for idx,layer in enumerate(model.model.layers):
    #     weights = layer.get_weights() # list of numpy arrays
    #     print('Weights for layer %d:',idx)
    #     print(weights)
    #model.model.fit(images['train_data'], meta['train_classes'],
    #                batch_size=BATCH_SIZE, epochs=args.epochs)
    #model.model.save('dump1.hd5')
    preds = model.model.predict(images['valid_data'], batch_size=BATCH_SIZE)
    print(preds)

if __name__ == '__main__':
    main()