#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Sep 4, 2017
# **********************************************************************/
#
# Build and train blackstones model
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
import keras.activations as ka

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

BATCH_SIZE=64
GRIDSIZE=5
RESOLUTION=80

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Build and train blackstones model
    Synopsis:
      %s --epochs <n> --rate <learning_rate>
    Description:
      Learn how to find the positions of Black circles on a 5x5 grid
    Example:
      %s --epochs 10 --rate 0.001
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg


# My simplest little model
#--------------------------
class StoneModel:
    #------------------------
    def __init__(self,resolution,gridsize,rate=0):
        self.resolution = resolution
        self.gridsize = gridsize
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        nb_colors=1
        nf=16
        inputs = kl.Input(shape=(nb_colors,self.resolution,self.resolution))
        #x = kl.Convolution2D(nf,(3,3), activation='relu', border_mode='same')(inputs)
        x = kl.Conv2D(nf,(3,3), activation='relu', padding='same')(inputs)
        x = kl.BatchNormalization(axis=1)(x)
        # 80x80 -> 40x40
        x = kl.MaxPooling2D()(x)
        x = kl.Conv2D(nf,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        # 40x40 -> 20x20
        x = kl.MaxPooling2D()(x)
        x = kl.Conv2D(nf,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        # 20x20 -> 10x10
        x = kl.MaxPooling2D()(x)
        x = kl.Conv2D(nf,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        # 10x10 -> 5x5
        x = kl.MaxPooling2D()(x)
        # Now try to get at stone or no stone with two channels only
        #x = kl.Convolution2D(2,3,3, activation='tanh', border_mode='same')(inputs)
        x = kl.Conv2D(2,(3,3), padding='same')(x)
        x = kl.Flatten()(x)
        # softmax across the two channels, gives one 5x5 matrix
        # outputs = ka.softmax(x,axis=1)
        #outputs = kl.Activation('softmax',axis=1)(x)
        outputs = kl.Activation('sigmoid')(x)
        self.model = km.Model(inputs=inputs, outputs=outputs)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam(self.rate)
        else:
            opt = kopt.Adam()
        #opt = kopt.Adam(0.001)
        #opt = kopt.SGD(lr=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        #self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    #parser.add_argument("--resolution", required=True, type=int)
    #parser.add_argument("--gridsize", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--rate", required=False, default=0, type=float)
    args = parser.parse_args()
    model = StoneModel(RESOLUTION, GRIDSIZE, args.rate)
    images = ut.get_data(SCRIPTPATH, (RESOLUTION,RESOLUTION))
    meta   = ut.get_output_by_key(SCRIPTPATH,'stones')
    # Normalize training and validation data by train data mean and std
    means,stds = ut.get_means_and_stds(images['train_data'])
    ut.normalize(images['train_data'],means,stds)
    ut.normalize(images['valid_data'],means,stds)

    train_output = meta['train_output'] # Lists of len 25 all ones and zeros
    for li in train_output:
        li += [1-x for x in li]
    valid_output = meta['valid_output'] # Lists of len 25 all ones and zeros
    for li in valid_output:
        li += [1-x for x in li]

    # Load the model and train
    if os.path.exists('model.h5'): model.model.load_weights('model.h5')
    #BP()
    model.model.fit(images['train_data'], train_output,
                    batch_size=BATCH_SIZE, epochs=args.epochs,
                    #validation_data=(images['valid_data'], valid_output))
                    validation_data=(images['train_data'], train_output))
    model.model.save_weights('model.h5')
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
