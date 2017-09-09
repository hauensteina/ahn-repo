#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Sep 9, 2017
# **********************************************************************/
#
# Build and train stonefeatures_conv model
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
      %s --  Build and train stonefeatures model
    Synopsis:
      %s --epochs <n> --rate <learning_rate>
    Description:
      Try to train and visualize a feature map for blak and white stones.
    Example:
      %s --epochs 10 --rate 0.001
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg


#--------------------------
class MapModel:
    #----------------------------------------------
    def __init__(self,resolution,gridsize,rate=0):
        self.resolution = resolution
        self.gridsize = gridsize
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        inputs = kl.Input(shape=(1,self.resolution,self.resolution))
        flatinp = kl.Flatten()(inputs)
        #x = kl.Dense(512, activation='softmax', name='dense_0')(x)
        dense0 = kl.Dense(4, activation='selu', name='dense_0')(flatinp)
        #x = kl.BatchNormalization()(x)
        dense1 = kl.Dense(4, activation='selu', name='dense_1')(dense0)
        #x = kl.BatchNormalization()(x)
        x = kl.Dense(4, activation='selu', name='dense_2')(dense1)
        #x = kl.BatchNormalization()(x)
        x = kl.Dense(4, activation='softmax', name='dense_3')(x)
        #x = kl.BatchNormalization()(x)
        # 1 or 0 for Black or White
        x_class  = kl.Dense(2,activation='softmax', name='class')(dense0)
        # center_x, center_y, r
        x_circle = kl.Dense(3, name='circle')(x)
        self.model = km.Model(inputs=inputs, outputs=[x_class,x_circle])
        # self.model = km.Model(inputs=inputs, outputs=x_circle)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam(self.rate)
        else:
            opt = kopt.Adam()
        self.model.compile(loss=['categorical_crossentropy','mse'], optimizer=opt,
                           metrics=['accuracy'], loss_weights=[1.,1.])
        # self.model.compile(loss='mse', optimizer=opt,
        #                    metrics=['accuracy'])

    #------------------------------------------------------------------------------------------
    def train(self,train_input, train_output, valid_input, valid_output, batch_size, epochs):
        print("fitting model...")
        self.model.fit(train_input, train_output,
                        validation_data=(valid_input, valid_output),
                        batch_size=batch_size, epochs=epochs)

    #---------------------
    def print_results(self, valid_input, valid_output):
        preds = self.model.predict(valid_input, batch_size=32)
        classpreds = preds[0]
        pospreds = preds[1]
        valid_output_class = valid_output[0]
        valid_output_xyr   = valid_output[1]
        for i,cp in enumerate(classpreds):
            pp = pospreds[i]
            tstr = 'class: %s pred: %s center: %.1f %.1f pred: %.1f %.1f' \
            %  ('b' if valid_output_class[i][1] else 'w',
                'b' if cp[1]>cp[0] else 'w',
                valid_output_xyr[i][0], valid_output_xyr[i][1],
                pp[0], pp[1])
            print(tstr)

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--rate", required=False, default=0, type=float)
    args = parser.parse_args()
    model = MapModel(RESOLUTION, GRIDSIZE, args.rate)
    images = ut.get_data(SCRIPTPATH, (RESOLUTION,RESOLUTION))
    output_class = ut.get_output_by_key(SCRIPTPATH,'class')
    output_xyr   = ut.get_output_by_key(SCRIPTPATH,'xyr')
    # Normalize training and validation data by train data mean and std
    means,stds = ut.get_means_and_stds(images['train_data'])
    ut.normalize(images['train_data'],means,stds)
    ut.normalize(images['valid_data'],means,stds)

    train_output_class = ut.onehot(output_class['train_output'])
    train_output_xyr   = output_xyr['train_output']
    valid_output_class = ut.onehot(output_class['valid_output'])
    valid_output_xyr   = output_xyr['valid_output']

    # Load the model and train
    if os.path.exists('model.h5'): model.model.load_weights('model.h5')
    #BP()
    model.train(images['train_data'], [train_output_class, train_output_xyr],
               images['valid_data'], [valid_output_class, valid_output_xyr],
               BATCH_SIZE, args.epochs)
    # model.model.fit(images['train_data'], train_output_xyr,
    #                 batch_size=BATCH_SIZE, epochs=args.epochs,
    #                 validation_data=(images['valid_data'], valid_output_xyr))
    model.model.save_weights('model.h5')
    model.print_results(images['valid_data'], [valid_output_class, valid_output_xyr])

    # preds = model.model.predict(images['valid_data'], batch_size=BATCH_SIZE)
    # classpreds = preds[0]
    # pospreds = preds[1]
    # for i,cp in enumerate(classpreds):
    #     pp = pospreds[i]
    #     tstr = 'class: %s pred: %s center: %.1f %.1f pred: %.1f %.1f' \
    #     %  ('b' if output_class['valid_output'][i] else 'w',
    #         'b' if cp[1]>cp[0] else 'w',
    #         valid_output_xyr[i][0], valid_output_xyr[i][1],
    #         pp[0], pp[1])
    #     print(tstr)

if __name__ == '__main__':
    main()
