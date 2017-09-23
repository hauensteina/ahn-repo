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
import keras.models as km
import keras.optimizers as kopt
import keras.activations as ka
import keras.backend as K
import theano as th

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

BATCH_SIZE=32
GRIDSIZE=0
RESOLUTION=0
MODELFILE='model.h5'
WEIGHTSFILE='weights.h5'
EMPTY=0
WHITE=1
BLACK=2
NCOLORS=3

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Build and train gcount model
    Synopsis:
      %s --epochs <n> --rate <learning_rate>
      or
      %s --visualize
    Description:
      Train a model to classify gridpoints as Empty, Black, White
    Example:
      %s --epochs 10 --rate 0.001
    ''' % (name,name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-----------------
class GCountModel:
    #----------------------------------------------
    def __init__(self,resolution,gridsize,rate=0):
        self.resolution = resolution
        self.gridsize = gridsize
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        # VGG style convolutional model
        inputs = kl.Input(shape=(1,self.resolution,self.resolution))
        x = kl.Conv2D(32,(3,3), activation='relu', padding='same', name='one_a')(inputs)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)
        x = kl.Conv2D(64,(3,3), activation='relu', padding='same', name='one_b')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)

        x = kl.Conv2D(128,(3,3), activation='relu', padding='same', name='two_a')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(64,(1,1), activation='relu', padding='same', name='two_b')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(128,(3,3), activation='relu', padding='same', name='two_c')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)

        x = kl.Conv2D(256,(3,3), activation='relu', padding='same', name='three_a')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(128,(1,1), activation='relu', padding='same', name='three_b')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(256,(3,3), activation='relu', padding='same', name='three_c')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)
        # Get down to three channels e,b,w. Softmax across channels such that c0+c1+c2 = 1.
        x_class_conv = kl.Conv2D(3,(1,1), activation=ut.softMaxAxis1, padding='same',name='lastconv')(x)

        # Split into channels, have a dense layer based on each
        channel0 = kl.Lambda(lambda x: x[:,0,:,:], output_shape=(1,GRIDSIZE,GRIDSIZE) )(x_class_conv)
        channel0_flat = kl.Flatten()(channel0)
        channel0_dense = kl.Dense(1, name='empty_count')(channel0_flat)

        channel1 = kl.Lambda(lambda x: x[:,1,:,:], output_shape=(1,GRIDSIZE,GRIDSIZE) )(x_class_conv)
        channel1_flat = kl.Flatten()(channel1)
        channel1_dense = kl.Dense(1, name='white_count')(channel1_flat)

        channel2 = kl.Lambda(lambda x: x[:,2,:,:], output_shape=(1,GRIDSIZE,GRIDSIZE) )(x_class_conv)
        channel2_flat = kl.Flatten()(channel2)
        channel2_dense = kl.Dense(1, name='black_count')(channel2_flat)

        # flatten into chan0,chan0,..,chan0,chan1,chan1,...,chan1,chan2,chan2,...chan2
        #x_flat = kl.Flatten(name='out')(x_class_conv)
        #x_out = kl.Dense(3, name='ewb_count')(x_flat)
        x_out = kl.concatenate([channel0_dense, channel1_dense, channel2_dense])

        # #x_conv_empty = kl.Conv2D(1,(1,1), activation=ut.softMaxAxis1, padding='same',name='x_conv_empty')(x)
        # x_conv_empty = kl.Conv2D(1,(1,1), padding='same',name='x_conv_empty')(x)
        # x_conv_white = kl.Conv2D(1,(1,1), padding='same',name='x_conv_white')(x)
        # x_conv_black = kl.Conv2D(1,(1,1), padding='same',name='x_conv_black')(x)
        # #x_count_e  = kl.Dense(2,activation='softmax', name='class')(x_flat0)
        # x_flat_conv_empty = kl.Flatten()(x_conv_empty)
        # x_flat_conv_white = kl.Flatten()(x_conv_white)
        # x_flat_conv_black = kl.Flatten()(x_conv_black)
        # x_out = kl.concatenate([x_flat_conv_empty, x_flat_conv_white, x_flat_conv_black])

        # flatten into chan0,chan0,..,chan0,chan1,chan1,...,chan1,chan2,chan2,...chan2
        # x_out = kl.Flatten(name='out')(x_class_conv)


        # Various templates
        #--------------------
        #channels_0_0 = kl.Lambda(lambda x: x[:,:,0,0], (x_class_conv._keras_shape[1],)) (x_class_conv)
        #channels_0_1 = kl.Lambda(lambda x: x[:,:,0,1], (x_class_conv._keras_shape[1],)) (x_class_conv)
        #out_0_0 = kl.Activation('softmax')(channels_0_0)
        #out_0_1 = kl.Activation('softmax')(channels_0_1)
        #out = channels_at_xy(0,0,x_class_conv.shape)(x_class_conv)
        #x_class_conv = kl.MaxPooling2D()(x)
        # Split into GRIDSIZE x GRIDSIZE groups of three
        # Conv layer used for classification
        #x_class_conv = kl.Conv2D(3,(3,3),name='classconv')(x)
        #x_class_pool = kl.GlobalAveragePooling2D()(x_class_conv)
        #x_class = kl.Activation('softmax', name='class')(x_class_pool)
        #x_class  = kl.Dense(2,activation='softmax', name='class')(x_flat0)


        self.model = km.Model(inputs=inputs, outputs=x_out)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam(self.rate)
        else:
            opt = kopt.Adam()
        self.model.compile(loss='mse', optimizer=opt,
                           metrics=[ut.element_match])

    #------------------------------------------------------------------------------------------
    def train(self,train_input, train_output, valid_input, valid_output, batch_size, epochs):
        print("Fitting model...")
        self.model.fit(train_input, train_output,
                        validation_data=(valid_input, valid_output),
                        batch_size=batch_size, epochs=epochs)

    #---------------------------------------------------
    def print_results(self, valid_input, valid_output_0, valid_output_1):
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        testpred = self.model.predict(valid_input[:1])
        BP()
        preds = self.model.predict(valid_input, batch_size=32)
        for i in range(len(preds[0])):
            tstr = 'color0: %s pred: %s || color1: %s pred: %s ' \
            %  (str(valid_output_0[i]), str(preds[0][i]),
                str(valid_output_1[i]), str(preds[1][i]))
            print(tstr)


# Dump jpegs of model conv layer channels to file
#---------------------------------------------------------------------
def visualize_channels(model, layer_name, channels, data, fname):
    # Run model up tp requested layer
    channel_data = ut.get_output_of_layer(model, layer_name, data)[0]

    plt.figure()
    nplots = len(channels) + 2
    ncols = 8
    nrows = nplots // ncols
    if nplots % ncols: nrows += 1

    # Show input image
    orig = data[0][0].astype(np.uint8)
    plt.subplot(nrows,ncols,1)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(orig,cmap='Greys')

    # Show channels individually
    for idx,channel in enumerate(channels):
        data = channel_data[channel]
        img  = scipy.misc.imresize(data, (RESOLUTION,RESOLUTION), interp='nearest')
        plt.subplot(nrows,ncols,idx+2)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #plt.imshow(img, cmap='cool', alpha=1.0)
        plt.imshow(img, cmap='Greys', alpha=1.0)
    # Show winning channel
    rows = channel_data[0].shape[0]
    cols = channel_data[0].shape[1]
    mychans_flat = [np.reshape(channel_data[c],(rows*cols,)) for c in channels]
    winner = np.argmax(mychans_flat,axis=0).reshape(rows,cols)
    img  = scipy.misc.imresize(winner, (RESOLUTION,RESOLUTION), interp='nearest')
    plt.subplot(nrows,ncols,nplots)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.imshow(img, cmap='cool', alpha=1.0)
    plt.imshow(img, cmap='Greys', alpha=1.0)

    plt.savefig(fname)

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    global GRIDSIZE, RESOLUTION
    RESOLUTION = GRIDSIZE * 2 * 2 * 2

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--gridsize", required=True, type=int)
    parser.add_argument("--epochs", required=False, default=10, type=int)
    parser.add_argument("--rate", required=False, default=0, type=float)
    parser.add_argument("--visualize", required=False, action='store_true')
    args = parser.parse_args()
    GRIDSIZE = args.gridsize
    RESOLUTION = GRIDSIZE * 2*2*2*2
    model = GCountModel(RESOLUTION, GRIDSIZE, args.rate)
    if args.visualize or not args.epochs:
        if os.path.exists(WEIGHTSFILE):
            print('Loading weights from file %s...' % WEIGHTSFILE)
            model.model.load_weights(WEIGHTSFILE)
    else:
        if os.path.exists(MODELFILE):
            print('Loading model from file %s...' % MODELFILE)
            model.model = km.load_model(MODELFILE, custom_objects={"th": th})
            if args.rate:
                model.model.optimizer.lr.set_value(args.rate)

    print('Reading data...')
    images = ut.get_data(SCRIPTPATH, (RESOLUTION,RESOLUTION))
    output = ut.get_output_by_key(SCRIPTPATH,'stones')

    #-----------------------------------------------------------
    # Reshape targets to look like the flattened network output
    tt = output['valid_output']
    valid_output = np.array([ [x.tolist().count(EMPTY), x.tolist().count(WHITE), x.tolist().count(BLACK)] for x in tt])
    tt = output['train_output']
    train_output = np.array([ [x.tolist().count(EMPTY), x.tolist().count(WHITE), x.tolist().count(BLACK)] for x in tt])

    means,stds = ut.get_means_and_stds(images['train_data'])
    ut.normalize(images['train_data'],means,stds)
    ut.normalize(images['valid_data'],means,stds)

    # Visualization
    #-----------------
    if args.visualize:
        print('Dumping conv layer images to jpg')
        visualize_channels(model.model, 'lastconv', range(0,3), images['train_data'][0:1], 'lastconv.jpg')
        exit(0)

    # If no epochs, just print output and what it should have been
    if not args.epochs:
        idx=0
        xx = ut.get_output_of_layer(model.model, 'out', images['train_data'][idx:idx+1])
        print(xx)
        print(train_output[idx:idx+1])
        BP()

    # Train
    if args.epochs:
        print('Start training...')
        model.train(images['train_data'], train_output,
                   images['valid_data'],  valid_output,
                   BATCH_SIZE, args.epochs)
        model.model.save_weights(WEIGHTSFILE)
        model.model.save(MODELFILE)

if __name__ == '__main__':
    main()
