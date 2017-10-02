#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Sep 29, 2017
# **********************************************************************/
#
# Build and train maxlambda model
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

BATCH_SIZE=4
GRIDSIZE=0
RESOLUTION=0
MODELFILE='model.h5'
WEIGHTSFILE='weights.h5'
CONV_WEIGHTSFILE='conv_weights.h5' # Save weights for just the convolutional layers
INIT_WEIGHTSFILE='init_weights.h5' # Init weights for just the convolutional layers
EMPTY=0
WHITE=1
BLACK=2
NCOLORS=3

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Build and train maxlambda model
    Synopsis:
      %s --epochs <n> --rate <learning_rate>
      or
      %s --visualize
    Description:
      Train a model to count Black and White stones without maxpooling.
    Example:
      %s --epochs 10 --rate 0.001
    ''' % (name,name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-----------------
class MaxLambdaModel:
    #----------------------------------------------
    def __init__(self,resolution,gridsize,batch_size,rate=0):
        self.resolution = resolution
        self.gridsize = gridsize
        self.batch_size = batch_size
        self.rate = rate
        self.build_model()

    @staticmethod
    #-----------------------------------
    def outshape_of_lambda(input_shape):
        return (input_shape[0], 1, input_shape[2], input_shape[3])

    #-----------------------
    def build_model(self):
        inputs = kl.Input(shape=(1,self.resolution,self.resolution))
        x = kl.Conv2D(32,(3,3), activation='relu', padding='same', name='one_a')(inputs)
        x = kl.BatchNormalization(axis=1, name='batch_one_a')(x)
        x = kl.MaxPooling2D()(x)
        x = kl.Conv2D(64,(3,3), activation='relu', padding='same', name='one_b')(x)
        x = kl.BatchNormalization(axis=1, name='batch_one_b')(x)
        x = kl.MaxPooling2D()(x)

        x = kl.Conv2D(128,(3,3), activation='relu', padding='same', name='two_a')(x)
        x = kl.BatchNormalization(axis=1, name='batch_two_a')(x)
        x = kl.Conv2D(64,(1,1), activation='relu', padding='same', name='two_b')(x)
        x = kl.BatchNormalization(axis=1, name='batch_two_b')(x)
        x = kl.Conv2D(128,(3,3), activation='relu', padding='same', name='two_c')(x)
        x = kl.BatchNormalization(axis=1, name='batch_two_c')(x)
        x = kl.MaxPooling2D()(x)

        x = kl.Conv2D(256,(3,3), activation='relu', padding='same', name='three_a')(x)
        x = kl.Conv2D(128,(1,1), activation='relu', padding='same', name='three_b')(x)
        x = kl.Conv2D(256,(3,3), activation='relu', padding='same', name='three_c')(x)
        #x = kl.MaxPooling2D()(x)

        # Get down to two channels e,b,w. Softmax across channels such that c0+c1+c2 = 1.
        x_class_conv = kl.Conv2D(3,(1,1), activation=ut.softMaxAxis1, padding='same',name='lastconv')(x)

        #chan_e  = kl.Lambda(lambda x: x[:,0,:,:], output_shape=(self.gridsize,self.gridsize), name='channel_e') (x_class_conv)
        #chan_e_flat = kl.Flatten(name='chan_e_flat')(chan_e)
        chan_w  = kl.Lambda(lambda x: x[:,1,:,:], output_shape=MaxLambdaModel.outshape_of_lambda, name='channel_w') (x_class_conv)
        chan_w_flat = kl.Flatten(name='chan_w_flat')(chan_w)
        chan_b  = kl.Lambda(lambda x: x[:,2,:,:], output_shape=MaxLambdaModel.outshape_of_lambda, name='channel_b') (x_class_conv)
        chan_b_flat = kl.Flatten(name='chan_b_flat')(chan_b)

        # x.shape[0] is the batch size
        # K.int_shape(x)[1] == GRIDSIZE * GRIDSIZE
        # count_e = kl.Lambda(lambda x: K.dot(x,K.ones(K.int_shape(x)[1])).reshape((x.shape[0],1)),
        #                     output_shape=((1,)),
        #                     name='count_e') (chan_e_flat)
        count_w = kl.Lambda(lambda x: K.dot(x,K.ones(K.int_shape(x)[1])).reshape((x.shape[0],1)),
                            output_shape=((1,)),
                            name='count_w') (chan_w_flat)
        count_b = kl.Lambda(lambda x: K.dot(x,K.ones(K.int_shape(x)[1])).reshape((x.shape[0],1)),
                            output_shape=((1,)),
                            name='count_b') (chan_b_flat)

        x_out = kl.concatenate([count_w,count_b])
        self.model = km.Model(inputs=inputs, outputs=x_out)
        self.model.summary()

        self.compile()

    #---------------------------
    def compile(self):
        if self.rate > 0:
            opt = kopt.Adam(self.rate)
        else:
            opt = kopt.Adam()
        self.model.compile(loss='mse', optimizer=opt,
                           metrics=[ut.element_match])

    # Get names of convolutional layers
    #------------------------------------
    def get_conv_layers(self):
        res = []
        conv_over = False
        for layer in self.model.layers:
            if not conv_over: res.append(layer.name)
            if layer.name == 'lastconv':
                conv_over = True
        return res

    # Save weights for convolutional layers only.
    # Do this by renaming the top layers, which we do not want to reload.
    #-----------------------------------------------------------------------
    def save_conv_weights(self,fname):
        conv_layers = self.get_conv_layers()
        for layer in self.model.layers:
            if not layer.name in conv_layers:
                layer.name = 'x_' + layer.name
        self.model.save_weights(fname)

    # Load weights for layers with matching name.
    # This allows us to extract convolutional layers from a smaller reolution model
    #---------------------------------------------------------------------------------
    def load_weights_by_name(self,fname,freeze=False):
        if not os.path.exists(fname):
            return
        print('load_weights_by_name(%s)' % fname)
        self.model.load_weights(fname, by_name=True)
        if freeze:
            conv_layers = self.get_conv_layers()
            for layer in self.model.layers:
                if layer.name in conv_layers:
                    layer.trainable = False
            self.compile()

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

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument("--gridsize", required=True, type=int)
    parser.add_argument("--epochs", required=False, default=10, type=int)
    parser.add_argument("--rate", required=False, default=0, type=float)
    parser.add_argument("--visualize", required=False, action='store_true')
    args = parser.parse_args()
    GRIDSIZE = args.gridsize
    RESOLUTION = GRIDSIZE * 2*2*2*2
    # Build Model
    model = MaxLambdaModel(RESOLUTION, GRIDSIZE, BATCH_SIZE, args.rate)
    if args.visualize or not args.epochs:
        if os.path.exists(WEIGHTSFILE):
            print('Loading weights from file %s...' % WEIGHTSFILE)
            model.model.load_weights(WEIGHTSFILE)
    else:
        # weight initialization for conv layers
        #model.load_weights_by_name(INIT_WEIGHTSFILE,True) # freeze conv weights
        model.load_weights_by_name(INIT_WEIGHTSFILE)
        if os.path.exists(MODELFILE):
            print('Loading model from file %s...' % MODELFILE)
            #model.model = km.load_model(MODELFILE, custom_objects={"th": th})
            model.model = km.load_model(MODELFILE)
            if args.rate:
                model.model.optimizer.lr.set_value(args.rate)

    print('Reading data...')
    images = ut.get_data(SCRIPTPATH, (RESOLUTION,RESOLUTION))
    output = ut.get_output_by_key(SCRIPTPATH,'stones')

    #-----------------------------------------------------------
    # Reshape targets to look like the flattened network output
    tt = output['valid_output']
    valid_output = np.array([ [x.tolist().count(WHITE), x.tolist().count(BLACK)] for x in tt])
    tt = output['train_output']
    train_output = np.array([ [x.tolist().count(WHITE), x.tolist().count(BLACK)] for x in tt])

    means,stds = ut.get_means_and_stds(images['train_data'])
    ut.normalize(images['train_data'],means,stds)
    ut.normalize(images['valid_data'],means,stds)

    #-----------------
    # Visualization
    if args.visualize:
        print('Dumping conv layer images to jpg')
        visualize_channels(model.model, 'lastconv', range(0,3), images['train_data'][700:701], 'lastconv0.jpg')
        visualize_channels(model.model, 'lastconv', range(0,3), images['train_data'][500:501], 'lastconv1.jpg')
        visualize_channels(model.model, 'lastconv', range(0,3), images['train_data'][400:401], 'lastconv2.jpg')
        visualize_channels(model.model, 'lastconv', range(0,3), images['train_data'][300:301], 'lastconv3.jpg')
        visualize_channels(model.model, 'lastconv', range(0,3), images['train_data'][200:201], 'lastconv4.jpg')
        exit(0)

    # If no epochs, just print output and what it should have been
    if not args.epochs:
        idx=0
        print ('lastconv')
        xx = ut.get_output_of_layer(model.model, 'lastconv', images['train_data'][idx:idx+1])
        print(xx)
        print ('count_e')
        xx = ut.get_output_of_layer(model.model, 'count_e', images['train_data'][idx:idx+1])
        print(xx)
        print ('count_w')
        xx = ut.get_output_of_layer(model.model, 'count_w', images['train_data'][idx:idx+1])
        print(xx)
        print ('count_b')
        xx = ut.get_output_of_layer(model.model, 'count_b', images['train_data'][idx:idx+1])
        print(xx)
        print ('out')
        xx = model.model.predict(images['train_data'][idx:idx+1],batch_size=1)
        print(xx)
        print ('target')
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
        model.save_conv_weights(CONV_WEIGHTSFILE)

if __name__ == '__main__':
    main()
