#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Sep 15, 2017
# **********************************************************************/
#
# Build and train lambda model
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

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

BATCH_SIZE=8
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
      %s --  Build and train lambda model
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


# Return a Lambda layer which gives me the channels at position x,y as a vector
#---------------------------------------------------------------------------------
def channels_at_xy(x,y,shape):
    def func(tens):
        res = tens[:,:,x,y]
        #BP()
        return res
    BP()
    return kl.Lambda(func, output_shape=(1,) + shape[1])

# # Custom Softmax along axis 1 (channels).
# # Use as an activation
# #-----------------------------------------
# def softMaxAxis1(x):
#     return ka.softmax(x,axis=1)

# # Make sure we can save and load a model with custom activation
# ka.softMaxAxis1 = softMaxAxis1

# # Custom metric returns 1.0 if all rounded elements
# # in y_pred match y_true, else 0.0 .
# #---------------------------------------------------------
# def bool_match(y_true, y_pred):
#     return K.switch(K.any(y_true-y_pred.round()), K.variable(0), K.variable(1))

#-----------------
class GoogleModel:
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
        x = kl.Conv2D(32,(3,3), activation='relu', padding='same')(inputs)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)
        x = kl.Conv2D(64,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)

        x = kl.Conv2D(128,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(64,(1,1), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(128,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)

        x = kl.Conv2D(256,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(128,(1,1), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.Conv2D(256,(3,3), activation='relu', padding='same')(x)
        x = kl.BatchNormalization(axis=1)(x)
        x = kl.MaxPooling2D()(x)
        # Get down to three channels e,b,w. Softmax across channels such that c0+c1+c2 = 1.
        x_class_conv = kl.Conv2D(3,(1,1), activation=ut.softMaxAxis1, padding='same',name='lastconv')(x)
        # flatten into chan0,chan0,..,chan0,chan1,chan1,...,chan1,chan2,chan2,...chan2
        x_out = kl.Flatten(name='out')(x_class_conv)
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
        self.model.compile(loss='mean_squared_error', optimizer=opt,
                           metrics=[ut.bool_match,ut.bitwise_match])


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

# Dump the top conv layer results for black and white to viz_<n>.jpg
# for the the first couple images in data
#---------------------------------------------------------------------
def visualize(model, layer_name, data, filenames):
    BLACK=1
    WHITE=0
    intermediate_layer_model = km.Model(inputs=model.model.input,
                                        outputs=model.model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(data)

    for img_num in range(5):
        conv_w = intermediate_output[img_num][WHITE]
        img_w = scipy.misc.imresize(conv_w, (80,80), interp='nearest')
        conv_b = intermediate_output[img_num][BLACK]
        img_b = scipy.misc.imresize(conv_b, (80,80), interp='nearest')

        plt.figure()
        plt.subplot(141)
        orig = data[img_num][0].astype(np.uint8)
        # Input image
        plt.imshow(orig,cmap='Greys')
        plt.subplot(142)
        # White convolution layer
        plt.imshow(img_w, cmap='Greys', alpha=1.0)
        plt.subplot(143)
        # Black convolution layer
        plt.imshow(img_b, cmap='Greys', alpha=1.0)
        plt.subplot(144)
        # Overlay black conv layer on original
        plt.imshow(orig, cmap='Greys', alpha=1.0)
        plt.imshow(img_b, cmap='cool', alpha=0.5)
        # Save
        plt.savefig('viz_%d.jpg' % img_num)


#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    global GRIDSIZE, RESOLUTION
    RESOLUTION = GRIDSIZE * 2 * 2 * 2

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--gridsize", required=True, type=int)
    parser.add_argument("--epochs", required=False, default=10, type=int)
    parser.add_argument("--rate", required=False, default=0, type=float)
    parser.add_argument("--visualize", required=False, action='store_true')
    args = parser.parse_args()
    GRIDSIZE = args.gridsize
    RESOLUTION = GRIDSIZE * 2*2*2*2
    model = GoogleModel(RESOLUTION, GRIDSIZE, args.rate)
    if args.visualize or not args.epochs:
        if os.path.exists(WEIGHTSFILE):
            print('Loading weights from file %s...' % WEIGHTSFILE)
            model.model.load_weights(WEIGHTSFILE)
    else:
        if os.path.exists(MODELFILE):
            print('Loading model from file %s...' % MODELFILE)
            model.model = km.load_model(MODELFILE)
            if args.rate:
                model.model.optimizer.lr.set_value(args.rate)

    print('Reading data...')
    images = ut.get_data(SCRIPTPATH, (RESOLUTION,RESOLUTION))
    output = ut.get_output_by_key(SCRIPTPATH,'stones')

    # Debug
    #----------------------------------------------------
    #last_conv_model = km.Model(inputs=model.model.input,
    #                        outputs=model.model.get_layer('lastconv').output)
    #tt = last_conv_model.predict(images['valid_data'][:1])
    #xx = model.model.predict(images['valid_data'][:1])
    #BP()

    #-----------------------------------------------------------
    # Reshape targets to look like the flattened network output
    tt = output['valid_output']
    valid_output = np.array([np.transpose(ut.onehot(x,NCOLORS)).reshape(GRIDSIZE*GRIDSIZE*3) for x in tt])
    tt = output['train_output']
    train_output = np.array([np.transpose(ut.onehot(x,NCOLORS)).reshape(GRIDSIZE*GRIDSIZE*3) for x in tt])

    means,stds = ut.get_means_and_stds(images['train_data'])
    ut.normalize(images['train_data'],means,stds)
    ut.normalize(images['valid_data'],means,stds)

    fname = output['train_filenames'][0]
    #tt = get_output_of_layer(model.model, 'lastconv', images['train_data'][:1])
    if not args.epochs:
        idx=0
        xx = ut.get_output_of_layer(model.model, 'out', images['train_data'][idx:idx+1])
        print(xx)
        print(train_output[idx:idx+1])
        BP()

    if args.visualize:
        print('Dumping conv layer images to jpg')
        visualize(model, 'classconv', images['train_data'], ['train/' + x for x in meta['train_filenames']])
        exit(0)

    # Train
    if args.epochs:
        print('Start training...')
        model.train(images['train_data'], train_output,
                   images['valid_data'],  valid_output,
                   BATCH_SIZE, args.epochs)
        model.model.save_weights(WEIGHTSFILE)
        model.model.save(MODELFILE)
    # model.print_results(images['valid_data'], valid_output_0, valid_output_1)

if __name__ == '__main__':
    main()
