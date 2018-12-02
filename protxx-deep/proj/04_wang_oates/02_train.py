#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Dec 1, 2018
# **********************************************************************/
#
# Train a convolutional two class multivariate(x,y,z) time series model.
# Architecture like Wang,Oates:Time Series Classification ...

from __future__ import division, print_function
from pdb import set_trace as BP
import inspect
import os,sys,re,json,shutil
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

BATCH_SIZE=256

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s -- Build and train a convolutional two class multivariate time series model
    Synopsis:
      %s --epochs <n> --rate <rate>
    Description:
      Build a NN model with Keras, train on the data in the train subfolder.
    Example:
      %s --epochs 100 --rate 0
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# A convolutional multivariate time series model
#==================================================
class ConvModel:
    #----------------------------------------
    def __init__(self, input_shape, rate=0):
        self.input_shape = input_shape
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        inputs = kl.Input( shape=self.input_shape)

        x = kl.Conv1D( 128, 3, activation='relu', padding='same', name='a1')(inputs)
        x = kl.BatchNormalization()(x)	
        x = kl.Conv1D( 256, 3, activation='relu', padding='same', name='a2')(x)
        x = kl.BatchNormalization()(x)	
        x = kl.Conv1D( 128, 3, activation='relu', padding='same', name='a3')(x)
        x = kl.BatchNormalization()(x)	

        # Classification block
        x_class_conv = kl.Conv1D( 2, 1, padding='same', name='lastconv')(x)
        x_class_pool = kl.GlobalAveragePooling1D()( x_class_conv)
        output = kl.Activation( 'softmax', name='class')(x_class_pool)

        self.model = km.Model( inputs=inputs, outputs=output)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam( self.rate)
        else:
            opt = kopt.Adam()
        #opt = kopt.SGD(lr=0.01)
        self.model.compile( loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#===================================================================================================

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    LENGTH = 100

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--epochs", required=True, type=int)
    parser.add_argument( "--rate", required=True, type=float)
    args = parser.parse_args()

    train_data, train_classes = ut.read_series( SCRIPTPATH + '/train/all_files', ['x','y','z'])
    valid_data, valid_classes = ut.read_series( SCRIPTPATH + '/valid/all_files', ['x','y','z'])
    # train_data = np.array( [[1,2,3,4,5],
    #                         [10,20,30,40,50]], float)
    # train_classes = ut.onehot( [0,1])

    # valid_data = np.array( [[1,2,3,4,5],
    #                         [10,20,30,40,50]], float)
    # valid_classes = ut.onehot( [0,1])

    model = ConvModel( (train_data.shape[1],train_data.shape[2]), args.rate)

    wfname =  'nn.weights'
    if os.path.exists( wfname):
        model.model.load_weights( wfname)

    model.model.fit( train_data, train_classes,
                     batch_size=BATCH_SIZE,
                     epochs=args.epochs,
                     validation_data=(valid_data, valid_classes))

    # Save weights and model
    if os.path.exists( wfname):
        shutil.move( wfname, wfname + '.bak')
    model.model.save( 'nn_bew.hd5')
    model.model.save_weights( wfname)

    # preds = model.model.predict(images['valid_data'], batch_size=BATCH_SIZE)
    # print(preds)

if __name__ == '__main__':
    main()
