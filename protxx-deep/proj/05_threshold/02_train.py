#!/usr/bin/env python

# /********************************************************************
# Filename: train.py
# Author: AHN
# Creation Date: Dec 1, 2018
# **********************************************************************/
#
# Train a single neuron on a threshold.

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
import keras

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

BATCH_SIZE=32

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s -- Train a single neuron on a threshold
    Synopsis:
      %s --epochs <n> --rate <rate>
    Example:
      %s --epochs 100 --rate 0
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# A single neuron model
#=======================
class SingleModel:
    #----------------------------------------
    def __init__(self, rate=0):
        self.rate = rate
        self.build_model()

    #-----------------------
    def build_model(self):
        inputs = kl.Input( shape=(1,))
        x = kl.Dense( 1, kernel_constraint=keras.constraints.UnitNorm())(inputs)  # very deep learning

        # Classification block
        output = kl.Activation( 'sigmoid', name='class')(x)

        self.model = km.Model( inputs=inputs, outputs=output)
        self.model.summary()
        if self.rate > 0:
            opt = kopt.Adam( self.rate)
        else:
            opt = kopt.Adam()
        #opt = kopt.SGD(lr=0.001)
        self.model.compile( loss='mse', optimizer=opt, metrics=['accuracy'])
#===================================================================================================

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--epochs", required=True, type=int)
    parser.add_argument( "--rate", required=True, type=float)
    args = parser.parse_args()

    train_data, train_classes = ut.read_series( SCRIPTPATH + '/train/all_files', ['x'])
    valid_data, valid_classes = ut.read_series( SCRIPTPATH + '/valid/all_files', ['x'])
    train_data = np.squeeze(train_data) 
    valid_data = np.squeeze(valid_data) 
    train_classes = [ c[1] for c in train_classes ]
    valid_classes = [ c[1] for c in valid_classes ]

    model = SingleModel( args.rate)

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

    print( "Weights:\n")
    print( model.model.get_weights())

    # preds = model.model.predict(images['valid_data'], batch_size=BATCH_SIZE)
    # print(preds)

if __name__ == '__main__':
    main()
