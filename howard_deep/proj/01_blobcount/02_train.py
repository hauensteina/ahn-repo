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
#import matplotlib as mpl
#mpl.use('Agg') # This makes matplotlib work without a display
#from matplotlib import pyplot as plt
import keras.layers as kl
import keras.models as km

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Build and train blobcount model
    Synopsis:
      %s --res <n>
    Description:
      Build a NN model with Keras, train on the data in the train subfolder.
    Example:
      %s
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-------------------
def build_model(res):
    inputs = kl.Input(shape=(res,res))
    x = kl.Flatten()(inputs)
    x = kl.Dense(64, activation='relu')(x)
    x = kl.Dense(64, activation='relu')(x)
    predictions = kl.Dense(10, activation='softmax')(x)
    #print(inspect.getargspec(km.Model.__init__))
    model = km.Model(input=inputs, output=predictions)

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--res", required=True, type=int)
    args = parser.parse_args()
    #np.random.seed(0) # Make things reproducible
    model = build_model(args.res)


if __name__ == '__main__':
    main()
