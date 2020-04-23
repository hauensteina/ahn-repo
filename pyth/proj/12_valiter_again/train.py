#!/usr/bin/env python

'''
Train a DNN model on shifting puzzle positions from folder generator.out
Python 3
AHN, Apr 2020
'''

from pdb import set_trace as BP
import argparse
import math, os, glob, json
import time
from math import log, exp
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from shiftmodel import ShiftModel
from state import State

def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Train a DNN model on shifting puzzle positions.
          Input files are expected in folders training_data and validation_data.
    Synopsis:
      %s --puzzlesize <int> [--batchsize <int>] [--valid_only]
    Description:
      Train one epoch, save model, repeat.
      A separate process running generator.py picks up the latest model and generates data.
      --puzzlesize: Side length of the puzzle. A 15-puzzle has puzzlesize 4.
      --batchsize: How many examples in a batch. Default 32.
      --valid_only: Just load and validate the model, then exit
    Examples:
      %s --puzzlesize 3
      %s --puzzlesize 3 --valid_only
--
    ''' % (name,name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

VALDIR = 'validation_data'
TRAINDIR = 'training_data'
MODELFNAME = 'net_v.hd5'

def main():
    N_EPOCHS = 3
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--puzzlesize', required=True, type=int)
    parser.add_argument( '--epochs', type=int, default=10)
    parser.add_argument( '--batchsize', type=int, default=32)
    parser.add_argument( '--valid_only', action = 'store_true')
    args = parser.parse_args()

    if args.valid_only:
        test_model( model, args.puzzlesize)
        exit(1)

    # If no data, at least give the generator an initialized model.
    if not os.path.exists( TRAINDIR):
        model.save( MODELFNAME)
        print( 'Folder %s not found. Saving model to %s and exiting.' % (TRAINDIR, MODELFNAME))
        exit(1)

    model = ShiftModel( args.puzzlesize)

    while 1: # Hit Ctrl-C to stop
        if os.path.exists( MODELFNAME):
            print( 'Loading model from %s' % MODELFNAME)
            model.load( MODELFNAME)

        valid_inputs, valid_targets, _ = load_folder_samples( VALDIR, args.puzzlesize)

        # checkpoint
        #filepath='model-improvement-{epoch:02d}-{val_loss:e}.hd5'
        filepath = MODELFNAME
        checkpoint = ModelCheckpoint( filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
        callbacks_list = [checkpoint]

        print( 'Loading validation data')
        valid_inputs, valid_targets, _ = load_folder_samples( VALDIR, args.puzzlesize)
        print( 'Loaded %d validation samples' % len(valid_inputs))
        print( 'Loading trainig data')
        train_inputs, train_targets, _ = load_folder_samples( TRAINDIR, args.puzzlesize)
        print( 'Loaded %d training samples' % len(train_inputs))
        print( 'Kicking off...')
        model.model.fit( x = train_inputs,
                         y = train_targets,
                         batch_size = args.batchsize,
                         epochs = N_EPOCHS,
                         validation_data = (valid_inputs, valid_targets),
                         callbacks = callbacks_list)

def test_model( model, puzzlesize, metric_func=None):
    ' Test model quality on validation data with our own metrics. '
    valid_inputs, valid_targets, valid_json = load_folder_samples( VALDIR, puzzlesize, return_json=True)
    print( 'Loaded %d validation samples' % len(valid_inputs))
    loss,metric = model.model.evaluate( valid_inputs, valid_targets)
    preds = model.predict( valid_inputs)
    # Our own metrics
    avg_sqerr = 0.0; n_corr_samples = 0;
    for idx,jsn in enumerate(valid_json):
        inp = valid_inputs[idx]
        pred = preds[idx][0]
        delta = pred - jsn['v']
        corr = jsn['dist']
        est  =  State.dist_from_v( jsn['v'])
        if round(est) == corr: n_corr_samples += 1
        avg_sqerr += delta * delta

    nsamples = len(valid_json)
    print( 'mse: %e pct_corr: %.2f' % (avg_sqerr / nsamples, 100 * n_corr_samples / nsamples ))

def load_folder_samples( folder, puzzlesize, return_json=False):
    ' Load all samples from folder into memory, split into inputs and targets'
    files = glob.glob( '%s/*.json' % folder) # [:1000]
    targets = np.empty( len(files), dtype=float)
    inputs = None
    json_list = []

    for idx,fname in enumerate(files):
        if idx % 10000 == 0:
            print( 'loaded %d/%d samples' % (idx,len(files)))
        try:
            with open( fname) as f:
                jsn = json.load(f)
        except Exception as e:
            print( 'load_folder_samples(): Could not load %s. Skipping.' % fname)
        if return_json: json_list.append( jsn)
        state = State.from_list( puzzlesize, jsn['state']['arr'])
        inp = state.encode()
        if inputs is None:
            inputs = np.zeros( (len(files),) + inp.shape)
        inputs[idx] = inp
        targets[idx] = jsn['v']

    return inputs, targets, json_list

if __name__ == '__main__':
    main()
