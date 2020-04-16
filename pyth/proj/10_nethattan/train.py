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
      %s: Train a DNN model on shifting puzzle positions from folder generator.out
    Synopsis:
      %s --puzzlesize <int> --batchsize <int> --loadsize <int>
    Description:
      --puzzlesize: Side length of the puzzle. A 15-puzzle has puzzlesize 4.
      --loadsize: How many batches to suck into memory at a time.
                  The model is saved when we are done with them.
      --batchsize: How many examples in a batch
      --mode: d means train on distance, v means train on v_from_dist in (0,1)
      --epochs: How many epochs to train
    Example:
      %s --puzzlesize 3 --batchsize 100 --loadsize 1000 --mode d --epochs 100
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

VALDIR = 'validation_data'

def main():
    TRAINDIR = 'training_data'
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--puzzlesize', required=True, type=int)
    parser.add_argument( '--batchsize', required=True, type=int)
    parser.add_argument( '--epochs', required=True, type=int)
    parser.add_argument( '--mode', required=True, choices=['v','d'])
    args = parser.parse_args()

    MODELFNAME = 'net_d.hd5'
    if args.mode == 'v': MODELFNAME = 'net_v.hd5'

    model = ShiftModel( args.puzzlesize, args.mode)
    if os.path.exists( MODELFNAME):
        print( 'Loading model from %s' % MODELFNAME)
        model.load( MODELFNAME)
        test_model( model, args.puzzlesize, args.mode); exit(1)

    valid_inputs, valid_targets = load_folder_samples( VALDIR, args.puzzlesize, args.mode)
    #train_generator = FitGenerator( TRAINDIR, args.batchsize, args.puzzlesize, args.mode)

    # checkpoint
    filepath='model-improvement-{epoch:02d}-{val_loss:e}.hd5'
    checkpoint = ModelCheckpoint( filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    #batches_per_epoch = train_generator.total_samples() // args.batchsize

    print( 'Loading validation data')
    valid_inputs, valid_targets, _ = load_folder_samples( VALDIR, args.puzzlesize, args.mode)
    print( 'Loaded %d validation samples' % len(valid_inputs))
    print( 'Loading trainig data')
    train_inputs, train_targets, _ = load_folder_samples( TRAINDIR, args.puzzlesize, args.mode)
    print( 'Loaded %d training samples' % len(train_inputs))
    print( 'Kicking off...')
    model.model.fit( x=train_inputs,
                     y=train_targets,
                     #train_generator.generate(),
                     #steps_per_epoch=batches_per_epoch, # For generator only
                     epochs=args.epochs,
                     validation_data = (valid_inputs, valid_targets),
                     callbacks=callbacks_list)
    model.save( MODELFNAME)

def load_random_samples( folder, n_files, puzzlesize, mode):
    ' Load random n_files into memory, split into inputs and targets'
    files = glob.glob( '%s/*.json' % folder)
    files = np.random.choice( files, n_files) # sample with replacement
    inputs = []
    targets = []
    for fname in files:
        try:
            with open( fname) as f:
                jsn = json.load(f)
        except Exception as e:
            print( 'load_random_samples(): Could not load %s. Skipping.' % fname)
        state = State.from_list( puzzlesize, jsn['state']['arr'])
        inp = state.encode()
        #target = (jsn['v'], np.array(jsn['p']))
        inputs.append( inp)
        if mode == 'v':
            targets.append( 2.0 * jsn['v'] - 1.0) # Map to (-1,1) for tanh
        else:
            targets.append( jsn['dist']) # Manhattan distance

    return np.array( inputs), np.array( targets)

def test_model( model, puzzlesize, mode, metric_func=None):
    ' Test model quality on validation data with our own metrics. '
    valid_inputs, valid_targets, valid_json = load_folder_samples( VALDIR, puzzlesize, mode, return_json=True)
    print( 'Loaded %d validation samples' % len(valid_inputs))
    loss,metric = model.model.evaluate( valid_inputs, valid_targets)
    preds = model.predict( valid_inputs)
    # Our own metrics
    avg_sqerr = 0.0; n_corr_samples = 0;
    for idx,jsn in enumerate(valid_json):
        #state = State.from_list( puzzlesize, jsn['state']['arr'])
        inp = valid_inputs[idx]
        pred = preds[idx][0]
        #BP()
        if mode == 'v':
            corr =  2.0 * jsn['v'] - 1.0 # Map to (-1,1) for tanh
        else:
            corr = jsn['dist']
            if round(pred) == corr: n_corr_samples += 1
        delta = pred - corr
        avg_sqerr += delta * delta

    nsamples = len(valid_json)
    print( 'mse: %.4f pct_corr: %.2f' % (avg_sqerr / nsamples, 100 * n_corr_samples / nsamples ))

def load_folder_samples( folder, puzzlesize, mode, return_json=False):
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
        #target = (jsn['v'], np.array(jsn['p']))
        #inputs.append( inp)
        if mode == 'v':
            targets[idx] = 2.0 * jsn['v'] - 1.0 # Map to (-1,1) for tanh
        else:
            targets[idx] = jsn['dist'] # Manhattan distance

    return inputs, targets, json_list

#=====================
class FitGenerator:
    ' Generate training and validation batches for fit_generator '
    def __init__( self, data_directory, batch_size, puzzlesize, mode):
        self.datadir = data_directory
        self.mode = mode
        self.batch_size = batch_size
        self.puzzlesize = puzzlesize

    #--------------------------
    def total_samples( self):
        files = glob.glob( '%s/*.json' % self.datadir)
        return len(files)

    #-----------------------
    def generate( self):
        while 1:
            inputs,targets = load_random_samples( self.datadir, self.batch_size, self.puzzlesize, self.mode)
            yield inputs, targets


if __name__ == '__main__':
    main()
