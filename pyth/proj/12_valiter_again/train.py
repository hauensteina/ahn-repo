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
import multiprocessing as mp
import shutil
from distutils.dir_util import copy_tree, remove_tree, mkpath

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

from shiftmodel import ShiftModel
from state import State


# Limit GPU memory usage to 5GB
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5*1024)])

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
GENDIR = 'generated_data'
ARCHDIR = 'train_data_archive'
MODELFNAME = 'net_v.hd5'

def main():
    MAX_EPOCHS = 1000
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( '--puzzlesize', required=True, type=int)
    parser.add_argument( '--batchsize', type=int, default=32)
    parser.add_argument( '--valid_only', action = 'store_true')
    args = parser.parse_args()

    if args.valid_only:
        model = ShiftModel( args.puzzlesize)
        test_model( model, args.puzzlesize)
        exit(1)

    # If no data, at least give the generator an initialized model.
    if not os.path.exists( TRAINDIR):
        model = ShiftModel( args.puzzlesize)
        model.save( MODELFNAME)
        print( 'Folder %s not found. Saving model to %s and exiting.' % (TRAINDIR, MODELFNAME))
        exit(1)

    # Restart training process whenever new files are available
    while(1):
        print( 'Restarting training on new data')
        ctx = mp.get_context('spawn')
        p = ctx.Process( target=train, args=(args.puzzlesize, args.batchsize, MAX_EPOCHS))
        p.start()
        p.join()

def train( puzzlesize, batchsize, max_epochs):
    ' Function to execute in a separate process and train until a new set of training files is available '
    model = ShiftModel( puzzlesize)
    # checkpoint
    filepath1='model-improvement-{epoch:02d}-{val_loss:e}.hd5'
    filepath2 = MODELFNAME
    checkpoint1 = ModelCheckpoint( filepath1, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
    checkpoint2 = ModelCheckpoint( filepath2, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
    callbacks_list = [checkpoint1, checkpoint2, stop_if_new_files()]

    if os.path.exists( MODELFNAME):
        print( 'Loading model from %s' % MODELFNAME)
        model.load( MODELFNAME)

    print( 'Loading validation data')
    valid_inputs, valid_targets, _ = load_folder_samples( VALDIR, puzzlesize)
    print( 'Loaded %d validation samples' % len(valid_inputs))
    print( 'Loading trainig data')
    train_inputs, train_targets, _ = load_folder_samples( TRAINDIR, puzzlesize)
    print( 'Loaded %d training samples' % len(train_inputs))
    print( 'Kicking off...')
    model.model.fit( x = train_inputs,
                     y = train_targets,
                     batch_size = batchsize,
                     epochs = max_epochs,
                     validation_data = (valid_inputs, valid_targets),
                     callbacks = callbacks_list)
    #model.save( MODELFNAME)

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

def load_folder_samples( folder, puzzlesize, hardest_percentile=1.0, return_json=False):
    ' Load all samples from folder into memory, split into inputs and targets'
    files = glob.glob( '%s/*.json' % folder) # [:1000]
    files = sorted( files)
    files = files[int(-1*len(files)*hardest_percentile):]
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

#==================================================
class stop_if_new_files( Callback):
    '''
    Stop training if enough new files are available for the next cycle
    '''
    def __init__( self, files_needed=2000):
        #super(keras.callbacks.Callback, self).__init__()
        super( stop_if_new_files, self).__init__()
        self.files_needed = files_needed

    def on_epoch_end( self, epoch, logs={}):
        files = glob.glob( "%s/*.json" % GENDIR)
        if len(files) >= self.files_needed:
            archfolder = ARCHDIR + '/' + '%04d' % random.randint( 1, 1000)
            print( 'Moving old training files to %s' % archfolder)
            copy_tree( TRAINDIR, archfolder)
            remove_tree( TRAINDIR)
            mkpath( TRAINDIR)
            print( 'Moving %d new files to %s' % (len(files), TRAINDIR))
            for f in files: shutil.move( f, TRAINDIR)
            self.model.stop_training = True

if __name__ == '__main__':
    main()
