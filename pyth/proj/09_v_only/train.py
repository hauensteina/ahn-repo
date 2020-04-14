#!/usr/bin/env python

'''
Train a DNN model with policy and value head to solve a shifting puzzle.
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

from player import Player
from shiftmodel import ShiftModel
from state import State

def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Train a DNN model with policy and value head to solve a shifting puzzle.
          Keeps going forever and periodically saves weights.
    Synopsis:
      %s --puzzlesize <int> --batchsize <int> --loadsize <int>
    Description:
      --puzzlesize: Side length of the puzzle. A 15-puzzle has puzzlesize 4.
      --loadsize: How many batches to suck into memory at a time.
                  The model is saved when we are done with them.
      --batchsize: How many examples in a batch
    Example:
      %s --puzzlesize 3 --batchsize 100 --loadsize 10
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    WEIGHTSFNAME = 'generator.h5'
    GENDIR = 'generator.out'
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--puzzlesize", required=True, type=int)
    parser.add_argument( "--batchsize", required=True, type=int)
    parser.add_argument( "--loadsize", required=True, type=int)
    args = parser.parse_args()

    model = ShiftModel( args.puzzlesize)
    if os.path.isfile( WEIGHTSFNAME):
        model.load_weights( WEIGHTSFNAME)

    # Save initial model so generator.py can start chugging
    model.save_weights( WEIGHTSFNAME)
    print( 'Waiting for generated files in %s ...' % GENDIR)
    while len( glob.glob( "%s/*.json" % GENDIR)) < 100:
        time.sleep(1)

    print( 'Kicking off...')
    print( 'Batchsize is %d. An epoch has %d batches = %d samples' % (args.batchsize, args.loadsize, args.loadsize * args.batchsize))

    epoch = 0
    while 1: # Hit Ctrl-C if you want to stop training
        epoch += 1
        print( '\nLoading %d samples...' % (args.loadsize * args.batchsize))
        inputs,v_targets = load_random_samples( GENDIR, args.loadsize * args.batchsize, args.puzzlesize)
        v_loss = 0.0
        for i in range( args.loadsize):
            batch_inputs = inputs[ i*args.batchsize : (i+1) * args.batchsize ]
            batch_v_targets = v_targets[ i*args.batchsize : (i+1) * args.batchsize ]
            cur_v_loss = model.train_on_batch( batch_inputs, [batch_v_targets] )
            v_loss += cur_v_loss
        v_loss /= args.loadsize
        #model.save_weights( WEIGHTSFNAME)
        #epfname = 'train_%d.weights' % epoch
        print( 'v_loss %e' % v_loss)
        #print( 'Saving weights after %d samples epoch %d' % (args.loadsize * args.batchsize, epoch))
        #model.save_weights( epfname)

def load_random_samples( folder, n_files, puzzlesize):
    ' Load n_files into memory, split into inputs and targets'
    files = glob.glob( "%s/*.json" % folder)
    files = np.random.choice( files, n_files) # sample with replacement
    inputs = []
    v_targets = []
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
        v_targets.append( 2.0 * jsn['v'] - 1.0) # Map to (-1,1) for tanh

    return np.array( inputs), np.array( v_targets)

if __name__ == '__main__':
    main()
