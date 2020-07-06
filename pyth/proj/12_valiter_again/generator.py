#!/usr/bin/env python

'''
Generate shifting puzzle games for reinforcement learning.
Python 3
AHN, Apr 2020

'''

import os
# Disable GPU. CPU is faster for single net evals.
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

from pdb import set_trace as BP
import argparse
import math, os, glob, json
import numpy as np
import random
from game import Game
from search import Search
from datetime import datetime
import shortuuid
from shiftmodel import ShiftModel
from state import State, StateJsonEncoder

def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Generate training data for reinforcement learning
    Synopsis:
      %s --size <int> --validpct <int> [--maxfiles <int>]
    Description:
      Solve increasingly more difficult shifting puzzles using the model in shiftmodel.py .
      Each (state,v) gets saved to training_data or validation_data for use by a separate training process.
      --validpct: Which percentage of generated samples is used for validation.
      --maxnodes: How many nodes to expand before giving up on the puzzle.
      --maxfiles: If specified, delete old files to keep total number of files below maxfiles.
                  If you have several generators running, only one of them should have maxfiles specified.
    Example:
      %s --size 4 --maxnodes 400 --validpct 10 --maxfiles 10000
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--validpct", type=int, required=True)
    parser.add_argument( "--size", type=int, required=True)
    parser.add_argument( "--maxnodes", type=int, required=True)
    parser.add_argument( "--maxfiles", type=int, default=0)
    args = parser.parse_args()
    model = ShiftModel( size=args.size)
    gen = Generator( model, validpct=args.validpct, maxfiles=args.maxfiles, maxnodes=args.maxnodes)
    gen.run()

#==================
class Generator:
    TRAIN_FOLDER = 'training_data'
    VALID_FOLDER = 'validation_data'

    def __init__( self, model,
                  validpct=10,
                  maxnodes=200,
                  chunksize=100,
                  max_shuffles = 4 * 1024,
                  maxfiles=0,
                  del_old=False):
        self.model = model # The model used for self-play
        self.validpct = validpct
        self.maxnodes = maxnodes # Abort game if still no solution at this man node expansions.
        self.chunksize = chunksize # If we solved this many, increase shuffles
        self.max_shuffles = max_shuffles # Upper limit for shuffles. We start with 1, then increase.
        self.maxfiles = maxfiles # Only keep the newest maxfiles training samples

        self.modeltime = None # Reload model if newer
        self.modelfile = 'net_v.hd5' # Separate training process stores updated weights here

    def run( self):
        print( '>>> Writing to folders %s and %s' % (Generator.TRAIN_FOLDER, Generator.VALID_FOLDER))
        if not os.path.isdir( Generator.TRAIN_FOLDER):
            os.mkdir( Generator.TRAIN_FOLDER)
        if not os.path.isdir( Generator.VALID_FOLDER):
            os.mkdir( Generator.VALID_FOLDER)
        nshuffles = 4
        gameno = 0
        while nshuffles <= self.max_shuffles:
            maxdepth = nshuffles
            # A training process runs independently and will occasionally
            # save a better model.
            self.reload_model_if_newer()
            failures = 0
            for idx in range( self.chunksize):
                gameno += 1
                state = State.random( self.model.size, nshuffles)
                search = Search( state, self.model, self.maxnodes, maxdepth)
                g = Game(search)
                seq, found = g.play()
                if not found:
                    failures += 1
                    #self.save_steps( seq)
                    print( 'Game %d   failed\n' % gameno)
                else:
                    self.save_steps( seq)
                    print( 'Game %d solved in %d steps\n' % (gameno, len(seq)))

            if failures == 0: # Our excellent model needs a new challenge
                print( '0/%d failures at %d shuffles' % (self.chunksize,nshuffles))
                #nshuffles *= 2
                nshuffles += 4
                print( '>>> %s increasing to %d shuffles' % (datetime.now(), nshuffles))
            else: # we still need to improve
                print( '%d/%d failures at %d shuffles' % (failures, self.chunksize,nshuffles))
                print( 'staying at %d shuffles' % nshuffles)

            if self.maxfiles:
                self.delete_old_files()

    def reload_model_if_newer( self):
        if not os.path.exists( self.modelfile):
            print( 'Model file %s not found, using initial model.' % self.modelfile)
            return
        modtime = datetime.utcfromtimestamp( os.path.getmtime( self.modelfile))
        if self.modeltime is None or modtime > self.modeltime:
            self.modeltime = modtime
            self.model.load( self.modelfile)
            print( '>>> %s loaded new model' % datetime.now())

    def save_steps( self, seq):
        'Save individual solution steps as training or validation samples'
        for idx,step in enumerate(seq):
            # The last one is either a solution or unexpanded. Do not save and train.
            if idx+1 == len(seq):
                break
            dist = int(round(State.dist_from_v( step.v)))
            nn_dist = int(round(State.dist_from_v( step.nn_v)))
            difficulty = abs(dist-nn_dist)

            folder = Generator.TRAIN_FOLDER
            if random.randint( 1,100) <= self.validpct:
                folder = Generator.VALID_FOLDER

            fname = folder + '/%d_%04d_%04d_%s.json' % (self.model.size, difficulty, dist, shortuuid.uuid()[:8])
            step = { 'state':step.state, 'v':step.v, 'dist':dist }
            jsn = json.dumps( step, cls=StateJsonEncoder)
            with open(fname,'w') as f:
                f.write(jsn)

    def delete_old_files( self):
        maxvalid = int(self.maxfiles * self.validpct / 100)
        maxtrain = self.maxfiles - maxvalid
        'Limit number of training samples'
        files = glob.glob("%s/*.json" % Generator.TRAIN_FOLDER)
        if len(files) > maxtrain:
            files = sorted( files) # easy ones first
            #files = files[int(-1*len(files)*hardest_percentile):]
            #files.sort( key=os.path.getmtime)
            delfiles = files[:len(files)-maxtrain] # delete the easier ones
            print( 'Deleting %d easy training files, leaving %d' % (len(delfiles),maxtrain))
            for f in delfiles:
                os.remove( f)
        'Limit number of validation samples'
        files = glob.glob("%s/*.json" % Generator.VALID_FOLDER)
        if len(files) > maxvalid:
            files = sorted( files) # easy ones first
            #files.sort( key=os.path.getmtime)
            delfiles = files[:len(files)-maxvalid]
            print( 'Deleting %d easy validation files, leaving %d' % (len(delfiles),maxvalid))
            for f in delfiles:
                os.remove( f)

if __name__ == '__main__':
    main()
