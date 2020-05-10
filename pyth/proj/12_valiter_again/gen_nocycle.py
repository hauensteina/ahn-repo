#!/usr/bin/env python

'''
Generate shifting puzzle games for reinforcement learning,
without cycles to better predict ctg.
Python 3
AHN, May 2020

'''

import os
# Disable GPU. CPU is faster for single net evals.
os.environ['CUDA_VISIBLE_DEVICES']='-1'
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
      %s --size <int> [--maxfiles <int>]
    Description:
      Solve increasingly more difficult shifting puzzles using the model in shiftmodel.py .
      Each (state,v) gets saved to generated_data for use by a separate training process.
      --maxfiles: If specified, delete old files to keep total number of files below maxfiles.
                  If you have several generators running, only one of them should have maxfiles specified.
    Example:
      %s --size 4 --maxfiles 10000
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    #parser.add_argument( '--validpct', type=int, required=True)
    parser.add_argument( '--size', type=int, required=True)
    #parser.add_argument( '--maxnodes', type=int, required=True)
    parser.add_argument( '--maxfiles', type=int, default=0)
    args = parser.parse_args()
    model = ShiftModel( size=args.size)
    gen = Generator( model, maxfiles=args.maxfiles)
    gen.run()

#==================
class Generator:
    OUTFOLDER = 'generated_data'
    EPSILON = 1.0

    def __init__( self, model,
                  #maxnodes=200,
                  chunksize=100,
                  #max_shuffles = 4 * 1024,
                  maxfiles=0):
        self.model = model # The model used for self-play
        #self.validpct = validpct
        #self.maxnodes = maxnodes # Abort game if still no solution at this man node expansions.
        self.chunksize = chunksize # If we solved this many, increase shuffles
        #self.max_shuffles = max_shuffles # Upper limit for shuffles. We start with 1, then increase.
        self.maxfiles = maxfiles # Only keep the newest maxfiles training samples

        self.modeltime = None # Reload model if newer
        self.modelfile = 'net_v.hd5' # Separate training process stores updated weights here

    def run( self):
        print( '>>> Writing to folder %s' % Generator.OUTFOLDER)
        if not os.path.isdir( Generator.OUTFOLDER):
            os.mkdir( Generator.OUTFOLDER)
        maxshuffles = 4
        failhisto = np.full( maxshuffles, Generator.EPSILON)
        while 1:
            gameno = 0
            # A training process runs independently and will occasionally
            # save a better model.
            self.reload_model_if_newer()

            # Pull sample for next chunk
            #failhisto = np.sqrt( failhisto)
            failhisto /= np.sum(failhisto)
            #print(failhisto)
            states = []
            for idx in range( self.chunksize):
                # Pull a bin index from the failhisto density
                nshuffles = np.random.choice( len(failhisto), size=1, p=failhisto)[0] + 1
                state = State.random_no_cycle( self.model.size, nshuffles)[0]
                states.append( (state,nshuffles))

            # Reset failure stats
            failhisto = np.full( maxshuffles, Generator.EPSILON)
            # Run on the new chunk
            lengths = set()
            nfailures = 0
            for state,nshuffles in states:
                max_expansions = 2 * nshuffles
                gameno += 1
                search = Search( state, self.model, max_expansions, maxdepth=nshuffles)
                g = Game(search)
                seq, found = g.play()
                if not found:
                    #BP()
                    failhisto[nshuffles-1] += 1
                    nfailures += 1
                    print( 'Game %d   failed' % gameno)
                else:
                    print( 'Game %d solved in %d steps' % (gameno, len(seq)-1))

                dist = nshuffles
                seqsteps = len(seq) - 1
                if found and seqsteps < dist:
                    # Save each step of the shorter solution for training
                    print( 'Found shorter solution: %d < %d' % (seqsteps, dist))
                    #for idx,s in enumerate(seq[:-1]):
                    #    self.save_one_state( s.state, len(seq) - idx - 1)
                # Save the whole tree with iterated values for training
                nodes = search.get_all_expanded_nodes()
                for node in nodes:
                    if State.dist_from_v( node.v) >= 100:
                        BP()
                        print('Oops')
                        exit(1)
                    self.save_one_state( node.state, State.dist_from_v( node.v))

            # We increase if up to one less than the max is error free, hence -1.
            # This means we train with a lookahead of 1, which helps a lot.
            nerrors = sum(failhisto[:-1] - Generator.EPSILON)
            if nerrors == 0:
                stepsize = 1
                newshuffles = maxshuffles + stepsize
                # Histogram support longer, all equally likely
                failhisto = np.full( newshuffles, Generator.EPSILON)
                # The new ones get half the total probability mass
                #failhisto[-stepsize:] = np.sum(failhisto[:maxshuffles]) / stepsize
                print('No failures at maxshuffles=%d, increasing to %d' % (maxshuffles, newshuffles))
                maxshuffles = newshuffles
            else:
                print('%d failures at maxshuffles=%d, staying at %d' % (nerrors, maxshuffles, maxshuffles))
                print(failhisto)

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

    def save_one_state( self, state, d):
        ' Save one state and its cost to go as training sample '
        v = State.v_from_dist( d)
        step = { 'state':state, 'v':v, 'dist':d }
        folder = Generator.OUTFOLDER
        fname = folder + '/%d_%04d_%s.json' % (self.model.size, d, shortuuid.uuid()[:8])
        jsn = json.dumps( step, cls=StateJsonEncoder)
        with open(fname,'w') as f:
            f.write(jsn)

    def delete_old_files( self):
        maxtrain = self.maxfiles
        'Limit number of training samples'
        files = glob.glob("%s/*.json" % Generator.OUTFOLDER)
        if len(files) > maxtrain:
            files = sorted( files) # easy ones first
            delfiles = files[:len(files)-maxtrain] # delete the easier ones
            print( 'Deleting %d old training files, leaving %d' % (len(delfiles),maxtrain))
            for f in delfiles:
                os.remove( f)

if __name__ == '__main__':
    main()
