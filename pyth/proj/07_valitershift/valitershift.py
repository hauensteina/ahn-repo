#!/usr/bin/env python

# Try to solve an nxn shifting puzzle by training a deep NN for value iteration.
# Inspired by "Solving the Rubik’s cube with deep reinforcement learning and search"
# https://www.nature.com/articles/s42256-019-0070-z
# Python 3
# AHN, Mar 2020

from pdb import set_trace as BP
import argparse
import os
from math import log, exp
import numpy as np
import random
from vimodel import VIModel
from experiencebuf import ExperienceBuf

import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint

num_cores = 4
GPU=0

if GPU:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session( config=config)
    K.set_session( session)
else:
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto( intra_op_parallelism_threads=num_cores,\
                             inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
                             device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session( config=config)
    K.set_session( session)


LEFT=0
RIGHT=1
UP=2
DOWN=3

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Train net to solve a shifting puzzle using deep value iteration.
    Synopsis:
      %s --size <int> --nsteps <int> --npuzzles <int>
    Description:
      --size: Side length. An 8-puzzle is size=3.
      --nsteps: Train until it solves states that are scrambled n steps.
      --minsteps: Start training with minsteps scrambling. Default is 1.
      --npuzzles: How many start configurations to explore per epoch.
    Example:
      %s --size 3 --nsteps 2 --npuzzles 1000
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#------------
def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--size", required=True, type=int)
    parser.add_argument( "--minsteps", required=False, default=1, type=int)
    parser.add_argument( "--nsteps", required=True, type=int)
    parser.add_argument( "--npuzzles", required=True, type=int)
    args = parser.parse_args()

    model = VIModel( args.size)

    # Load weights of some previous iteration, if they are there
    for steps in reversed(range(1,args.nsteps+1)):
        fname = 'model_%0.2d' % steps
        if model.load_weights( fname):
            print( '>>>>>>>>>>>>>>> Loaded model weights from %s' % fname)
            break

    steps = args.minsteps
    epoch = 1
    while steps <= args.nsteps:
        misses = train( model, args.npuzzles, steps)
        if misses == 0:
            print( '>>>>>>>> Solved %d without miss for %d steps.\n' % (args.npuzzles, steps))
            model.save_weights( 'model_%0.2d' % steps)
            steps += 1
            epoch = 1
        else:
            print('=========================')
            print( 'Epoch: %d Misses: %d / %d\n' % (epoch, misses,args.npuzzles))
            epoch += 1

#-------------------------------------------
def train( model, npuzzles, nsteps):
    MOVELIMIT = nsteps * 2
    BATCHSIZE = 500
    BUFSIZE = 2 * BATCHSIZE
    samples = ExperienceBuf( BUFSIZE)
    misses = 0
    for gamenum in range( npuzzles):
        gameloss = 0.0
        state = State.random( model.size, nsteps)
        if gamenum % ( npuzzles // 10) == 0:
            print( '>>>>> Puzzle %d' % gamenum)
        #print( state)
        v = np.zeros( len(state.get_action_flags()), float)
        nmoves = 0
        state_vals = []
        while not state.solved() and nmoves < MOVELIMIT:
            nmoves += 1
            #print( 'move: %d' % nmoves)
            actions = state.get_action_flags()
            v.fill( 0.0)
            for action, possible in enumerate(actions):
                if not possible: continue
                new_state = state.act(action)
                if new_state.hash() in state.history: continue
                if new_state.solved():
                    #print( '>>>>> solved!')
                    v[action] = 1.0
                else:
                    new_state_enc = new_state.encode()
                    v[action] = model.predict( new_state_enc)
            best_action = np.argmax(v)
            best_value = np.max(v)
            # This improves the current value estimate sort of like Q-learning.
            # Instead of a discount multiplier, we add 1 to the distance.
            updated_value = state.dist_to_v( state.v_to_dist( best_value) + 1 )
            state_vals.append( (state,updated_value) )
            state = state.act( best_action)
            #print(state)

        # Remember new values after each successful solution
        if state.solved():
            for s,v in state_vals:
                samples.remember( s.encode(), np.array( (v,), float ))
            inputs, targets = samples.get_batch( model, batch_size=BATCHSIZE)
            batchloss, acc =  model.train_on_batch( inputs, targets)
            gameloss += batchloss
            if gamenum % ( npuzzles // 10) == 0:
                print( 'Game %d/%d | Moves %d | Loss %e' % (gamenum, npuzzles, nmoves, gameloss))
        else:
            #print( '##################### Not solved, ignoring')
            misses += 1
    return misses


#=================
class State:
    #----------------------------------
    def __init__( self, size):
        self.s = size
        self.empty_idx = 0
        self.hashval = None
        self.arr = None
        self.history = set()

    #---------------------
    def __repr__( self):
        res = ''
        old_row = -1
        for idx, x in enumerate( self.arr):
            row, col = self.coords( idx)
            if row != old_row:
                old_row = row
                res += '\n'
            if x: res += '%3d' % x
            else: res += '   '
        return res

    #--------------------------
    def clone( self):
        res = State( self.s)
        res.empty_idx = self.empty_idx
        res.arr = self.arr.copy()
        res.hist = self.history.copy()
        return res

    @classmethod
    #---------------------------------
    def random( cls, size, nsteps):
        while 1:
            res = cls( size)
            res.arr = np.zeros( size * size, int)
            for i,_ in enumerate( res.arr): res.arr[i] = i
            for _ in range(nsteps):
                acts = res.get_action_list()
                res = res.act( random.choice(acts), store_history = False )
            if not res.solved(): break
        return res

    #-------------------
    def solved( self):
        for i in range( self.s * self.s):
            if i != self.arr[i]: return False
        return True

    #-------------------------
    def coords( self, index):
        row = index // self.s
        col = index % self.s
        return row,col

    #---------------------------
    def index( self, row, col):
        return self.s * row + col

    #-----------------------------
    def new_idx( self, action):
        row, col = self.coords( self.empty_idx)
        if action == LEFT:
            return self.index( row, col-1)
        elif action == RIGHT:
            return self.index( row, col+1)
        elif action == UP:
            return self.index( row-1, col)
        elif action == DOWN:
            return self.index( row+1, col)
        else:
            print( 'Error: new_idx(): Invalid action %d' % action)

    V_TEMPERATURE = 2.0
    # Turn estimate of distance into a [0,1] float
    #-----------------------------------------------
    def dist_to_v( self, dist):
        alpha = State.V_TEMPERATURE * (1.0 / (self.s * self.s * self.s))
        q = exp( -1 * alpha * dist)
        return q

    # Turn [0,1] float value back into distance estimate
    #-------------------------------------------------------
    def v_to_dist( self, v):
        alpha = State.V_TEMPERATURE * (1.0 / (self.s * self.s * self.s))
        dist = -1 * log(v) / alpha
        return dist

    # Return possible actions in array of size 4, nonzero means possible.
    #----------------------------------------------------------------------
    def get_action_flags( self):
        mmax = self.s - 1
        row,col = self.coords( self.empty_idx)
        res = np.zeros( 4, int)
        if col == 0 and row == 0:
            res[DOWN] = res[RIGHT] = 1
        elif col == 0 and row == mmax:
            res[UP] = res[RIGHT] = 1
        elif col == mmax and row == 0:
            res[DOWN] = res[LEFT] = 1
        elif col == mmax and row == mmax:
            res[UP] = res[LEFT] = 1
        elif col == 0:
            res[UP] = res[DOWN] = res[RIGHT] = 1
        elif col == mmax:
            res[UP] = res[DOWN] = res[LEFT] = 1
        elif row == 0:
            res[LEFT] = res[RIGHT] = res[DOWN] = 1
        elif row == mmax:
            res[LEFT] = res[RIGHT] = res[UP] = 1
        else:
            res[LEFT] = res[RIGHT] = res[UP] = res[DOWN] = 1

        return res

    # Get a list of possible actions
    #--------------------------------
    def get_action_list( self):
        flags = self.get_action_flags()
        res = np.nonzero(flags)
        return res[0]

    # Apply an action and return new state
    #----------------------------------------
    def act( self, action_idx, store_history=True):
        tile_idx = self.new_idx( action_idx)
        new_state = self.clone()
        new_state.arr[self.empty_idx] = new_state.arr[tile_idx]
        new_state.arr[tile_idx] = 0
        new_state.empty_idx = tile_idx
        if (store_history):
            new_state.history.add( self.hash())
        return new_state

    # Encode state to feed into DNN
    #---------------------------------
    def encode( self):
        # Width, height, tile-channels
        res = np.zeros( (self.s, self.s, self.s * self.s), float)
        for idx in range( self.s * self.s): # for each position
            tile = self.arr[idx] # which tile is here
            r,c = self.coords( idx)
            res[r,c,tile] = 1 # store position in channel for this tile
        return res

    # Hash state to avoid cycles
    #-----------------------------
    def hash( self):
        if not self.hashval: self.hashval = hash(self.arr.tobytes())
        return self.hashval


if __name__ == '__main__':
    main()
