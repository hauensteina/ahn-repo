#!/usr/bin/env python

# Try to solve an nxn shifting puzzle by UCT search, without a neural net.
# The 15-puzzle (n=4) is most popular. 4x4 spots, one is empty.
# Python 3
# AHN, Mar 2020

from uctree import UCTree
import argparse
import os
import numpy as np
import random
import math

from pdb import set_trace as BP

LEFT=0
RIGHT=1
UP=2
DOWN=3

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Solve a shifting puzzle using UCT search and simple heuristics.
    Synopsis:
      %s --size <s> --playouts <playouts>
    Description:
      --size: Side length. s=4 is a 15-puzzle.
      --playouts: How many additional nodes to explore before deciding on a move.
         The relevant part of the tree is inherited from the previous move.
    Example:
      %s --size 3 --playouts 10
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
    parser.add_argument( "--playouts", required=True, type=int)
    args = parser.parse_args()

    if args.playouts < 4:
        print( 'Error: You need at least four playouts for LEFT, RIGHT, UP, DOWN')
        exit(1)

    state = State.random( args.size)
    tree =  UCTree( state, c_puct=0.6)
    niter = 0
    while not tree.done( state):
        niter += 1
        print( state)
        action, state = tree.search( n_playouts=args.playouts)
    print( '>>> final state after %d iterations: %s' % (niter,state))


#=================
class State:
    #--------------------------
    def __init__( self, size):
        self.s = size
        self.empty_idx = 0
        self.arr = None
        self.history = []
        self.hashval = None

    #---------------------
    def __repr__( self):
        res = ''
        old_row = -1
        for idx, x in enumerate( self.arr):
            row, col = self.coords( idx)
            if row != old_row:
                old_row = row
                res += '\n'
            res += ' %d' % x
        res += '\n'
        return res

    #--------------------------
    def clone( self):
        res = State( self.s)
        res.arr = self.arr.copy()
        res.history = self.history.copy()
        res.s = self.s
        return res

    @classmethod
    #-------------------------
    def random( cls, size):
        NMOVES = 1000
        res = cls( size)
        res.arr = np.zeros( size * size, int)
        for i,_ in enumerate( res.arr): res.arr[i] = i

        for _ in range(NMOVES):
            acts = res.possible_actions() # len 4, impossible ones are marked 0
            acts = np.nonzero( acts)[0] # indexes of nonzero actions
            res = res.act( random.choice(acts), store_history=False)
        return res

    # Return an array of length 4 where impossible actions are marked 0
    #--------------------------------------------------------------------
    def possible_actions( self):
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

    #-------------------------
    def coords( self, index):
        row = index // self.s
        col = index % self.s
        return row,col

    #---------------------------
    def index( self, row, col):
        return self.s * row + col

    #---------------------------------
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

    # Manhattan distance from solution
    #-----------------------------------
    def dist_from_solution( self):
        res = 0
        for idx,x in enumerate( self.arr):
            sol_row, sol_col = self.coords( x)
            row, col = self.coords( idx)
            d = abs( sol_row - row) + abs( sol_col - col)
            res += d
        return res

    # A number in (0,1] where 1 indicates a solution.
    #--------------------------------------------------
    def quality( self):
        TEMP = 4.0
        alpha = TEMP * (1.0 / (self.s * self.s * self.s))
        d = self.dist_from_solution()
        q = math.exp( -1 * alpha * d)
        return q

    #----------------
    def hash( self):
        if not self.hashval: self.hashval = hash(self.arr.tobytes())
        return self.hashval

    #=============================
    # Methods required by UCTree
    #=============================

    # Apply an action and return new state
    #------------------------------------------------
    def act( self, action_idx, store_history=True):
        tile_idx = self.new_idx( action_idx)
        new_state = self.clone()
        new_state.arr[self.empty_idx] = new_state.arr[tile_idx]
        new_state.arr[tile_idx] = 0
        new_state.empty_idx = tile_idx
        if (store_history):
            new_state.history.append( self.hash())
        return new_state

    # v: Normalized Manhattan distance from solution.
    # p[i]: Normalized v values of next possible states.
    # v,p == 1.0,None means we found a solution.
    #--------------------------------------------
    def get_v_p( self):
        # Current quality
        v = self.quality()

        if v == 1.0: # We're done
            return v,None

        # quality with lookahaead 1 gives p
        p = np.zeros( 4, float)
        possible = self.possible_actions()
        for action in (LEFT,RIGHT,UP,DOWN):
            if (possible[action]):
                next_state = self.act( action)
                if next_state.hash() in self.history: # No cycles.
                    p[action] = 0.0
                else:
                    p[action] = next_state.quality()

        if np.sum(p):
            p /= np.sum(p)

        return v,p

if __name__ == '__main__':
    main()
