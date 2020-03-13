#!/usr/bin/env python

# Try to solve an nxn shifting puzzle by A* search.
# Use Manhattan distance as heuristic for remaining moves.
# Python 3
# AHN, Mar 2020

from pdb import set_trace as BP
import argparse
import os
from astar import Astar
import numpy as np
import random

LEFT=0
RIGHT=1
UP=2
DOWN=3

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Solve a shifting puzzle using A* search.
    Synopsis:
      %s --size <int>
    Description:
      --size: Side length. An 8-puzzle is size=4.
    Example:
      %s --size 3
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
    args = parser.parse_args()

    state = State.random( args.size)
    astar =  Astar( state)
    niter = 0
    while not astar.done:
        niter += 1
        if niter % 1000 == 0:
            print('iteration: %d estimate: %d dist:%d depth: %d' %
                  (niter, astar.best_leaf_estimate, astar.best_leaf_distance, astar.best_leaf_depth))
        astar.search()
    solution = astar.get_solution()
    print( 'Tree size: %d\n' % astar.treesize)
    for state in solution:
        print( state)

#=================
class State:
    #--------------------------
    def __init__( self, size):
        self.s = size
        self.empty_idx = 0
        self.arr = None
        self.estimate = None

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
        res += '\nest: %d' % self.estimate
        return res

    #--------------------------
    def clone( self):
        res = State( self.s)
        res.empty_idx = self.empty_idx
        res.arr = self.arr.copy()
        return res

    @classmethod
    #-------------------------
    def random( cls, size):
        NMOVES = 1000
        res = cls( size)
        res.arr = np.zeros( size * size, int)
        for i,_ in enumerate( res.arr): res.arr[i] = i

        for _ in range(NMOVES):
            acts = res.get_actions()
            res = res.act( random.choice(acts) )
        return res

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

    #=============================
    # Methods required by Astar
    #=============================

    # Return possible actions
    #---------------------------
    def get_actions( self):
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

        #acts = np.nonzero( res)[0]
        res = [ i for i in range(4) if res[i] ]
        return res

    # Apply an action and return new state
    #----------------------------------------
    def act( self, action_idx):
        tile_idx = self.new_idx( action_idx)
        new_state = self.clone()
        new_state.arr[self.empty_idx] = new_state.arr[tile_idx]
        new_state.arr[tile_idx] = 0
        new_state.empty_idx = tile_idx
        new_state.estimate = new_state.dist_from_solution()
        return new_state

if __name__ == '__main__':
    main()
