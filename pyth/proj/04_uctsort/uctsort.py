#!/usr/bin/env python

# Sort an array of numbers using UCTree.search() .
# We know if we're sorted, and we can swap two numbers.
# The search handles the rest.
# Python 3
# AHN, Mar 2020

from uctree import UCTree
import argparse
import os
import numpy as np
import random

from pdb import set_trace as BP

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Sort shuffled numbers 0..size-1 ucing UCT search and simple heuristics.
    Synopsis:
      %s --size <size> --playouts <playouts>
    Description:
      --size: Size of the array to sort
      --playouts: How many additional nodes to explore before deciding on a move.
         The relevant part of the tree is inherited from the previous move.
    Example:
      %s --size 5 --playouts 10
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
    initial_state = State.random( args.size)
    state = initial_state
    tree =  UCTree( state, get_v_p, get_next_state, c_puct=0.6)
    while not tree.done( state):
        print( '>>> state: %s' % str(state.arr))
        action, state = tree.search( n_playouts=args.playouts)
    print( '>>> final state: %s' % str(state.arr))

# Action on state returns next state
#---------------------------------------
def get_next_state( state, action_idx):
    next_state = State( state.arr.copy())
    arr = next_state.arr; idx = action_idx
    arr[idx+1], arr[idx] = arr[idx], arr[idx+1]
    #print( '>>>>>> trying: %s -> %s' % (str(state.arr), str(arr)))
    return next_state

# v: How many are in the right place.
# p[i]: 1 if p[i] larger than p[i+1] else 0
# Both v and p get normalized.
# v,p == 1.0,None means we found a solution.
#--------------------------------------------
def get_v_p( state):
    arr = state.arr
    ngood = 0
    for i,x in enumerate( arr):
        if i == x: ngood +=1
    v = ngood / len( arr)
    p = np.zeros( len(arr)-1)
    for i,x in enumerate(p):
        p[i] = 1 if arr[i+1] < arr[i] else 0
    if ngood == len(arr):     # Solution
        v = 1.0
        p = None
    else: # No solution, keep going
        p /= np.sum(p)

    return v,p

# State is just a shuffled array of integers 0 .. size-1
#=========================================================
class State:
    #--------------------------
    def __init__( self, arr):
        self.arr = arr

    @classmethod
    #-------------------------
    def random( cls, size):
        return cls( random.sample(range(0,size), size))

# Action swaps elements swap_index and swap_index+1
#====================================================
class Action:
    #---------------------------------
    def __init__( self, swap_index):
        self.swap_index = swap_index

if __name__ == '__main__':
    main()
