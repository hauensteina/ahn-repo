#!/usr/bin/env python

'''
Try to solve an nxn shifting puzzle using a DNN model
Python 3
AHN, Mar 2020
'''

from pdb import set_trace as BP
import argparse
import os
from math import log, exp
import numpy as np
import random

from player import Player
from shiftmodel import ShiftModel
from state import State

SIZE = 3

def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Solve a shifting puzzle using UCT search with a DNN.
      Uses model_3x3.weights, taken from 07_valitershift.py .
    Synopsis:
      %s --playouts <int> [--cpuct <float=0.1>]
    Description:
      --playouts: How many additional nodes to explore before deciding on a move.
         The relevant part of the tree is inherited from the previous move.
      --cpuct: Hope factor. Larger means more exploration.
    Examples:
      %s --playouts 256 --cpuct 0.1
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--playouts", required=True, type=int)
    parser.add_argument( "--cpuct", type=float, default=0.1)
    args = parser.parse_args()

    if args.playouts < 4:
        print( 'Error: You need at least four playouts for LEFT, RIGHT, UP, DOWN')
        exit(1)

    model = ShiftModel( SIZE)
    model.load_weights( 'model_3x3.weights')
    state = State.random( SIZE, nmoves=2)
    player = Player( state, model, int(args.playouts), args.cpuct)
    niter = 0
    while not state.solved():
        niter += 1
        print( state)
        print('iteration: %d' % niter)
        if player.root.N:
            print('v: %.4f' % ( player.root.v / player.root.N ))
            print('visits: %d' % ( player.root.N ))
        action, state = player.move()
    print( '>>> final state after %d iterations: %s' % (niter,state))


if __name__ == '__main__':
    main()
