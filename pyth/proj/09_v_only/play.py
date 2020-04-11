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
      %s --playouts <int> --nshuffles <int> [--cpuct <float=0.1>]
    Description:
      --playouts: How many additional nodes to explore before deciding on a move.
         The relevant part of the tree is inherited from the previous move.
      --nshuffles: Start position is this many shifts away from solution.
      --cpuct: Hope factor. Larger means more exploration.
    Examples:
      %s --playouts 256 --nshuffles 2
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
    parser.add_argument( "--nshuffles", required=True, type=int)
    parser.add_argument( "--cpuct", type=float, default=0.1)
    args = parser.parse_args()

    # if args.playouts < 4:
    #     print( 'Error: You need at least four playouts for LEFT, RIGHT, UP, DOWN')
    #     exit(1)

    model = ShiftModel( SIZE)
    model.load_weights( 'model_3x3.weights')
    state = State.random( SIZE, nmoves=args.nshuffles)
    #state = State.from_list( SIZE, [3,1,2,6,4,5,0,7,8]) # test
    player = Player( state, model, int(args.playouts), args.cpuct)
    niter = 0
    while not state.solved():
        niter += 1
        # if niter > 10:
        #     print('>>>>>>>>> aborting')
        #     break
        print( state)
        print('iteration: %d' % niter)
        print('v: %.4f' % player.v())
        print('visits: %d' % player.N() )
        action, state = player.move()
    print( '>>> final state after %d iterations: %s' % (niter,state))


if __name__ == '__main__':
    main()
