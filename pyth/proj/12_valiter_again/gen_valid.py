#!/usr/bin/env python

'''
Generate shifting puzzle validation set.
Python 3
AHN, May 2020

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
      %s: Generate validation data for reinforcement learning shifting puzzles
    Synopsis:
      %s --size <int> --maxshuffles <int> --nfiles <int>
    Description:
      Pull a number 1 to maxshuffles. Shuffle without cycles, save.
    Example:
      %s --size 4 --maxshuffles 100 --nfiles 1000
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--size", type=int, required=True)
    parser.add_argument( "--maxshuffles", type=int, required=True)
    parser.add_argument( "--nfiles", type=int, required=True)
    args = parser.parse_args()
    gen = Generator( args.size, args.maxshuffles, args.nfiles)
    gen.run()

#==================
class Generator:
    VALID_FOLDER = 'validation_data'

    def __init__( self, size, maxshuffles, nfiles):
        self.size = size
        self.maxshuffles = maxshuffles
        self.nfiles = nfiles

    def run( self):
        print( '>>> Writing to folder %s' % Generator.VALID_FOLDER)
        if not os.path.isdir( Generator.VALID_FOLDER):
            os.mkdir( Generator.VALID_FOLDER)

        for idx in range( self.nfiles):
            nshuffles = random.randint( 1, self.maxshuffles)
            print( '%d %d' % (idx, nshuffles))
            state, nshuffles = State.random_no_cycle( self.size, nshuffles)
            fname = Generator.VALID_FOLDER + '/%d_%04d_%s.json' % (self.size, nshuffles, shortuuid.uuid()[:8])
            v = State.v_from_dist( nshuffles)
            s = { 'state':state, 'v':v, 'dist':nshuffles }
            jsn = json.dumps( s, cls=StateJsonEncoder)
            with open(fname,'w') as f:
                f.write(jsn)

if __name__ == '__main__':
    main()
