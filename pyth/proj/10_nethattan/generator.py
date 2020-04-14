#!/usr/bin/env python

'''
Generate random shifting puzzle positions for DNN Manhattan distance training
Python 3
AHN, Apr 2020

'''

from pdb import set_trace as BP
import argparse
import math, os, glob, json
import numpy as np
from datetime import datetime
import shortuuid
from state import State, StateJsonEncoder

def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''

    Name:
      %s: Generate training data for reinforcement learning
    Synopsis:
      %s --size <int> [--nshuffles <int>] [--nsamples <int>]
    Description:
      Generate random shifting puzzle positions with Manhattan distance
      as json in generator.out, for DNN training.
    Example:
      %s --size 3 --nshuffles 1000 --nsamples 100000
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--size", required=True, type=int)
    parser.add_argument( "--nshuffles", type=int, default=1000)
    parser.add_argument( "--nsamples", type=int, default=100000)
    args = parser.parse_args()

    OUTFOLDER = 'generator.out'
    if not os.path.isdir( OUTFOLDER):
        os.mkdir( OUTFOLDER)
    for idx in range(args.nsamples):
        if idx and idx % 1000 == 0:
            print( 'Generated %d/%d samples' % (idx, args.nsamples))
        state = State.random( args.size, args.nshuffles)
        dist = state.manhattan_dist()
        sample = { 'state':state, 'dist':dist, 'v':State.v_from_dist( dist) }
        jsn = json.dumps( sample, cls=StateJsonEncoder) + '\n'
        fname = OUTFOLDER + '/%d_%04d_%s.json' % (args.size, dist, shortuuid.uuid()[:8])
        with open(fname,'w') as f:
            f.write(jsn)
    print( 'Generated %d samples' % args.nsamples )


if __name__ == '__main__':
    main()
