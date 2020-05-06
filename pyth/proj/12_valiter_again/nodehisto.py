#!/usr/bin/env python

'''
Generate histogram for the model output node
Python 3
AHN, Apr 2020

'''

import os
# Disable GPU. CPU is faster for single net evals.
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf

from pdb import set_trace as BP
import argparse
import math, os, glob, json
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
      %s: Generate histogram for the model output node
    Synopsis:
      %s --model <fname> --nbins <int> --nsamples <int>
    Description:
      --model: Filename to load the model from
      --nbins: How many bins in the histo
      --nsamples: How many inputs to run through the model
    Example:
      %s --model net_v.hd5 --nsamples 1000 --nbins 10
--
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument( "--model", required=True)
    parser.add_argument( "--nbins", type=int, required=True)
    parser.add_argument( "--nsamples", type=int, required=True)
    args = parser.parse_args()

    model = ShiftModel.from_file( args.model)
    inputs = []
    NSHUFFLES = 1000
    print( 'Generating inputs ...')
    inputs = []
    for idx in range( args.nsamples):
        inp = State.random( model.size, NSHUFFLES).encode()
        inputs.append( inp)
    print( 'Running the model ...')
    outputs = model.predict( np.asarray(inputs))

    dists = [ State.dist_from_v( o[0]) for o in outputs ]
    plot_histo( dists, nbins=args.nbins)

def plot_histo( vals, nbins):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots()

    ax.hist( vals, nbins, rwidth=0.95)
    ax.set_xlabel( 'Estimated Cost to Go')
    ax.set_ylabel( 'Count')
    ax.set_title( 'Output Values')

    fig.tight_layout()
    plt.savefig('nodehisto.svg')
    print( 'Output is in nodehisto.svg')


if __name__ == '__main__':
    main()
