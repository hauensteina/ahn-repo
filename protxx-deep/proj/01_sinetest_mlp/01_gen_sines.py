#!/usr/bin/env python

# /********************************************************************
# Filename: 01_gen_sines.py
# Author: AHN
# Creation Date: Nov 29, 2018
# **********************************************************************/
#
# Generate simple sine wave train and valid data to get started on
# time series classification.
# Two classes, sines of different frequencies, noise added.
#

from __future__ import division, print_function
from pdb import set_trace as BP
import os,sys,re,json
import numpy as np
from numpy.random import random
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

WAVELEN1 = 10.0
WAVELEN2 = 20.0
AMPLITUDE = 2.0
NOISE = 2.0 # sigma, relative to amplitude

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Generate two classes with sinusoidal time series data
    Synopsis:
      %s [--randphase] --length <n> --ntrain <n> --nval <n>
    Description:
      Generate two classes with sinusoidal time series data.
      The two classes have different frequencies.
      Noise is added to each sample.
      If --randphase is given, the signals start with a random phase shift.
    Example:
      %s --length 1000 --ntrain 1000 --nval 100
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--length",    required=True, type=int)
    parser.add_argument( "--ntrain",    required=True, type=int)
    parser.add_argument( "--nval",      required=True, type=int)
    parser.add_argument( "--randphase", required=False, action='store_true')
    args = parser.parse_args()

    np.random.seed(0) # Make things reproducible

    trainfolder = 'train/all_files'
    valfolder   = 'valid/all_files'
    if not os.path.exists( trainfolder): os.makedirs( trainfolder)
    if not os.path.exists( valfolder):   os.makedirs( valfolder)

    gen_sines( args.ntrain, trainfolder, WAVELEN1, WAVELEN2, AMPLITUDE, NOISE, args.length, args.randphase )
    gen_sines( args.nval, valfolder, WAVELEN1, WAVELEN2, AMPLITUDE, NOISE, args.length, args.randphase )

# Generate a sine wave with the given parameters.
# Randomize the phase if randphase is true.
#------------------------------------------------------------
def gen_sine( wavelen, ampl, noise, length, randphase):
    phase = 0
    if randphase:
        phase = np.random.randint( wavelen)
    t = np.arange( phase, phase + length)
    y = ampl * np.sin( t * (2*np.pi) / wavelen)
    noise = np.random.normal( 0, noise * ampl, length)
    y += noise
    return y

# Generate n sines, half with wavelen1, half with wavelen2.
# Sore in folder.
# Also generate a json indicating the class.
#-------------------------------------------------------------------------------------
def gen_sines( n_sines, folder, wavelen1, wavelen2, ampl, noise, length, randphase):
    n1 = int(n_sines / 2)

    def gendump( idx, wavelen, cclass):
        fname = '%07d' % idx
        fcsv  = folder + '/' + fname + '.csv'
        fjson = folder + '/' + fname + '.json'
        sine = gen_sine( wavelen, ampl, noise, length, randphase)
        # Make csv
        csv = 't,y\n'
        for t,y in enumerate(sine):
            csv += '%d,%f\n' % (t,y)
        with open( fcsv,'w') as f: f.write( csv)
        meta =  json.dumps( { 'class': cclass })
        with open( fjson,'w') as f: f.write( meta)

    for idx in range( n1):
        gendump( idx, wavelen1, 0)
    for idx in range( n1, n_sines):
        gendump( idx, wavelen2, 1)


if __name__ == '__main__':
    main()
