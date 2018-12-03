#!/usr/bin/env python

# /********************************************************************
# Filename: 01_gen_sines.py
# Author: AHN
# Creation Date: Nov 30, 2018
# **********************************************************************/
#
# Generate single float inputs from two classes, <42 and >42
# We use this to train a one neuron model that just finds the threshold(42).
#

from __future__ import division, print_function
from pdb import set_trace as BP
import os,sys,re,json
import numpy as np
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt


#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Generate two single value classes separated by a threshold at 42.
    Synopsis:
      %s  --ntrain <n> --nval <n>
    Example:
      %s --ntrain 1000 --nval 100
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
    parser.add_argument( "--ntrain",    required=True, type=int)
    parser.add_argument( "--nval",      required=True, type=int)
    args = parser.parse_args()

    np.random.seed(0) # Make things reproducible

    trainfolder = 'train/all_files'
    valfolder   = 'valid/all_files'
    if not os.path.exists( trainfolder): os.makedirs( trainfolder)
    if not os.path.exists( valfolder):   os.makedirs( valfolder)

    gen_rands( args.ntrain, trainfolder, 0, 42, 100)
    gen_rands( args.nval, valfolder, 0, 42, 100)

#-------------------------------------------------------
def gen_rands( nsamps, folder, mmin, thresh, mmax):
    n1 = int( nsamps / 2)

    def dump( vals, cclass, sstart):
        for idx,v in enumerate(vals):
           fname = '%07d' % (idx+sstart)
           fcsv  = folder + '/' + fname + '.csv'
           fjson = folder + '/' + fname + '.json'
           # Make csv
           csv = 'x\n' + '%f\n' % v
           with open( fcsv,'w') as f: f.write( csv)
           meta =  json.dumps( { 'class': cclass }) + '\n'
           with open( fjson,'w') as f: f.write( meta)

    dump( np.random.uniform( mmin, thresh, n1), 0, 0)
    dump( np.random.uniform( thresh+1, mmax, nsamps-n1), 1, n1)


if __name__ == '__main__':
    main()

