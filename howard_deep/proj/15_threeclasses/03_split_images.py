#!/usr/bin/env python

# /********************************************************************
# Filename: split_images.py
# Author: AHN
# Creation Date: Feb 15, 2018
# **********************************************************************/
#
# Divide images in a folder into train, valid, test sets
#

from __future__ import division, print_function
from pdb import set_trace as BP
import os,sys,re,json, glob
import numpy as np
import random
import shutil
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Divide images in a folder into train, valid, test sets
    Synopsis:
      %s --folder <folder> --trainpct <n> --validpct <n> --substr <substring>
    Description:
      Splits the jpg files in folder into train, valid, and test files.
      Only use files containing <substring> in the name.
    Example:
      %s --folder images --trainpct 80 --validpct 10 --substr rgb
      The remaining 10pct will be test data
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# Randomly split the jpg files in a folder into
# train, valid, test
#-------------------------------------------------
def split_files( folder, trainpct, validpct, substr=''):
    files = glob.glob( folder + '/*_hood_*.jpg')
    files = [os.path.basename(f) for f in files];
    files = [f for f in files if substr in f];
    random.shuffle( files)
    ntrain = int( round( len( files) * (trainpct / 100.0)))
    nvalid = int( round( len( files) * (validpct / 100.0)))
    trainfiles = files[:ntrain]
    validfiles = files[ntrain:ntrain+nvalid]
    testfiles  = files[ntrain+nvalid:]

    # Also get the threshed versions
    trainfiles_thresh = [ re.sub( '_hood_', '_thresh_', x) for x in trainfiles ]
    validfiles_thresh = [ re.sub( '_hood_', '_thresh_', x) for x in validfiles ]
    testfiles_thresh  = [ re.sub( '_hood_', '_thresh_', x) for x in testfiles ]

    os.makedirs( 'test/all_files')
    os.makedirs( 'train/all_files')
    os.makedirs( 'valid/all_files')

    for f in trainfiles:
        shutil.copy2( folder + '/' + f, 'train/all_files/' + f)
    for f in validfiles:
        shutil.copy2( folder + '/' + f, 'valid/all_files/' + f)
    for f in testfiles:
        shutil.copy2( folder + '/' + f, 'test/all_files/' + f)

    for f in trainfiles_thresh:
        shutil.copy2( folder + '/' + f, 'train/all_files/' + f)
    for f in validfiles_thresh:
        shutil.copy2( folder + '/' + f, 'valid/all_files/' + f)
    for f in testfiles_thresh:
        shutil.copy2( folder + '/' + f, 'test/all_files/' + f)

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--folder",      required=True)
    parser.add_argument( "--substr",      required=True)
    parser.add_argument( "--trainpct",    required=True, type=int)
    parser.add_argument( "--validpct",    required=True, type=int)
    args = parser.parse_args()
    split_files( args.folder, args.trainpct, args.validpct, args.substr)


if __name__ == '__main__':
    main()
