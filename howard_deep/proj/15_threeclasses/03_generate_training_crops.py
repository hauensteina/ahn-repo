#!/usr/bin/env python

# /********************************************************************
# Filename: generate_training_crops.py
# Author: AHN
# Creation Date: Feb 22, 2018
# **********************************************************************/
#
# Get individual intersection crops for NN training material in wallstedt format
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

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

import cv2

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --   Get individual intersection crops for NN training material
    Synopsis:
      %s --infolder <ifolder> --outfolder <ofolder> --trainpct <n> --validpct <n>
    Description:
      Scales and perspective transforms input images, save 361 crops for each.
    Example:
      %s --infolder ~/training_data_kc/mike/20180113/verified --outfolder crops-kc --trainpct 80 --validpct 10
      The remaining 10pct will be test data
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# Collect matching speg, json, sgf in a dictionary
#----------------------------------------------------
def collect_files( infolder):
    # Find images
    imgs = ut.find( infolder, '*.jpeg')
    imgs += ut.find( infolder, '*.jpg')
    # Basenames
    basenames = [os.path.basename(f) for f in imgs]
    basenames = [os.path.splitext(f)[0] for f in basenames]
    # sgf files
    sgfs = [ut.find( infolder, '%s.sgf' % f)[0] for f in basenames]
    # json files
    jsons = [ut.find( infolder, '%s_intersections.json' % f)[0] for f in basenames]
    # Collect in dictionary
    files = {}
    for i,bn in enumerate( basenames):
        d = {}
        files[bn] = d
        d['img'] = imgs[i]
        d['sgf'] = sgfs[i]
        d['json'] = jsons[i]
    # Sanity check
    for bn in files.keys():
        d = files[bn]
        if not bn in d['img']:
            print( 'ERROR: Wrong img name for key %s' % (d['img'], bn))
            exit(1)
        elif not bn in d['sgf']:
            print( 'ERROR: Wrong sgf name %s for key %s' % (d['sgf'], bn))
            exit(1)
        elif not bn in d['json']:
            print( 'ERROR: Wrong json name for key %s' % (d['json'], bn))
            exit(1)
    return files

#---------------------------------
def zoom_in( imgfile, jsonfile):
    # Read the image
    img = cv2.imread( imgfile, 1)
    # Parse json
    columns = json.load( open( jsonfile))
    for c,col in enumerate( columns):
        for r, row in enumerate( col):
            x = row['x']
            y = row['y']
            cv2.circle( img,
                        (int(x), int(y)),
                        10,
                        ( 0, 0, 255 ),
                        -1 )
    cv2.imwrite( '/home/ubuntu/ahn-repo/howard_deep/proj/15_threeclasses/tt.jpg', img);

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--infolder",      required=True)
    parser.add_argument( "--outfolder",      required=True)
    parser.add_argument( "--trainpct",    required=True, type=int)
    parser.add_argument( "--validpct",    required=True, type=int)
    args = parser.parse_args()
    files = collect_files( args.infolder)

    for i,k in enumerate( files.keys()):
        f = files[k]
        img = zoom_in( f['img'], f['json'])
        if i > 3: break


if __name__ == '__main__':
    main()
