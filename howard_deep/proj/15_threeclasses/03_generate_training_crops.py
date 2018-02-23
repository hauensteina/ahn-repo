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

CROPSZ = 23

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --   Get individual intersection crops for NN training material
    Synopsis:
      %s --infolder <ifolder> --outfolder <ofolder>
    Description:
      Scales and perspective transforms input images, save 361 crops for each.
    Example:
      %s --infolder ~/training_data_kc/mike/20180113/verified --outfolder crops-kc
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

# Collect matching jpeg and json in a dictionary
#----------------------------------------------------
def collect_files( infolder):
    # Find images
    imgs = ut.find( infolder, '*.jpeg')
    imgs += ut.find( infolder, '*.jpg')
    # Basenames
    basenames = [os.path.basename(f) for f in imgs]
    basenames = [os.path.splitext(f)[0] for f in basenames]
    # json files
    jsons = [ut.find( infolder, '%s_intersections.json' % f)[0] for f in basenames]
    # Collect in dictionary
    files = {}
    for i,bn in enumerate( basenames):
        d = {}
        files[bn] = d
        d['img'] = imgs[i]
        d['json'] = jsons[i]
    # Sanity check
    for bn in files.keys():
        d = files[bn]
        if not bn in d['img']:
            print( 'ERROR: Wrong img name for key %s' % (d['img'], bn))
            exit(1)
        elif not bn in d['json']:
            print( 'ERROR: Wrong json name for key %s' % (d['json'], bn))
            exit(1)
    return files

# Rescale image and intersections to width 350.
# Make the board square.
# Returns the new image and the transformed list of intersections like
# [{u'y': 17, u'x': 17, u'val': u'EMPTY'}, ...
#------------------------------------------------
def zoom_in( imgfile, jsonfile):
    # Read the image
    img = cv2.imread( imgfile, 1)
    # Parse json
    columns = json.load( open( jsonfile))
    # Linearize
    board_sz = len(columns)
    intersections = [0] * (board_sz * board_sz)
    for c,col in enumerate( columns):
        for r, row in enumerate( col):
            idx = board_sz * r + c
            intersections[idx] = row
    #         x = row['x']
    #         y = row['y']
    #         cv2.circle( img,
    #                     (int(x), int(y)),
    #                     10,
    #                     ( 0, 0, 255 ),
    #                     -1 )
    # cv2.imwrite( '/home/ubuntu/ahn-repo/howard_deep/proj/15_threeclasses/tt.jpg', img);

    # Perspective transform
    #-------------------------
    # Corners
    tl = intersections[0]
    tr = intersections[board_sz-1]
    br = intersections[board_sz * board_sz - 1]
    bl = intersections[board_sz * board_sz - board_sz]
    corners = np.array([
        [tl['x'], tl['y']],
        [tr['x'], tr['y']],
        [br['x'], br['y']],
        [bl['x'], bl['y']]], dtype = "float32")

    WIDTH = 350
    marg = WIDTH / 20.0;
    # Target square for transform
    square = np.array([
        [marg, marg],
        [WIDTH - marg, marg],
        [WIDTH - marg, WIDTH - marg],
        [marg, WIDTH - marg]], dtype = "float32")
    M = cv2.getPerspectiveTransform( corners, square)
    warped_img = cv2.warpPerspective( img, M, (WIDTH, WIDTH))

    coords = []
    for isec in intersections:
        coords.append( [isec['x'], isec['y']])
    coords = np.array( coords)
    # Transform the intersections
    # This needs a stupid empty dimension added
    sz = len(coords)
    coords_zoomed = cv2.perspectiveTransform( coords.reshape( 1, sz, 2).astype('float32'), M)
    # And now get rid of the extra dim and back to int
    coords_zoomed = coords_zoomed.reshape(sz,2).astype('int')
    # Back to the old format
    intersections_zoomed = []
    for idx,isec in enumerate( intersections):
        intersections_zoomed.append( isec.copy())
        nnew = intersections_zoomed[-1]
        nnew['x'] = coords_zoomed[idx][0]
        nnew['y'] = coords_zoomed[idx][1]
    res = (warped_img, intersections_zoomed)
    return res

# Save intersection crops of size rxr
#-------------------------------------------------------------------
def save_intersections( img, intersections, r, basename, folder):
    dx = int(r / 2)
    dy = int(r / 2)
    for i,isec in enumerate( intersections):
        color = isec['val'][0]
        x = isec['x']
        y = isec['y']
        hood = img[y-dy:y+dy+1, x-dx:x+dx+1]
        fname = "%s/%s_rgb_%s_hood_%03d.jpg" % (folder, color, basename, i)
        cv2.imwrite( fname, hood)

#-----------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument( "--infolder",      required=True)
    parser.add_argument( "--outfolder",     required=True)
    args = parser.parse_args()

    os.makedirs( args.outfolder)
    files = collect_files( args.infolder)

    for i,k in enumerate( files.keys()):
        print( '%s ...' % k)
        f = files[k]
        img, intersections = zoom_in( f['img'], f['json'])
        if len(intersections) != 19*19:
            print( 'not a 19x19 board, skipping')
            continue
        save_intersections( img, intersections, CROPSZ, k, args.outfolder)


if __name__ == '__main__':
    main()
