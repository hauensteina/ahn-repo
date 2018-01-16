#!/usr/bin/env python

# /********************************************************************
# Filename: 01_persp.py
# Author: AHN
# Creation Date: Jan, 2018
# **********************************************************************/
#
# Experiments with perspective transform
#

from __future__ import division, print_function
from pdb import set_trace as BP
import inspect
import os,sys,re,json,shutil,glob
import numpy as np
import scipy.signal
from numpy.random import random
import argparse
import cv2
from matplotlib import pyplot as plt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahn_opencv_util as ut
g_frame=''

np.set_printoptions(linewidth=np.nan)


#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Perspective transform experiments
    Synopsis:
      %s --run
    Description:
      Perform perspective transform, look for invariants.
    Example:
      %s --run
    ''' % (name,name,name)
    if printmsg:
        print(msg)
        exit(1)
    else:
        return msg

#---------------------
def main():
    if len(sys.argv) == 1:
        usage(True)

    parser = argparse.ArgumentParser(usage=usage())
    #parser.add_argument("--img",    required=False, default=0, type=int)
    #parser.add_argument("--folder", required=True, type=str)
    #parser.add_argument("--squares", required=False, action='store_true')
    parser.add_argument("--run", required=False, action='store_true')
    args = parser.parse_args()

    # An RxR grid with 100 pixels spacing
    R = 20
    grid = np.zeros( (R,R,2), np.float32)
    for x in range(0,R):
        for y in range(0,R):
            grid[x,y] = (100 * x, 100 * y)

    # Where to warp the for points closest to (0,0)
    src = np.array([ [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0,1.0] ], dtype = "float32") * 100.0
    dst = np.array([ [0.0, 0.0], [1.0, 0.0], [0.1, 0.9], [0.9,0.85] ], dtype = "float32")
    dst += (0.5, 0.25)
    dst *= 100


    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    grid_warped = cv2.perspectiveTransform( grid, M)

    # Draw
    height = 100 * R; width = 100 * R
    img = np.full( (height, width, 3), 255, dtype=np.uint8)
    for p in grid.reshape( grid.shape[0]*grid.shape[1],2):
        cv2.circle( img, (p[0],p[1]), 3, (0,255,0), thickness=-1)
    for p in grid_warped.reshape( grid.shape[0]*grid.shape[1],2):
        cv2.circle( img, (p[0],p[1]), 2, (255,0,0), thickness=-1)

    # Test for invariants
    #=========================
    gw = grid_warped

    # Vanishing points
    print( '\nVanishing points:')
    for x in range(0,R-1):
        p = ut.intersection( [gw[x,0], gw[x,4]], [gw[x+1,0], gw[x+1,4]])
        print('vp %d%d: (%.2f, %.2f)' % (x,x+1,p[0],p[1]))
    p = ut.intersection( [gw[0,0], gw[0,4]], [gw[4,0], gw[4,4]])
    print('vp %d%d: (%.2f, %.2f)' % (0,4,p[0],p[1]))

    # delta x values
    print( '\nDelta x:')
    delta_x = np.zeros( (R,R), np.float32)
    for x in range( 1, R):
        for y in range( 0,R):
            delta_x[x,y] = cv2.norm( gw[x,y] - gw[x-1,y])
    print( delta_x.transpose())
    #print( delta_x)

    # delta y values
    print( '\nDelta y:')
    delta_y = np.zeros( (R,R), np.float32)
    for y in range( 1, R):
        for x in range( 0,R):
            delta_y[x,y] = cv2.norm( gw[x,y] - gw[x,y-1])
    print( delta_y.transpose())
    #print( delta_y)

    # x shrinkage
    print( '\nx shrinkage left to right:')
    for x in range(1,R):
        shrinkrat = delta_x[x,4] / delta_x[x,0]
        print( '%d: %.4f' % (x, shrinkrat))

    # y shrinkage
    print( '\ny shrinkage bottom to top:')
    for y in range(1,R):
        shrinkrat = delta_y[4,y] / delta_y[0,y]
        print( '%d: %.4f' % (y, shrinkrat))

    # dleft
    dleft = np.zeros( (R,R), np.float32)
    for x in range( 1,R):
        for y in range( 0, R):
            dleft[x,y] = cv2.norm( gw[x,y] - gw[x-1,y])

    # dbot
    dbot  = np.zeros( (R,R), np.float32)
    for y in range( 1,R):
        for x in range( 0, R):
            dbot[x,y] = cv2.norm( gw[x,y] - gw[x, y-1])

    # dleft / dbot
    dleftbot  = np.zeros( (R,R), np.float32)
    for y in range( 1,R):
        for x in range( 1, R):
            dleftbot[x,y] = dleft[x,y] / dbot[x,y]

    for x in range( 1,R):
        leftshrink = dleft[x,1] / dleft[x,R-1]
        botshrink = dbot[x,1] / dbot[x,R-1]
        print( 'x: %d leftshrink: %.2f botshrink: %.2f' % (x, leftshrink, botshrink))

    # Print various
    #===================
    for y in range(R):
        print( gw[0,y][1])

    #BP()
    # Show
    plt.figure( figsize=( 10, 10))
    plt.subplot( 1,1,1);
    plt.imshow( img, origin='lower');
    plt.show()

if __name__ == '__main__':
    main()
