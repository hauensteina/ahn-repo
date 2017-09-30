#!/usr/bin/env python

# /********************************************************************
# Filename: generate_images.py
# Author: AHN
# Creation Date: Sep 21, 2017
# **********************************************************************/
#
# Generate training and validation data for project gcount
#

from __future__ import division,print_function
from pdb import set_trace as BP
import os,sys,re,json
import numpy as np
#from numpy.random import random
import argparse
import matplotlib as mpl
mpl.use('Agg') # This makes matplotlib work without a display
from matplotlib import pyplot as plt

# Look for modules in our pylib folder
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(re.sub(r'/proj/.*',r'/pylib', SCRIPTPATH))
import ahnutil as ut

EMPTY = 0
WHITE = 1
BLACK = 2

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Generate training and validation data
    Synopsis:
      %s --resolution <n> --gridsize <n> --ntrain <n> --nval <n>
    Description:
      Generates jpegs in subfolders train and val, plus labels in json files.
      Each image has a random number of B,W, and empty intersections.
    Example:
      %s --resolution 80 --gridsize 5 --ntrain 1000 --nval 100
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
    parser.add_argument( "--resolution",      required=True, type=int)
    parser.add_argument( "--gridsize", required=True, type=int)
    parser.add_argument( "--ntrain",   required=True, type=int)
    parser.add_argument( "--nval",     required=True, type=int)
    args = parser.parse_args()
    #if (args.resolution % (2*args.gridsize)): usage(True)
    #np.random.seed(0) # Make things reproducible
    trainfolder = 'train/all_files'
    valfolder   = 'valid/all_files'
    if not os.path.exists(trainfolder): os.makedirs(trainfolder)
    if not os.path.exists(valfolder):   os.makedirs(valfolder)
    gen_images(args.ntrain, args.resolution, args.gridsize, trainfolder)
    gen_images(args.nval,   args.resolution, args.gridsize, valfolder)

# Given an int in range(gridsize*gridsize),
# return (x,y,r) to draw circle in matplotlib
#-----------------------------------------------
def lin2xyr(linpos, gridsize, resolution):
    p = (linpos // gridsize, linpos % gridsize)
    r = resolution // (2*gridsize)
    p = (p[1]*2*r + r, p[0]*2*r + r)
    x,y,r = p[0]/resolution, p[1]/resolution, 0.8*r/resolution
    return x,y,r

# Generate one image of resolution res x res
# with a random pattern of B,W,and empty intersections
#---------------------------------------------------------
def gen_image(resolution,gridsize,ofname):
    # Random stone pattern
    sz = gridsize*gridsize
    lin = [BLACK] * sz + [WHITE] * sz + [EMPTY] * sz
    np.random.shuffle(lin)
    lin = lin[:sz]

    # Set up matplotlib
    dpi=100.0
    fig = plt.figure(figsize=(resolution/dpi,resolution/dpi),dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    for linpos,col in enumerate(lin):
        x,y,r = lin2xyr(linpos, gridsize, resolution)
        if col in (BLACK,WHITE):
            circle = plt.Circle((x,y), r)
            circle.set_edgecolor('k')
            circle.set_facecolor('none' if col == WHITE else 'k')
            ax.add_artist(circle)
    plt.savefig(ofname)
    # Now matrix rows go from top to bottom, image coordinates go
    # from bottom to top. Therefore we need to reverse the rows.
    # Otherwise, our poor convolutions will have to learn how to turn
    # the image upside down.
    lin = np.array(lin)
    res = np.flipud(lin.reshape(gridsize,gridsize)).reshape(len(lin))
    tt = list(res)
    return list(res)

# Generate nb_imgs images with one massive or hollow circle.
# Also generate a json file for each, giving the bounding box and class.
#--------------------------------------------------------------------------
def gen_images(nb_imgs,resolution,gridsize,folder):
    #BP()
    for i in range(nb_imgs):
        if i and not i%100: print(i)
        fname = '%07d' % i
        fjpg  = folder + '/' + fname + '.jpg'
        fjson = folder + '/' + fname + '.json'
        stones = gen_image(resolution, gridsize, fjpg)
        #stones = ut.onehot(stones).tolist()
        plt.close('all')
        # Dump metadata
        meta = { 'stones':stones }
        meta_json = json.dumps(meta) + '\n'
        with open(fjson,'w') as f: f.write(meta_json)


if __name__ == '__main__':
    main()
