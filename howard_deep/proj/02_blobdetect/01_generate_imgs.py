#!/usr/bin/env python

# /********************************************************************
# Filename: generate_images.py
# Author: AHN
# Creation Date: Aug 26, 2017
# **********************************************************************/
#
# Generate training and validation data for project blobcount
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

#---------------------------
def usage(printmsg=False):
    name = os.path.basename(__file__)
    msg = '''
    Name:
      %s --  Generate training and validation data for project blobcount
    Synopsis:
      %s --res <n> --minblobs <n> --maxblobs <n> --ntrain <n> --nval <n>
    Description:
      Generates jpegs in subfolders train and val, plus labels in json files.
      Each image has between minblobs and maxblobs black circles in it.
    Example:
      %s --res 128 --minblobs 0 --maxblobs 1 --ntrain 1000 --nval 100
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
    parser.add_argument( "--res",      required=True, type=int)
    parser.add_argument( "--minblobs", required=True, type=int)
    parser.add_argument( "--maxblobs", required=True, type=int)
    parser.add_argument( "--ntrain",   required=True, type=int)
    parser.add_argument( "--nval",     required=True, type=int)
    args = parser.parse_args()
    #np.random.seed(0) # Make things reproducible
    trainfolder = 'train/all_files'
    valfolder   = 'valid/all_files'
    if not os.path.exists(trainfolder): os.makedirs(trainfolder)
    if not os.path.exists(valfolder):   os.makedirs(valfolder)
    gen_images(args.ntrain, args.res, args.minblobs, args.maxblobs, trainfolder)
    gen_images(args.nval,   args.res, args.minblobs, args.maxblobs, valfolder)

# Generate one image of resolution resxres with nblobs circles in it.
# Image goes to folder/fname
#----------------------------------------
def gen_image(res,nblobs,ofname):
    # Set up matplotlib
    dpi=100.0
    fig = plt.figure(figsize=(res/dpi,res/dpi),dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    centers_x = np.random.uniform(0,1,nblobs)
    centers_y = np.random.uniform(0,1,nblobs)
    r = np.random.uniform(0.1,0.2,nblobs)

    for i in range(nblobs):
        circle = plt.Circle((centers_x[i], centers_y[i]), r[i], color='k')
        ax.add_artist(circle)
    plt.savefig(ofname)
    return (list(centers_x), list(centers_y), list(r))

# Generate nb_imgs images with minblobs to maxblobs circles.
# Also generate a json file for each, giving the number of circles.
#-------------------------------------------------------------------
def gen_images(nb_imgs,res,minblobs,maxblobs,folder):
    #BP()
    nblobs = minblobs + np.random.randint(maxblobs-minblobs+1, size=nb_imgs)
    for i in range(nb_imgs):
        fname = '%07d' % i
        fjpg  = folder + '/' + fname + '.jpg'
        fjson = folder + '/' + fname + '.json'
        # Make jpeg
        (x,y,r) = gen_image(res, nblobs[i], fjpg)
        plt.close('all')
        # Dump metadata
        #meta = { 'centers_x':x, 'centers_y':y, 'radii':r }
        meta = { 'class':nblobs[i] }
        #BP()
        meta_json = json.dumps(meta) + '\n'
        with open(fjson,'w') as f: f.write(meta_json)



if __name__ == '__main__':
    main()
